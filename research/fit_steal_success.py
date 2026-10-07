import json
import platform
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata

T0 = time.time()
STAGES = {}
ROOT = Path(__file__).resolve().parent.parent
FIT = ROOT / "data" / "derived" / "steal" / "steal_fit_table_2023_2025.parquet"
ATTRS = ROOT / "data" / "derived" / "steal" / "player_season_attributes_2023_2025.parquet"
PLAYS = ROOT / "data" / "derived" / "plays_regular_2016_2025.parquet"
PITCHER_FILLS = ["pitcher_primary_lead", "pitcher_lead_gained"]  # filled by season and pitcher hand
SAVANT_RUNNER = ROOT / "data" / "raw" / "savant" / "runner_basestealing_2b_{}.csv"
OUT = ROOT / "data" / "derived" / "steal"
SEED = 2026
TRAIN, TEST, ALL = [2023, 2024], [2025], [2023, 2024, 2025]
GRID = np.logspace(0, 3.5, 8)  # candidate ridge penalties per player group
N_FOLDS = 5
N_BOOT = 200
FIXED_RIDGE = 1e-4  # tiny ridge on fixed effects (not the intercept) for numerical stability
ROLES = ["runner", "catcher", "pitcher"]
THETA_BOUNDS = (-20.0, 5.0)  # log slope per standard deviation for constrained terms
LEAK_GAP = 0.002

# Continuous attribute on the fit table -> (role, column in the player-season attribute table)
ATTR_SOURCE = {"sprint_speed": ("runner", "runner_sprint_speed"), "runner_lead_gained_opp": ("runner", "runner_lead_gained_opp"),
               "runner_aggressiveness": ("runner", "runner_agg_shrunk"), "pop_time": ("catcher", "catcher_pop_2b"),
               "exchange": ("catcher", "catcher_exchange_2b"), "arm_strength": ("catcher", "catcher_arm_2b"),
               "pitcher_primary_lead": ("pitcher", "pitcher_primary_lead"), "pitcher_lead_gained": ("pitcher", "pitcher_lead_gained")}
LAGGED = ["sprint_speed", "pop_time", "pitcher_primary_lead", "pitcher_lead_gained", "runner_aggressiveness"]
BASE_CON = [("sprint_speed", 1), ("pop_time", 1), ("pitcher_primary_lead", 1), ("pitcher_lead_gained", 1)]
DROPPED_OPP = ("runner lead gained on opportunities (runner_lead_gained_opp) is dropped from every model and kept on the fit table: "
               "its estimate was indistinguishable from zero (0.030 per ft, SE 0.030, all three seasons)")


def spec(name, constrained, players):
    return {"name": name, "constrained": constrained, "free_cont": ["runner_aggressiveness"], "players": players}


SPECS = {
    "A32": spec("A32", BASE_CON, players=False),
    "B32": spec("B32", BASE_CON, players=True),
    "B-arm32": spec("B-arm32", [c for c in BASE_CON if c[0] != "pop_time"] + [("arm_strength", -1), ("exchange", 1)], players=True),
}

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_rows", 200)

print("the attempts version of runner lead gained is removed from every model: it is a season average over the runner's own attempts, "
      "so it contains the outcomes being predicted")
print("missing indicators and their ridge are removed; filled attributes keep the season-mean fill only")
print(DROPPED_OPP)
fit_table = pd.read_parquet(FIT)
fit_table["pitchout"] = fit_table["pitchout"].astype(int)
fit_table["count"] = fit_table["balls"].astype(str) + "-" + fit_table["strikes"].astype(str)
fit_table["count_3_2"] = (fit_table["count"] == "3-2").astype(int)
c32 = fit_table[fit_table["count_3_2"] == 1]
print(f"3-2 throw situations: {len(c32):,}; pitch result among them: {c32['pitch_result'].value_counts().to_dict()} - every 3-2 throw situation is a "
      "strike-three throw (ball four and fouls are not throw situations), so the count=3-2 term is the 3-2 strike-three effect")
attrs = pd.read_parquet(ATTRS)
train = fit_table[fit_table["season"].isin(TRAIN)].reset_index(drop=True)
test = fit_table[fit_table["season"].isin(TEST)].reset_index(drop=True)
y_train, y_test = train["throw_safe"].to_numpy(float), test["throw_safe"].to_numpy(float)
print(f"fit table: {len(fit_table):,} throw situations; train {TRAIN} {len(train):,}, test {TEST} {len(test):,}")


def prepare(sp, df):
    cont = [c for c, _ in sp["constrained"]] + sp["free_cont"]
    st = {"spec": sp, "con": sp["constrained"], "free": sp["free_cont"], "mean": {c: float(df[c].mean()) for c in cont},
          "sd": {c: float(df[c].std()) for c in cont}, "seasons": sorted(df["season"].unique().tolist())}
    st["players"] = {r: {p: i for i, p in enumerate(sorted(df[f"{r}_id"].unique()))} for r in ROLES} if sp["players"] else {r: {} for r in ROLES}
    return st


def build(df, st):
    n = len(df)
    z = lambda c: ((df[c] - st["mean"][c]) / st["sd"][c]).to_numpy(float)  # noqa: E731
    cols, names = [np.ones(n)], ["intercept"]
    for c in st["free"]:
        cols.append(z(c))
        names.append(c)
    for name, values in [("pitch_result=strike", df["pitch_result"] == "strike"), ("pitch_result=strike_three", df["pitch_result"] == "strike_three"),
                         ("count=3-2", df["count_3_2"] == 1), ("pickoff_throws=1", df["pickoff_throws"] == "1"),
                         ("pickoff_throws=2+", df["pickoff_throws"] == "2+"), ("pitchout", df["pitchout"] == 1),
                         ("pitcher_hand=L", df["pitcher_hand"] == "L"), ("bat_side=L", df["bat_side"] == "L")]:
        cols.append(values.to_numpy(float))
        names.append(name)
    seen = df["season"].isin(st["seasons"]).to_numpy()
    for s in st["seasons"][1:]:
        # an unseen season gets the average of the training seasons' effects
        cols.append(np.where(seen, (df["season"] == s).to_numpy(float), 1.0 / len(st["seasons"])))
        names.append(f"season={s}")
    xc = np.column_stack([z(c) for c, _ in st["con"]])
    blocks, sizes = [], []
    for r in ROLES:
        idx = df[f"{r}_id"].map(st["players"][r])
        ok = idx.notna().to_numpy()
        size = len(st["players"][r])
        blocks.append(sparse.csr_matrix((np.ones(ok.sum()), (np.flatnonzero(ok), idx[ok].astype(int).to_numpy())), shape=(n, size)))
        sizes.append(size)
    return {"Xf": np.column_stack(cols), "Xc": xc, "sign": np.array([s for _, s in st["con"]], float), "Z": sparse.hstack(blocks).tocsr(),
            "names": names, "con_names": [c for c, _ in st["con"]], "sizes": sizes}


def unpack(w, d):
    nf, nc = d["Xf"].shape[1], d["Xc"].shape[1]
    return w[:nf], w[nf:nf + nc], w[nf + nc:]


def eta_of(w, d):
    bf, th, u = unpack(w, d)
    return d["Xf"] @ bf + d["Xc"] @ (d["sign"] * np.exp(th)) + d["Z"] @ u


def fit(d, y, lam, x0=None):
    lam_vec = np.concatenate([np.full(s, l) for s, l in zip(d["sizes"], lam)])
    nf, nc = d["Xf"].shape[1], d["Xc"].shape[1]
    ridge = np.array([0.0 if nm == "intercept" else FIXED_RIDGE for nm in d["names"]])

    def obj(w):
        bf, th, u = unpack(w, d)
        slope = d["sign"] * np.exp(th)
        eta = d["Xf"] @ bf + d["Xc"] @ slope + d["Z"] @ u
        g = expit(eta) - y
        f = np.sum(np.logaddexp(0, eta) - y * eta) + 0.5 * np.sum(lam_vec * u * u) + 0.5 * np.sum(ridge * bf * bf)
        grad = np.concatenate([d["Xf"].T @ g + ridge * bf, (d["Xc"].T @ g) * slope, d["Z"].T @ g + lam_vec * u])
        return f, grad

    if x0 is None:
        x0 = np.zeros(nf + nc + d["Z"].shape[1])
        x0[0] = np.log(y.mean() / (1 - y.mean()))
        x0[nf:nf + nc] = np.log(0.05)
    bounds = [(None, None)] * nf + [THETA_BOUNDS] * nc + [(None, None)] * d["Z"].shape[1]
    return minimize(obj, x0, jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": 5000}).x


def log_loss(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def auc(y, p):
    r = rankdata(p)
    n1 = y.sum()
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def metrics(y, p):
    return {"log_loss": log_loss(y, p), "brier": float(np.mean((p - y) ** 2)), "auc": auc(y, p)}


def calibration(y, p):
    t = pd.DataFrame({"pred": p, "obs": y, "decile": pd.qcut(p, 10, labels=False, duplicates="drop")})
    return t.groupby("decile").agg(mean_predicted=("pred", "mean"), observed=("obs", "mean"), n=("obs", "size"))


def fit_predict(sp, tr, te, lam):
    st = prepare(sp, tr)
    d_tr, d_te = build(tr, st), build(te, st)
    w = fit(d_tr, tr["throw_safe"].to_numpy(float), lam)
    return st, d_tr, d_te, w, expit(eta_of(w, d_te))


rng = np.random.default_rng(SEED)
games = np.array(sorted(train["gid"].unique()))
fold_of_game = dict(zip(rng.permutation(games), np.arange(len(games)) % N_FOLDS))
fold = train["gid"].map(fold_of_game).to_numpy()
folds = [(np.flatnonzero(fold != k), np.flatnonzero(fold == k)) for k in range(N_FOLDS)]
print(f"cross-validation: {N_FOLDS} folds by game over {len(games):,} training games; fold sizes {[len(v) for _, v in folds]}")


def cv_loss(sp, lam, warm):
    total = 0.0
    for k, (tr_idx, va_idx) in enumerate(folds):
        tr, va = train.iloc[tr_idx], train.iloc[va_idx]
        st = prepare(sp, tr)
        d_tr, d_va = build(tr, st), build(va, st)
        w = fit(d_tr, y_train[tr_idx], lam, warm.get(k))
        warm[k] = w
        total += log_loss(y_train[va_idx], expit(eta_of(w, d_va))) * len(va_idx)
    return total / len(train)


def search_penalties(sp):
    cache, warm = {}, {}

    def ev(lam):
        key = tuple(float(x) for x in lam)
        if key not in cache:
            cache[key] = cv_loss(sp, key, warm)
        return cache[key]

    diag = [ev((g, g, g)) for g in GRID]
    best = [float(GRID[int(np.argmin(diag))])] * 3
    for cycle in range(4):
        changed = False
        for i in range(3):
            vals = []
            for g in GRID:
                lam = list(best)
                lam[i] = float(g)
                vals.append(ev(lam))
            new = float(GRID[int(np.argmin(vals))])
            if new != best[i]:
                best[i], changed = new, True
        if not changed:
            break
    surface = {}
    for i, r in enumerate(ROLES):
        row = []
        for g in GRID:
            lam = list(best)
            lam[i] = float(g)
            row.append(ev(lam))
        surface[r] = row
    return best, diag, surface, len(cache)


print("\n=== penalty search for B32 (game-split five-fold CV log loss, coordinate-wise over the grid)")
t = time.time()
lam_b, diag, surface, n_evals = search_penalties(SPECS["B32"])
STAGES["penalty search"] = time.time() - t
print(f"grid: {', '.join(f'{g:.1f}' for g in GRID)}")
print("all three equal (starting diagonal): " + ", ".join(f"{g:.1f}: {v:.5f}" for g, v in zip(GRID, diag)))
print(f"chosen penalties: runner {lam_b[0]:.1f}, catcher {lam_b[1]:.1f}, pitcher {lam_b[2]:.1f} ({n_evals} penalty settings evaluated; B-arm32 reuses them)")
surf = pd.DataFrame(surface, index=[f"{g:.1f}" for g in GRID])
surf.index.name = "penalty"
print("CV log loss along each coordinate, other two held at the optimum:")
print(surf.to_string(float_format="{:.5f}".format))

print("\n=== models trained on 2023-2024, tested on 2025")
t = time.time()
results, preds, fits = {}, {}, {}
base_rate = y_train.mean()
for name, sp in SPECS.items():
    lam = lam_b if sp["players"] else (1.0, 1.0, 1.0)
    st, d_tr, d_te, w, p = fit_predict(sp, train, test, lam)
    fits[name] = (st, d_tr, d_te, w)
    preds[name] = p
    results[name] = {**metrics(y_test, p), "train_log_loss": log_loss(y_train, expit(eta_of(w, d_tr)))}
STAGES["logistic models"] = time.time() - t

t = time.time()
c_cont = [c for c, _ in BASE_CON] + ["runner_aggressiveness"]
mono = dict(BASE_CON)
cat_maps = {f"{r}_id": {p: i for i, p in enumerate(sorted(train[f"{r}_id"].unique()))} for r in ROLES}


def lgb_frame(df):
    out = pd.DataFrame({c: df[c].to_numpy(float) for c in c_cont})
    out["pitch_result"] = df["pitch_result"].map({"ball": 0, "strike": 1, "strike_three": 2}).to_numpy(float)
    out["count_3_2"] = df["count_3_2"].to_numpy(float)
    out["pickoff_throws"] = df["pickoff_throws"].map({"0": 0, "1": 1, "2+": 2}).to_numpy(float)
    out["pitchout"] = df["pitchout"].to_numpy(float)
    out["pitcher_hand_L"] = (df["pitcher_hand"] == "L").to_numpy(float)
    out["bat_side_L"] = (df["bat_side"] == "L").to_numpy(float)
    out["season"] = df["season"].map({s: i for i, s in enumerate(TRAIN)}).to_numpy(float)
    for col, mp in cat_maps.items():
        out[col] = df[col].map(mp).to_numpy(float)  # unseen players are missing
    return out


X_lgb_tr, X_lgb_te = lgb_frame(train), lgb_frame(test)
cats = ["pitch_result", "season"] + list(cat_maps)
mono_list = [mono.get(c, 0) for c in X_lgb_tr.columns]
grid_rows, best_c = [], None
for leaves in (4, 8, 16):
    for min_leaf in (50, 100, 200):
        params = {"objective": "binary", "metric": "binary_logloss", "learning_rate": 0.03, "num_leaves": leaves, "min_data_in_leaf": min_leaf,
                  "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "monotone_constraints": mono_list,
                  "seed": SEED, "deterministic": True, "force_row_wise": True, "verbose": -1, "num_threads": 4}
        ds = lgb.Dataset(X_lgb_tr, label=y_train, categorical_feature=cats, free_raw_data=False)
        cvr = lgb.cv(params, ds, num_boost_round=3000, folds=folds, stratified=False, callbacks=[lgb.early_stopping(100, verbose=False)])
        key = [k for k in cvr if k.endswith("binary_logloss-mean")][0]
        loss, rounds = float(cvr[key][-1]), len(cvr[key])
        grid_rows.append({"num_leaves": leaves, "min_data_in_leaf": min_leaf, "cv_log_loss": loss, "rounds": rounds})
        if best_c is None or loss < best_c[0]:
            best_c = (loss, params, rounds)
booster = lgb.train(best_c[1], lgb.Dataset(X_lgb_tr, label=y_train, categorical_feature=cats), num_boost_round=best_c[2])
preds["C"] = booster.predict(X_lgb_te)
results["C"] = {**metrics(y_test, preds["C"]), "train_log_loss": log_loss(y_train, booster.predict(X_lgb_tr))}
STAGES["lightgbm"] = time.time() - t
print("LightGBM grid (same game-split folds, early stopping at 100 rounds, count=3-2 included):")
print(pd.DataFrame(grid_rows).to_string(index=False, float_format="{:.5f}".format))
print(f"chosen: num_leaves {best_c[1]['num_leaves']}, min_data_in_leaf {best_c[1]['min_data_in_leaf']}, {best_c[2]} rounds")

base = np.full(len(y_test), base_rate)
print(f"\n=== 2025 test comparison (constant baseline = training mean {base_rate:.4f})")
comp = pd.DataFrame(results).T
comp.loc["baseline (constant)"] = {"log_loss": log_loss(y_test, base), "brier": float(np.mean((base - y_test) ** 2)), "auc": np.nan,
                                   "train_log_loss": log_loss(y_train, np.full(len(y_train), base_rate))}
print(comp[["log_loss", "brier", "auc", "train_log_loss"]].to_string(float_format="{:.5f}".format))
for name in ["A32", "B32"]:
    print(f"\ncalibration by predicted decile, {name}:")
    print(calibration(y_test, preds[name]).to_string(float_format="{:.3f}".format))

print("\n=== lagged-attribute check (continuous attributes from the player's previous season)")
t = time.time()
season_means = attrs.groupby("season")[[src for _, src in ATTR_SOURCE.values()]].mean()
# Pitcher hand per pitcher-season: the hand he threw with most often in the Retrosheet plays.
hands = pd.read_parquet(PLAYS, columns=["season", "pitcher", "pithand"])
hands = hands[hands["season"].isin(ALL)].groupby(["pitcher", "season"])["pithand"].agg(lambda h: h.value_counts().index[0])
attrs["pitcher_hand"] = [hands.get((r, s)) for r, s in zip(attrs["retro_id"], attrs["season"])]
pitcher_rows = attrs[attrs[ATTR_SOURCE["pitcher_primary_lead"][1]].notna()]
hand_means = pitcher_rows.groupby(["season", "pitcher_hand"])[[ATTR_SOURCE[c][1] for c in PITCHER_FILLS]].mean()
hand_counts = pitcher_rows.groupby(["season", "pitcher_hand"]).size()
n_no_hand = int(pitcher_rows["pitcher_hand"].isna().sum())


def lag(df, cols):
    out = df.copy()
    filled = {}
    for c in cols:
        role, src = ATTR_SOURCE[c]
        prev = attrs.assign(season=attrs["season"] + 1).set_index(["retro_id", "season"])[src]
        v = prev.reindex(pd.MultiIndex.from_arrays([df[f"{role}_id"], df["season"]])).to_numpy(float)
        fill_vals = df["season"].map(lambda s: season_means.loc[s - 1, src] if s - 1 in season_means.index else np.nan).to_numpy(float)
        miss = np.isnan(v)
        out[c] = np.where(miss, fill_vals, v)
        filled[c] = int(miss.sum())
    return out, filled


tr24 = train[train["season"] == 2024].reset_index(drop=True)
tr24_lag, filled_tr = lag(tr24, LAGGED)
te_lag, filled_te = lag(test, LAGGED)
print("rows filled with the previous season's league mean (player has no previous-season value):")
print(pd.DataFrame({"2024 training rows (2023 values)": filled_tr, "2025 test rows (2024 values)": filled_te}).to_string())
print(f"2024 training rows: {len(tr24):,}; 2025 test rows: {len(test):,}")
lag_rows = []
gaps = {}
for name in ["A32", "B32"]:
    sp = SPECS[name]
    lam = lam_b if sp["players"] else (1.0, 1.0, 1.0)
    same24 = fit_predict(sp, tr24, test, lam)[4]
    lagged_all = fit_predict(sp, tr24_lag, te_lag, lam)[4]
    lag_rows.append({"model": name, "same-season, trained 2023-2024": results[name]["log_loss"], "same-season, trained 2024 only": log_loss(y_test, same24),
                     "all lagged, trained 2024 only": log_loss(y_test, lagged_all)})
    for c in LAGGED:
        tr_one, _ = lag(tr24, [c])
        te_one, _ = lag(test, [c])
        gaps[(name, c)] = log_loss(y_test, fit_predict(sp, tr_one, te_one, lam)[4]) - log_loss(y_test, same24)
STAGES["lagged check"] = time.time() - t
print("2025 log loss, same rows (all 2025 throw situations):")
print(pd.DataFrame(lag_rows).set_index("model").to_string(float_format="{:.5f}".format))
gap_tab = pd.Series(gaps).unstack(0)
print("change in 2025 log loss when one attribute is lagged (lagged minus same-season, both trained on 2024):")
print(gap_tab.to_string(float_format="{:+.5f}".format))
leaks = [f"{c} ({m})" for (m, c), g in gaps.items() if g > LEAK_GAP]
print("leak check (gap above 0.002): " + (", ".join(leaks) + " - the same-season value predicts better than last season's by more than 0.002"
      if leaks else "no attribute's same-season value appears to leak"))
print("  caveat: a previous-season value is also a noisier measure of current skill, so a small positive gap is expected without any leak")


def constraint_table(st, d, w):
    bf, th, u = unpack(w, d)
    rows = []
    for nm, b in zip(d["names"], bf):
        if nm in st["mean"]:
            rows.append({"term": nm, "estimate": b / st["sd"][nm], "per_sd": b, "constraint": "none", "status": "free", "center": st["mean"][nm]})
        else:
            rows.append({"term": nm, "estimate": b, "per_sd": np.nan, "constraint": "none", "status": "free", "center": np.nan})
    for nm, sgn, t_ in zip(d["con_names"], d["sign"], th):
        slope = sgn * np.exp(t_)
        rows.append({"term": nm, "estimate": slope / st["sd"][nm], "per_sd": slope, "constraint": ">= 0" if sgn > 0 else "<= 0",
                     "status": "at bound (about 0)" if abs(slope) < 1e-3 else "interior", "center": st["mean"][nm]})
    return pd.DataFrame(rows)


def diagnostics(name):
    st, d_tr, d_te, w = fits[name]
    print(f"\n--- {name}")
    print("coefficients and constraint status (training fit, 2023-2024):")
    print(constraint_table(st, d_tr, w).set_index("term")[["estimate", "per_sd", "constraint", "status", "center"]].to_string(float_format="{:.4f}".format))
    tb = test.assign(pred=preds[name], resid=y_test - preds[name])
    tb["inning_group"] = pd.cut(tb["inning"], [0, 3, 6, 9, 99], labels=["1-3", "4-6", "7-9", "10+"])
    for col in ["count", "outs", "inning_group", "pitch_result", "pickoff_throws"]:
        r = tb.groupby(col, observed=True).agg(n=("resid", "size"), observed=("throw_safe", "mean"), predicted=("pred", "mean"), mean_residual=("resid", "mean"))
        print(f"mean residual (observed minus predicted), 2025, by {col}:")
        print(r.to_string(float_format="{:.4f}".format))
    ob = tb.copy()
    ob["sprint_bin"] = pd.cut(ob["sprint_speed"], [-np.inf, 26, 27, 28, 29, np.inf], labels=["26 and under", "26 to 27", "27 to 28", "28 to 29", "29 and up"])
    ob["pop_tercile"] = pd.qcut(ob["pop_time"], 3, labels=["quick pop", "middle pop", "slow pop"])
    ranges = ", ".join(f"{t}: {g.min():.2f}-{g.max():.2f}" for t, g in ob.groupby("pop_tercile", observed=True)["pop_time"])
    print(f"observed and predicted success by sprint speed bin within pop time terciles, 2025 ({ranges}):")
    tab = ob.groupby(["sprint_bin", "pop_tercile"], observed=False).agg(observed=("throw_safe", "mean"), predicted=("pred", "mean"), n=("pred", "size")).unstack("pop_tercile")
    tab.columns = [f"{t} {s}" for s, t in tab.columns]
    print(tab[[f"{t} {s}" for t in ["quick pop", "middle pop", "slow pop"] for s in ["observed", "predicted", "n"]]].to_string(float_format="{:.3f}".format))
    if st["spec"]["players"]:
        _, _, u = unpack(w, d_tr)
        off = np.cumsum([0] + d_tr["sizes"])
        print("player-effect standard deviations in the training fit (logit scale):")
        for i, r in enumerate(ROLES):
            print(f"  {r}: {d_tr['sizes'][i]:,} players, sd {u[off[i]:off[i + 1]].std():.4f}, penalty {lam_b[i]:.1f}")


print("\n=== diagnostics, 2025")
diagnostics("A32")
diagnostics("B32")

gain = {k: results["A32"]["log_loss"] - results[k]["log_loss"] for k in ["B32", "B-arm32"]}
chosen = "A32"
for k in ["B32", "B-arm32"]:
    if gain[k] > 0.002 and (chosen == "A32" or gain[k] > gain[chosen]):
        chosen = k
print(f"\n=== selection: 2025 log loss gains over A32: B32 {gain['B32']:+.5f}, B-arm32 {gain['B-arm32']:+.5f} (rule: switch only above 0.002)")
print(f"chosen specification: {chosen}. " + ("Neither B32 nor B-arm32 beats A32 by more than 0.002, so A32 is production." if chosen == "A32"
      else f"{chosen} beats A32 by more than 0.002.") + " LightGBM (C) is a benchmark only.")

print(f"\n=== refit of {chosen} on {ALL}, game bootstrap ({N_BOOT} replicates)")
t = time.time()
sp_final = SPECS[chosen]
lam_final = lam_b if sp_final["players"] else None
y_all = fit_table["throw_safe"].to_numpy(float)
st_f = prepare(sp_final, fit_table)
d_f = build(fit_table, st_f)
w_f = fit(d_f, y_all, lam_final or (1.0, 1.0, 1.0))
coef_f = constraint_table(st_f, d_f, w_f)
STAGES["final fit"] = time.time() - t
t = time.time()
all_games = np.array(sorted(fit_table["gid"].unique()))
rows_of_game = pd.Series(np.arange(len(fit_table))).groupby(fit_table["gid"].to_numpy()).apply(np.array)
boot_rng = np.random.default_rng(SEED + 1)
boot = []
for b in range(N_BOOT):
    pick = boot_rng.choice(all_games, size=len(all_games), replace=True)
    idx = np.concatenate([rows_of_game[g] for g in pick])
    d_b = {**d_f, "Xf": d_f["Xf"][idx], "Xc": d_f["Xc"][idx], "Z": d_f["Z"][idx]}
    boot.append(constraint_table(st_f, d_b, fit(d_b, y_all[idx], lam_final or (1.0, 1.0, 1.0), w_f))["estimate"].to_numpy())
coef_f["se"] = np.array(boot).std(axis=0, ddof=1)
STAGES["bootstrap"] = time.time() - t
print(coef_f[["term", "estimate", "se", "constraint", "status", "center"]].to_string(index=False, float_format="{:.4f}".format))
final_metrics = metrics(y_all, expit(eta_of(w_f, d_f)))
print(f"in-sample metrics, all three seasons: log loss {final_metrics['log_loss']:.5f}, Brier {final_metrics['brier']:.5f}, AUC {final_metrics['auc']:.4f}")

written = {}
coef_out = coef_f.rename(columns={"center": "centering_mean"})[["term", "estimate", "se", "constraint", "centering_mean"]]
path = OUT / "steal_success_v1_coefficients.csv"
coef_out.to_csv(path, index=False)
written[path] = len(coef_out)
_, _, u_f = unpack(w_f, d_f)
off_f = np.cumsum([0] + d_f["sizes"])
pe = [pd.DataFrame(columns=["role", "retro_id", "effect", "throws"])]
for i, r in enumerate(ROLES):
    ids = list(st_f["players"][r])
    if ids:
        counts = fit_table[f"{r}_id"].value_counts()
        pe.append(pd.DataFrame({"role": r, "retro_id": ids, "effect": u_f[off_f[i]:off_f[i + 1]], "throws": counts.reindex(ids).to_numpy()}))
pe = pd.concat(pe, ignore_index=True)
path = OUT / "steal_success_v1_player_effects.csv"
pe.to_csv(path, index=False)
written[path] = len(pe)
used = [c for c, _ in sp_final["constrained"]] + sp_final["free_cont"]
rejected = {
    "B-att (attempts version of runner lead gained)": "leakage: the attribute is a season average over the runner's own attempts, so it contains the "
    "outcomes being predicted; with 2024 lead values B-att's 2025 log loss was 0.5050 against 0.5002 for B (S2b-ii check)",
    "missing indicators": "removed: each covered 0 to 15 rows, mostly all safe, so their coefficients separated; attributes keep the season-mean fill",
    "C (LightGBM)": f"benchmark only by rule; 2025 log loss {results['C']['log_loss']:.5f}",
    "runner_lead_gained_opp term": "dropped from every model (kept on the fit table): its estimate was indistinguishable from zero "
    "(0.030 per ft, SE 0.030 on all three seasons)",
}
for k in ["A32", "B32", "B-arm32"]:
    if k != chosen:
        rejected[k] = f"2025 log loss {results[k]['log_loss']:.5f} against {results[chosen]['log_loss']:.5f} for {chosen}; " + (
            "does not beat A32 by more than 0.002" if k != "A32" else f"{chosen} beats it by more than 0.002")
meta = {
    "specification": chosen,
    "rejected": rejected,
    "seasons": {"evaluation_train": TRAIN, "evaluation_test": TEST, "final_fit": ALL},
    "penalties": dict(zip(ROLES, lam_final)) if lam_final else None,
    "penalties_searched_for_B32": dict(zip(ROLES, lam_b)),
    "penalty_grid": GRID.tolist(),
    "fixed_effect_ridge": FIXED_RIDGE,
    "metrics": {"train_2023_2024_in_sample": {"log_loss": results[chosen]["train_log_loss"]},
                "test_2025": {k: results[chosen][k] for k in ["log_loss", "brier", "auc"]},
                "test_2025_baseline": {"log_loss": log_loss(y_test, base), "brier": float(np.mean((base - y_test) ** 2))},
                "test_2025_all_models": {k: {m: v[m] for m in ["log_loss", "brier", "auc"]} for k, v in results.items()},
                "final_in_sample_2023_2025": final_metrics},
    "lagged_check": {"rows": lag_rows, "one_attribute_gaps": {f"{m}: {c}": g for (m, c), g in gaps.items()}, "leaks_above_0.002": leaks},
    "fill_values_by_season": {str(s): {c: ({h: float(hand_means.loc[(s, h), ATTR_SOURCE[c][1]]) for h in ("L", "R")} if c in PITCHER_FILLS
                                           else float(season_means.loc[s, ATTR_SOURCE[c][1]])) for c in used} for s in season_means.index},
    "fill_values_note": "season means over player-seasons; pitcher_primary_lead and pitcher_lead_gained by season and pitcher hand, over pitcher-seasons of that hand",
    "centering": {c: {"mean": st_f["mean"][c], "sd": st_f["sd"][c]} for c in st_f["mean"]},
    "count_3_2_term": "indicator for a 3-2 count; every 3-2 throw situation is a strike-three throw, so it is the 3-2 strike-three effect "
                      "on top of pitch_result=strike_three",
    "factor_levels": {"pitch_result": ["ball (reference)", "strike", "strike_three"], "count_3_2": [0, 1], "pickoff_throws": ["0 (reference)", "1", "2+"],
                      "pitchout": [0, 1], "pitcher_hand": ["R (reference)", "L"], "bat_side": ["R (reference)", "L"],
                      "season": [f"{st_f['seasons'][0]} (reference)"] + [str(s) for s in st_f["seasons"][1:]],
                      "unseen_season": "the latest fitted season's effect and fill values are used, as in the engine"},
    "unseen_player": "effect 0",
    "bootstrap": {"replicates": N_BOOT, "unit": "game", "seed": SEED + 1},
    "seed": SEED,
    "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "lightgbm": lgb.__version__},
}
print("\nfill values recorded in the meta file (season means over player-seasons; pitcher attributes by hand, over pitcher-seasons of that hand):")
fill_rows = []
for s in season_means.index:
    row = {"season": s}
    for c in used:
        if c in PITCHER_FILLS:
            for h in ("L", "R"):
                row[f"{c} {h}"] = hand_means.loc[(s, h), ATTR_SOURCE[c][1]]
        else:
            row[c] = season_means.loc[s, ATTR_SOURCE[c][1]]
    row["pitcher-seasons L / R"] = f"{int(hand_counts.get((s, 'L'), 0))} / {int(hand_counts.get((s, 'R'), 0))}"
    fill_rows.append(row)
print(pd.DataFrame(fill_rows).set_index("season").to_string(float_format="{:.3f}".format))
print(f"pitcher-seasons with lead data but no hand in the plays (left out of the hand means): {n_no_hand}")
path = OUT / "steal_success_v1_meta.json"
path.write_text(json.dumps(meta, indent=2))
written[path] = 1

print("\nfiles:")
for path, n in written.items():
    print(f"  {path.relative_to(ROOT)}: {n:,} rows" + (" (one JSON object)" if path.suffix == ".json" else "") + (" (header only)" if n == 0 else ""))
print("runtime by stage:")
for k, v in STAGES.items():
    print(f"  {k}: {v:.1f} s")
print(f"  total: {time.time() - T0:.1f} s")
