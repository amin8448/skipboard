from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import betaln

ROOT = Path(__file__).resolve().parent.parent
PA = ROOT / "data" / "derived" / "pa_outcomes_2016_2025.parquet"
TABLES = ROOT / "models" / "pa_model"

CATS = ["K", "BB", "HBP", "1B", "2B", "3B", "HR", "GB_OUT", "AIR_OUT", "ROE", "FC", "OTHER"]
MODEL = ["K", "BB", "HBP", "1B", "2B", "3B", "HR", "GB", "AIR_OUT"]  # GB = GB_OUT + FC
FULL = MODEL + ["ROE", "OTHER"]  # ROE and OTHER stay at league rates
COUNT_LABEL = [f"{b}-{s}" for b in range(4) for s in range(3)]
MATCHUPS = ["L_vs_L", "L_vs_R", "R_vs_L", "R_vs_R"]
WEIGHTS = {0: 6, 1: 5, 2: 4, 3: 3}  # seasons back from the current one
GB_I = MODEL.index("GB")

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)

def load_pa():
    pa = pd.read_parquet(PA)
    pa = pa[~pa["excluded"] & pa["count_ok"]].reset_index(drop=True)
    pa["mcat"] = pa["category"].replace({"GB_OUT": "GB", "FC": "GB"})
    pa["bat_side"] = np.where(pa["bathand"] == "B", np.where(pa["pithand"] == "R", "L", "R"), pa["bathand"])
    pa["matchup"] = pa["bat_side"] + "_vs_" + pa["pithand"]
    pa["state"] = pa["base_pre"] * 3 + pa["outs_pre"]
    return pa


def onehot(cats, labels):
    return (np.asarray(cats)[:, None] == np.array(labels)[None, :]).astype(float)


def dist(df, labels, col):
    return df[col].value_counts().reindex(labels, fill_value=0) / len(df)


def through(df, count):
    return df["counts_seen"].str.contains(rf"(?:^|\|){count}(?:\||$)", regex=True)


def league_tables(df):
    rates12 = dist(df, CATS, "category")
    rates11 = dist(df, FULL, "mcat")
    L9 = rates11[MODEL] / rates11[MODEL].sum()
    in_model = df[df["mcat"].isin(MODEL)]
    platoon = pd.DataFrame({m: dist(in_model[in_model["matchup"] == m], MODEL, "mcat") for m in MATCHUPS}).T
    side = pd.DataFrame({s: dist(in_model[in_model["bat_side"] == s], MODEL, "mcat") for s in ["L", "R"]}).T
    ratio_league = platoon / L9
    ratio_side = platoon / side.loc[[m[0] for m in MATCHUPS]].to_numpy()
    counts = pd.DataFrame({c: dist(df[through(df, c)], FULL, "mcat") for c in COUNT_LABEL}).T
    counts12 = pd.DataFrame({c: dist(df[through(df, c)], CATS, "category") for c in COUNT_LABEL}).T
    gb = df[df["mcat"] == "GB"]
    fc = gb.groupby("state")["category"].agg(lambda c: (c == "FC").mean()).reindex(range(24), fill_value=0.0)
    state_counts = pd.crosstab(in_model["state"], in_model["mcat"]).reindex(index=range(24), columns=MODEL, fill_value=0)
    state_ratio = state_ratio_table(state_counts)["ratio"]
    return {"rates12": rates12, "rates11": rates11, "L9": L9, "platoon": platoon, "ratio_league": ratio_league,
            "ratio_side": ratio_side, "counts": counts, "counts12": counts12, "fc_share": fc, "state_ratio": state_ratio}



def state_ratio_table(counts):
    # counts: PAs per base-out state (rows) and matchup category (columns).
    # Method of moments per category: between-state variance of the shares minus the average
    # sampling variance gives the prior variance tau2, and m = p(1 - p) / tau2 - 1. If tau2 <= 0
    # the category gets m = inf, which sets its ratio to 1 in every state.
    x = counts.to_numpy(dtype=float)
    n = x.sum(axis=1, keepdims=True)
    share = x / n
    overall = x.sum(axis=0) / x.sum()
    tau2 = ((share - overall) ** 2).mean(axis=0) - (overall * (1 - overall) / n).mean(axis=0)
    m = np.where(tau2 > 0, np.maximum(overall * (1 - overall) / np.where(tau2 > 0, tau2, 1.0) - 1, 0.0), np.inf)
    shrunk = np.where(np.isinf(m), overall, (n * share + np.where(np.isinf(m), 0.0, m) * overall) / (n + np.where(np.isinf(m), 0.0, m)))
    frame = lambda a: pd.DataFrame(a, index=counts.index, columns=counts.columns)  # noqa: E731
    return {"n": pd.Series(n[:, 0], index=counts.index), "share": frame(share), "raw_ratio": frame(share / overall),
            "m": pd.Series(m, index=counts.columns), "ratio": frame(shrunk / overall)}


def prior_counts(df, role, season):
    sub = df[df["season"].between(season - 3, season - 1) & df["mcat"].isin(MODEL)]
    w = (season - sub["season"]).map(WEIGHTS).to_numpy()
    oh = onehot(sub["mcat"], MODEL)
    wc = pd.DataFrame(oh * w[:, None], columns=MODEL).groupby(sub[role].to_numpy()).sum()
    n = sub.groupby(role).size().rename("n")
    wpa = pd.Series(w, index=sub.index).groupby(sub[role]).sum().rename("wpa")
    return wc, n, wpa


def rates_as_of(df, test, role, season):
    wc, _, _ = prior_counts(df, role, season)
    cur = df[(df["season"] == season) & df["mcat"].isin(MODEL)]
    daily = pd.DataFrame(onehot(cur["mcat"], MODEL), columns=MODEL).groupby([cur[role].to_numpy(), cur["date"].to_numpy()]).sum()
    daily.index.names = ["player", "date"]
    before = daily.groupby(level="player").cumsum() - daily  # strictly before the date; same-day games excluded
    keys = pd.MultiIndex.from_arrays([test[role].to_numpy(), test["date"].to_numpy()])
    cur_counts = before.reindex(keys).fillna(0).to_numpy()
    prior = wc.reindex(test[role].to_numpy()).fillna(0).to_numpy()
    return WEIGHTS[0] * cur_counts + prior


def shrink(wcounts, k, L):
    r = (wcounts + k * L) / (wcounts.sum(axis=1, keepdims=True) + k)
    return r / r.sum(axis=1, keepdims=True)


def fit_k(df, role, season, L9):
    wc, n, wpa = prior_counts(df, role, season)
    n = n.reindex(wc.index).to_numpy().astype(float)
    wpa = wpa.reindex(wc.index).to_numpy()
    mean_w = wpa.sum() / n.sum()
    rows = []
    for j, cat in enumerate(MODEL):
        x = wc[cat].to_numpy() / wpa * n  # weighted rate on the actual-PA scale
        mu = L9[cat]

        def nll(logk):
            k = np.exp(logk)
            return -(betaln(x + mu * k, n - x + (1 - mu) * k) - betaln(mu * k, (1 - mu) * k)).sum()

        res = minimize_scalar(nll, bounds=(np.log(1.0), np.log(1e6)), method="bounded")
        k_pa = float(np.exp(res.x))
        rows.append({"role": role, "category": cat, "k": k_pa * mean_w, "k_pa": k_pa, "league_rate": mu})
    return pd.DataFrame(rows), len(n), mean_w


def to_full(p9, lg, state):
    other = lg["rates11"][["ROE", "OTHER"]].to_numpy()
    p11 = np.hstack([p9 * (1 - other.sum()), np.broadcast_to(other, (len(p9), 2))])
    return split_gb(p11, lg, state)


def split_gb(p11, lg, state):
    s = lg["fc_share"].to_numpy()[state]
    gb = p11[:, GB_I]
    out = pd.DataFrame(p11, columns=FULL)
    out["GB_OUT"] = gb * (1 - s)
    out["FC"] = gb * s
    return out[CATS].to_numpy()


def log_loss(p12, cats):
    idx = pd.Index(CATS).get_indexer(cats)
    p = p12[np.arange(len(p12)), idx]
    return float(-np.log(np.maximum(p, 1e-12)).mean()), int((p <= 0).sum())


def combine(b, q, L, ratio=None):
    p = b * q / L
    if ratio is not None:
        p = p * ratio
    return p / p.sum(axis=1, keepdims=True)


def full_model(b, q, lg, matchups, states, use_state=True):
    # Odds-ratio matchup, then the league platoon ratio, then the base-out state ratio.
    ratio = lg["ratio_league"].loc[matchups].to_numpy()
    if use_state:
        ratio = ratio * lg["state_ratio"].loc[states].to_numpy()
    return combine(b, q, lg["L9"].to_numpy(), ratio)

def count_condition(p9, lg, counts11, state):
    # counts11: league outcome distribution over FULL at the PA's count, one row per PA
    L11 = lg["rates11"].to_numpy()
    other = L11[-2:]
    p11 = np.hstack([p9 * (1 - other.sum()), np.broadcast_to(other, (len(p9), 2))])
    cond = counts11 * (p11 / L11)
    cond = cond / cond.sum(axis=1, keepdims=True)
    return split_gb(cond, lg, state)


def merge_gb(df12):
    out = df12.rename(columns={"GB_OUT": "GB"}).drop(columns="FC")
    out["GB"] = out["GB"] + df12["FC"]
    return out


def saved_league_tables(tables_dir=TABLES):
    counts = pd.read_csv(tables_dir / "count_outcomes_v1.csv")
    counts["count"] = counts["balls"].astype(str) + "-" + counts["strikes"].astype(str)
    c12 = counts.pivot(index="count", columns="category", values="p").reindex(index=COUNT_LABEL, columns=CATS)
    c12 = c12.div(c12.sum(axis=1), axis=0)
    c11 = merge_gb(c12)[FULL]
    rates11 = c11.loc["0-0"]
    L9 = rates11[MODEL] / rates11[MODEL].sum()
    plat = pd.read_csv(tables_dir / "platoon_rates_v1.csv")
    plat = plat[plat["group_type"] == "matchup"].pivot(index="group", columns="category", values="p").reindex(index=MATCHUPS, columns=CATS)
    platoon = merge_gb(plat)[MODEL]
    platoon = platoon.div(platoon.sum(axis=1), axis=0)
    fc = pd.read_csv(tables_dir / "fc_share_v1.csv")
    fc_share = pd.Series(fc["fc_share"].to_numpy(), index=fc["base_code"] * 3 + fc["outs"]).sort_index()
    sr = pd.read_csv(tables_dir / "state_ratio_v1.csv")
    state_ratio = sr.assign(state=sr["base_code"] * 3 + sr["outs"]).pivot(index="state", columns="category", values="ratio").reindex(index=range(24), columns=MODEL)
    return {"rates11": rates11, "L9": L9, "platoon": platoon, "ratio_league": platoon / L9, "counts": c11, "fc_share": fc_share,
            "state_ratio": state_ratio}


def main():
    pa = load_pa()
    print(f"plate appearances used (non-excluded, count_ok): {len(pa):,}")

    # Final league tables come from the saved 2023-2025 files; rebuild them from data as a check.
    lg_final = league_tables(pa[pa["season"].between(2023, 2025)])
    saved_counts = pd.read_csv(TABLES / "count_outcomes_v1.csv")
    saved_plat = pd.read_csv(TABLES / "platoon_rates_v1.csv")
    saved_rates = saved_counts[(saved_counts["balls"] == 0) & (saved_counts["strikes"] == 0)].set_index("category")["p"].reindex(CATS)
    saved_cnt = saved_counts.assign(c=saved_counts["balls"].astype(str) + "-" + saved_counts["strikes"].astype(str)).pivot(index="c", columns="category", values="p").reindex(index=COUNT_LABEL, columns=CATS)
    saved_m = saved_plat[saved_plat["group_type"] == "matchup"].pivot(index="group", columns="category", values="p").reindex(index=MATCHUPS, columns=CATS)
    rebuilt_m = pd.DataFrame({m: dist(pa[pa["season"].between(2023, 2025) & (pa["matchup"] == m)], CATS, "category") for m in MATCHUPS}).T
    print("\nsaved 2023-2025 tables vs rebuilt from data (max absolute difference)")
    print(f"  league rates (count table 0-0 row): {np.abs(saved_rates.to_numpy() - lg_final['rates12'].to_numpy()).max():.2e}")
    print(f"  count table: {np.abs(saved_cnt.to_numpy() - lg_final['counts12'].to_numpy()).max():.2e}")
    print(f"  platoon table: {np.abs(saved_m.to_numpy() - rebuilt_m.to_numpy()).max():.2e} (saved table did not require count_ok)")
    print(f"  state ratio table: {np.abs(saved_league_tables()['state_ratio'].to_numpy() - lg_final['state_ratio'].to_numpy()).max():.2e}")
    L9_final = saved_rates.rename({"GB_OUT": "GB"}).reindex(MODEL)
    L9_final["GB"] += saved_rates["FC"]
    L9_final = L9_final / L9_final.sum()

    print("\n=== shrinkage (beta-binomial fit on 2022-2024 weighted counts, as of the start of 2025)")
    print("k is fitted on the actual-PA scale (weighted rate x actual PA) and converted to the weighted scale")
    print("with the average weight per PA, because weights of 3 to 6 would otherwise count each PA several times.")
    fits = []
    for role in ["batter", "pitcher"]:
        f, n_players, mean_w = fit_k(pa, role, 2025, L9_final)
        print(f"  {role}s: {n_players:,} players, average weight per PA {mean_w:.3f}")
        fits.append(f)
    shrink_final = pd.concat(fits, ignore_index=True)
    tab = shrink_final.pivot(index="category", columns="role", values=["k_pa", "k"]).reindex(MODEL)
    tab.columns = [f"{a} {b}" for a, b in tab.columns]
    tab["league rate"] = L9_final
    print(tab.to_string(formatters={c: "{:,.0f}".format for c in tab.columns if c.startswith("k")} | {"league rate": "{:.4f}".format}))

    TABLES.mkdir(parents=True, exist_ok=True)
    shrink_final[["role", "category", "k", "league_rate", "k_pa"]].round(6).to_csv(TABLES / "shrinkage_v1.csv", index=False)
    print("saved models/pa_model/shrinkage_v1.csv (k is on the weighted scale used in the formula; k_pa is per actual PA)")

    print("\n=== validation: league tables from 2023-2024, tested on 2025")
    lg = league_tables(pa[pa["season"].between(2023, 2024)])
    L9 = lg["L9"].to_numpy()
    k_val = {}
    for role in ["batter", "pitcher"]:
        f, _, _ = fit_k(pa, role, 2025, lg["L9"])
        k_val[role] = f.set_index("category").loc[MODEL, "k"].to_numpy()
    test = pa[pa["season"] == 2025].reset_index(drop=True)
    bw = rates_as_of(pa, test, "batter", 2025)
    pw = rates_as_of(pa, test, "pitcher", 2025)
    print(f"2025 PAs: {len(test):,}; batter weighted PA as of date: median {np.median(bw.sum(1)):,.0f}, zero for {int((bw.sum(1) == 0).sum()):,} PAs; "
          f"pitcher: median {np.median(pw.sum(1)):,.0f}, zero for {int((pw.sum(1) == 0).sum()):,} PAs")
    b = shrink(bw, k_val["batter"], L9)
    q = shrink(pw, k_val["pitcher"], L9)
    ratio = lg["ratio_league"].loc[test["matchup"]].to_numpy()
    ratio_side = lg["ratio_side"].loc[test["matchup"]].to_numpy()
    state = test["state"].to_numpy()
    cats = test["category"].to_numpy()

    models = {
        "league only": np.broadcast_to(L9, b.shape),
        "batter only": b,
        "batter + pitcher": combine(b, q, L9),
        "batter + pitcher + platoon": combine(b, q, L9, ratio),
        "  variant: platoon ratio vs batter-side average": combine(b, q, L9, ratio_side),
        "batter + pitcher + platoon + state (full model)": full_model(b, q, lg, test["matchup"], state),
    }
    print("\nmultinomial log loss per PA over the 12 categories (GB split into GB_OUT and FC by base-out state)")
    base_loss = None
    full_p12 = None
    for name, p9 in models.items():
        p12 = to_full(p9, lg, state)
        loss, zeros = log_loss(p12, cats)
        if base_loss is None:
            base_loss = loss
        if name.endswith("(full model)"):
            full_p12, full_p9 = p12, p9
        print(f"  {name:<50} {loss:.5f}   improvement over league {base_loss - loss:+.5f} ({(base_loss - loss) / base_loss:+.2%})" + (f"   zero-probability outcomes: {zeros}" if zeros else ""))

    print("\ncalibration under the full model (deciles of predicted probability)")
    for cat in ["K", "HR"]:
        j = CATS.index(cat)
        pred = full_p12[:, j]
        hit = (cats == cat).astype(float)
        bins = pd.qcut(pred, 10, labels=False, duplicates="drop")
        cal = pd.DataFrame({"pred": pred, "hit": hit, "bin": bins}).groupby("bin").agg(mean_pred=("pred", "mean"), actual=("hit", "mean"), n=("hit", "size"))
        cal["gap"] = cal["actual"] - cal["mean_pred"]
        print(f"  {cat} (overall predicted {pred.mean():.4f}, actual {hit.mean():.4f})")
        print(cal.to_string(formatters={"mean_pred": "{:.4f}".format, "actual": "{:.4f}".format, "gap": "{:+.4f}".format, "n": "{:,}".format}))

    print("\ncount conditioning: league count table vs count-conditioned full model")
    print("(assumes the matchup effect, matchup_i / league_i, is the same at every count)")
    for c in ["0-2", "3-0", "3-2"]:
        m = through(test, c).to_numpy()
        table = np.broadcast_to(lg["counts"].loc[c].to_numpy(), (int(m.sum()), len(FULL)))
        loss_table, _ = log_loss(split_gb(table, lg, state[m]), cats[m])
        loss_cond, _ = log_loss(count_condition(full_p9[m], lg, table, state[m]), cats[m])
        loss_nostate, _ = log_loss(count_condition(models["batter + pitcher + platoon"][m], lg, table, state[m]), cats[m])
        print(f"  {c}: n={int(m.sum()):,}  league count table {loss_table:.5f}  count-conditioned model {loss_cond:.5f}  improvement {loss_table - loss_cond:+.5f} ({(loss_table - loss_cond) / loss_table:+.2%})"
              f"  [without state adjustment {loss_nostate:.5f}]")


if __name__ == "__main__":
    main()
