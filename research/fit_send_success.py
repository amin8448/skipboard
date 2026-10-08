import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize
from scipy.special import log_ndtr, ndtr
from scipy.stats import multivariate_normal
from skipboard.state import GameState
from skipboard.value import wp_after

T0 = time.time()
STAGES = {}
ROOT = Path(__file__).resolve().parent.parent
FIT = ROOT / "data" / "derived" / "send" / "send_fit_table_2023_2025.parquet"
POP = ROOT / "data" / "derived" / "send" / "send_population_2023_2025.parquet"
FILLS = ROOT / "data" / "derived" / "send" / "send_fill_values_2023_2025.json"
RE = ROOT / "models" / "run_expectancy" / "re24_v1.csv"
OUT = ROOT / "data" / "derived" / "send"
TRAIN, TEST, ALL = [2023, 2024], [2025], [2023, 2024, 2025]
SEED = 2026
N_BOOT = 200
RIDGE = 1e-4  # on every slope (not the intercepts or rho), standardized scale
THETA_BOUNDS = (-20.0, 5.0)  # log slope per standard deviation for constrained terms
ATANH_RHO_BOUNDS = (-3.0, 3.0)  # |rho| <= 0.995
GL_NODES = 48
SWITCH_GAIN = 0.002
PROFILE_GRID = np.round(np.arange(0.30, 0.99 + 1e-9, 0.01), 2)  # rho values for the profile likelihood
PROFILE_CUTOFF = 3.84  # chi-square with one degree of freedom, 95%
POS_LABEL = {7: "LF", 8: "CF", 9: "RF"}
SPRAY_CONVENTION = ("spray_toward_line: degrees from straightaway toward the fielding outfielder's foul line; LF: minus the spray angle "
                    "(spray angle negative toward left field), RF: the spray angle, CF: the absolute spray angle (toward either gap); "
                    "positive means toward the line on the fielder's side")

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 300)
GL_X, GL_W = np.polynomial.legendre.leggauss(GL_NODES)


def stage(name, t):
    STAGES[name] = time.time() - t
    return time.time()


def bvn_cdf(a, b, rho):
    # P(X < a, Y < b) for a standard bivariate normal with correlation rho (scalar): Drezner and Wesolowsky's
    # integral over the correlation, with r = sin(t), by Gauss-Legendre quadrature.
    a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
    base = ndtr(a) * ndtr(b)
    if rho == 0:
        return base
    t_max = np.arcsin(rho)
    t = t_max / 2 * (GL_X + 1)
    sin_t, cos2_t = np.sin(t), np.cos(t) ** 2
    hk, hs = (a * b)[..., None], ((a * a + b * b) / 2)[..., None]
    f = np.exp((hk * sin_t - hs) / cos2_t)
    return base + t_max / 2 * (f @ GL_W) / (2 * np.pi)


def bvn_parts(a, c, r):
    # The CDF and its derivatives in a, c and r.
    p = np.clip(bvn_cdf(a, c, r), 1e-300, 1.0)
    s = np.sqrt(1 - r * r)
    pa = np.exp(-a * a / 2) / np.sqrt(2 * np.pi) * ndtr((c - r * a) / s)
    pc = np.exp(-c * c / 2) / np.sqrt(2 * np.pi) * ndtr((a - r * c) / s)
    pr = np.exp(-(a * a - 2 * r * a * c + c * c) / (2 * s * s)) / (2 * np.pi * s)
    return p, pa, pc, pr


t = time.time()
grid = np.linspace(-3, 3, 13)
rhos = [-0.99, -0.95, -0.9, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
max_err, worst = 0.0, None
for r in rhos:
    mvn = multivariate_normal(mean=[0, 0], cov=[[1, r], [r, 1]], maxpts=10_000_000, abseps=1e-12, releps=1e-12)
    A, B = np.meshgrid(grid, grid)
    mine = bvn_cdf(A.ravel(), B.ravel(), r)
    ref = np.array([mvn.cdf([x, y]) for x, y in zip(A.ravel(), B.ravel())])
    err = np.abs(mine - ref)
    if err.max() > max_err:
        max_err, worst = float(err.max()), (float(A.ravel()[err.argmax()]), float(B.ravel()[err.argmax()]), r)
print(f"bivariate normal CDF ({GL_NODES}-point Gauss-Legendre, Drezner-Wesolowsky with r = sin t) against scipy.stats.multivariate_normal.cdf "
      f"on a 13 x 13 grid of (a, b) in [-3, 3] and {len(rhos)} values of rho in [-0.99, 0.99]: maximum absolute error {max_err:.2e} at (a, b, rho) = {worst}")
if max_err > 1e-6:
    raise SystemExit("bivariate normal CDF is not accurate to 1e-6")
t = stage("bivariate normal check", t)

fit_table = pd.read_parquet(FIT)
df = fit_table.copy()
miss_launch = df["launch_speed"].isna() | df["launch_angle"].isna()
df["bb"] = df["bb_type"].replace({"popup": "fly_ball"})
launch_fills = df.groupby(["season", "bb"])[["launch_speed", "launch_angle"]].mean()  # popups count as fly balls, as in the model
for c in ["launch_speed", "launch_angle"]:
    df[c] = df[c].fillna(df.groupby(["season", "bb"])[c].transform("mean"))
print(f"\nfit table: {len(df):,} rows (sends {int(df['sent'].sum()):,}, holds {int((~df['sent']).sum()):,}); missing launch speed or angle filled with "
      f"the season mean for the batted-ball type (popups counted as fly balls): {int(miss_launch.sum())} "
      f"({fit_table.loc[miss_launch, 'bb_type'].value_counts().to_dict()})")
df["spray_toward_line"] = np.select([df["fielder_pos"] == 7, df["fielder_pos"] == 9], [-df["spray_angle"], df["spray_angle"]], df["spray_angle"].abs())
print(f"spray convention: {SPRAY_CONVENTION}")
print("spray_toward_line by position (degrees):", df.groupby(df["fielder_pos"].map(POS_LABEL))["spray_toward_line"].describe()[
    ["mean", "min", "50%", "max"]].round(1).to_dict("index"))
df["throw_over_arm"] = df["throw_distance"] / df["arm_avg"]  # ft per mph
df["fielder_over_ls"] = df["fielder_distance"] / df["launch_speed"]  # ft per mph
df["fielding_over_ls"] = df["landing_distance"] / df["launch_speed"]  # ft per mph
df["inv_speed"] = 1 / df["runner_sprint_speed"]  # s per ft
df["score_c"] = df["score_diff"].clip(-4, 4).astype(int)
df["inning_group"] = pd.cut(df["inning"], [0, 3, 6, 8, 99], labels=["1-3", "4-6", "7-8", "9+"]).astype(str)
df["late_close"] = ((df["inning"] >= 7) & (df["score_diff"].abs() <= 1)).astype(float)
for p, lab in [(8, "CF"), (9, "RF")]:
    df[f"pos={lab}"] = (df["fielder_pos"] == p).astype(float)
for b in ["line_drive", "fly_ball"]:
    df[f"bb={b}"] = (df["bb"] == b).astype(float)
df["bat_side=L"] = (df["stand"] == "L").astype(float)
for o in [1, 2]:
    df[f"outs={o}"] = (df["outs"] == o).astype(float)
for sc in [-4, -3, -2, -1, 1, 2, 3, 4]:
    df[f"score={sc:+d}"] = (df["score_c"] == sc).astype(float)
for g in ["4-6", "7-8", "9+"]:
    df[f"inning={g}"] = (df["inning_group"] == g).astype(float)
df["sent"] = df["sent"].astype(bool)
df["safe"] = df["safe"].astype(bool)

FACTORS = ["pos=CF", "pos=RF", "bb=line_drive", "bb=fly_ball", "bat_side=L", "outs=1", "outs=2"]
Z_EXTRA_FACTORS = [f"score={s:+d}" for s in [-4, -3, -2, -1, 1, 2, 3, 4]] + ["inning=4-6", "inning=7-8", "inning=9+", "late_close"]
X_SPECS = {
    "linear": {"cont": ["fielding_distance", "spray_toward_line", "fielder_distance", "launch_speed", "launch_angle", "arm_avg", "runner_sprint_speed"],
               "signs": {"fielding_distance": 1, "fielder_distance": 1, "arm_avg": -1, "runner_sprint_speed": 1}},
    "race": {"cont": ["throw_over_arm", "fielder_over_ls", "fielding_over_ls", "inv_speed"],
             "signs": {"throw_over_arm": 1, "fielder_over_ls": 1, "fielding_over_ls": 1, "inv_speed": -1}},
}
df["fielding_distance"] = df["landing_distance"]
UNITS = {"fielding_distance": "ft", "spray_toward_line": "degrees", "fielder_distance": "ft", "launch_speed": "mph", "launch_angle": "degrees",
         "arm_avg": "mph", "runner_sprint_speed": "ft/s", "throw_over_arm": "ft per mph", "fielder_over_ls": "ft per mph",
         "fielding_over_ls": "ft per mph", "inv_speed": "s per ft", "on_deck_quality": "runs"}
print("X-race units: throw_over_arm = throw distance / arm strength (ft per mph); fielder_over_ls = fielder distance to the ball / launch speed "
      "(ft per mph); fielding_over_ls = fielding-point distance / launch speed (ft per mph); inv_speed = 1 / sprint speed (s per ft)")


def columns(spec):
    xc = X_SPECS[spec]["cont"]
    return {"x": (xc, FACTORS), "z": (xc + ["on_deck_quality"], FACTORS + Z_EXTRA_FACTORS)}


def standardizer(train, cont):
    return {c: (float(train[c].mean()), float(train[c].std())) for c in cont}


def design(d, cont, factors, scale, signs=None, free=False):
    cols = [np.ones(len(d))] + [((d[c] - scale[c][0]) / scale[c][1]).to_numpy(float) for c in cont] + [d[f].to_numpy(float) for f in factors]
    names = ["intercept"] + cont + factors
    sg = np.array([0] + [0 if free or signs is None else signs.get(c, 0) for c in cont] + [0] * len(factors), float)
    return np.column_stack(cols), names, sg


def slopes(t, sg):
    return np.where(sg == 0, t, sg * np.exp(np.clip(t, *THETA_BOUNDS)))


def ridge_vec(n):
    r = np.full(n, RIDGE)
    r[0] = 0.0
    return r


def probit_fit(X, sg, y, x0=None):
    rv = ridge_vec(X.shape[1])
    q = np.where(y, 1.0, -1.0)

    def obj(w):
        b = slopes(w, sg)
        e = q * (X @ b)
        ll = log_ndtr(e)
        g = q * np.exp(-e * e / 2 - 0.5 * np.log(2 * np.pi) - ll)
        db = X.T @ g - rv * b
        return -(ll.sum() - 0.5 * np.sum(rv * b * b)), -(db * np.where(sg == 0, 1.0, b))

    if x0 is None:
        x0 = np.where(sg == 0, 0.0, np.log(0.05))
        x0[0] = float(np.clip(np.sqrt(2) * scipy.special.erfinv(2 * y.mean() - 1), -3, 3))
    bounds = [(None, None) if s == 0 else THETA_BOUNDS for s in sg]
    return minimize(obj, x0, jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": 5000}).x


def to_raw(b, sg):
    return np.where(sg == 0, b, np.log(np.maximum(np.abs(b), 1e-8)))


def sel_fit(Z, X, sgz, sgx, sent, safe, x0, rho_fixed=None):
    nz = Z.shape[1]
    rz, rx = ridge_vec(nz), ridge_vec(X.shape[1])
    held, s1, s0 = ~sent, sent & safe, sent & ~safe

    def obj(w):
        g = slopes(w[:nz], sgz)
        b = slopes(w[nz:-1], sgx)
        rho = np.tanh(w[-1])
        a, xb = Z @ g, X @ b
        da, dxb, drho = np.zeros(len(a)), np.zeros(len(a)), 0.0
        lh = log_ndtr(-a[held])
        ll = lh.sum()
        da[held] = -np.exp(-a[held] ** 2 / 2 - 0.5 * np.log(2 * np.pi) - lh)
        for mask, q in [(s1, 1.0), (s0, -1.0)]:
            p, pa, pc, pr = bvn_parts(a[mask], q * xb[mask], q * rho)
            ll += np.log(p).sum()
            da[mask] = pa / p
            dxb[mask] = q * pc / p
            drho += q * np.sum(pr / p)
        pen = 0.5 * (np.sum(rz * g * g) + np.sum(rx * b * b))
        dg = (Z.T @ da - rz * g) * np.where(sgz == 0, 1.0, g)
        dbv = (X.T @ dxb - rx * b) * np.where(sgx == 0, 1.0, b)
        return -(ll - pen), -np.concatenate([dg, dbv, [drho * (1 - rho * rho)]])

    rho_bounds = ATANH_RHO_BOUNDS if rho_fixed is None else (float(np.arctanh(rho_fixed)),) * 2
    bounds = [(None, None) if s == 0 else THETA_BOUNDS for s in sgz] + [(None, None) if s == 0 else THETA_BOUNDS for s in sgx] + [rho_bounds]
    if rho_fixed is not None:
        x0 = np.concatenate([x0[:-1], [np.arctanh(rho_fixed)]])
    return minimize(obj, x0, jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": 10000}).x


def log_loss(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y, p):
    return float(np.mean((p - y) ** 2))


def prepare(spec, train, free=False):
    cols = columns(spec)
    scale = standardizer(train, sorted(set(cols["x"][0] + cols["z"][0])))
    return {"spec": spec, "cols": cols, "scale": scale, "free": free}


def matrices(m, d):
    X, xn, sgx = design(d, *m["cols"]["x"], m["scale"], X_SPECS[m["spec"]]["signs"], m["free"])
    Z, zn, sgz = design(d, *m["cols"]["z"], m["scale"])
    return X, xn, sgx, Z, zn, sgz


def fit_sel(m, d, x0=None, rho_fixed=None):
    X, xn, sgx, Z, zn, sgz = matrices(m, d)
    sent, safe = d["sent"].to_numpy(), d["safe"].to_numpy()
    if x0 is None:
        g0 = probit_fit(Z, sgz, sent)
        b0 = probit_fit(X[sent], sgx, safe[sent])
        x0 = np.concatenate([g0, b0, [0.0]])
    w = sel_fit(Z, X, sgz, sgx, sent, safe, x0, rho_fixed)
    nz = Z.shape[1]
    return {"w": w, "gamma": slopes(w[:nz], sgz), "beta": slopes(w[nz:-1], sgx), "rho": float(np.tanh(w[-1])), "xn": xn, "zn": zn, "sgx": sgx, "sgz": sgz}


def predict_sel(m, f, d):
    X, _, _, Z, _, _ = matrices(m, d)
    a, xb = Z @ f["gamma"], X @ f["beta"]
    p_send = ndtr(a)
    p_safe_given_sent = np.clip(bvn_cdf(a, xb, f["rho"]) / np.maximum(p_send, 1e-300), 0, 1)
    return p_send, p_safe_given_sent, ndtr(xb)


def sel_loglik(m, f, d):
    X, _, _, Z, _, _ = matrices(m, d)
    a, xb = Z @ f["gamma"], X @ f["beta"]
    sent, safe = d["sent"].to_numpy(), d["safe"].to_numpy()
    ll = log_ndtr(-a[~sent]).sum()
    for mask, q_ in [(sent & safe, 1.0), (sent & ~safe, -1.0)]:
        ll += np.log(np.clip(bvn_cdf(a[mask], q_ * xb[mask], q_ * f["rho"]), 1e-300, 1)).sum()
    return float(ll)


def fit_sonly(m, d, x0=None):
    X, xn, sgx, Z, zn, sgz = matrices(m, d)
    sent, safe = d["sent"].to_numpy(), d["safe"].to_numpy()
    b = slopes(probit_fit(X[sent], sgx, safe[sent], x0), sgx)
    g = probit_fit(Z, sgz, sent)
    return {"beta": b, "gamma": slopes(g, sgz), "xn": xn, "zn": zn, "sgx": sgx}


def calibration(y, p):
    t = pd.DataFrame({"pred": p, "obs": y.astype(float), "q": pd.qcut(p, 5, labels=False, duplicates="drop")})
    return t.groupby("q").agg(mean_predicted=("pred", "mean"), observed=("obs", "mean"), n=("obs", "size"))


train = df[df["season"].isin(TRAIN)].reset_index(drop=True)
test = df[df["season"].isin(TEST)].reset_index(drop=True)
print(f"train {TRAIN}: {len(train):,} rows ({int(train['sent'].sum()):,} sends); test {TEST}: {len(test):,} rows ({int(test['sent'].sum()):,} sends)")

print("\n=== models trained on 2023-2024, tested on 2025")
models = {"S-only": prepare("linear", train), "SEL-linear": prepare("linear", train), "SEL-race": prepare("race", train),
          "SEL-linear-unconstrained": prepare("linear", train, free=True)}
fits, rows, calib = {}, [], {}
ys_test, yf_test = test["sent"].to_numpy(), test["safe"].to_numpy()
sent_test = ys_test
base_send, base_safe = train["sent"].mean(), train.loc[train["sent"], "safe"].mean()
for name, m in models.items():
    if name == "S-only":
        f = fit_sonly(m, train)
        X, _, _, Z, _, _ = matrices(m, test)
        p_send, p_safe = ndtr(Z @ f["gamma"]), ndtr(X @ f["beta"])
    else:
        f = fit_sel(m, train)
        p_send, p_safe, _ = predict_sel(m, f, test)
    fits[name] = f
    rows.append({"model": name, "rho": f.get("rho", np.nan), "send log loss": log_loss(ys_test, p_send), "send Brier": brier(ys_test, p_send),
                 "safe|sent log loss": log_loss(yf_test[sent_test], p_safe[sent_test]), "safe|sent Brier": brier(yf_test[sent_test], p_safe[sent_test])})
    calib[name] = (calibration(ys_test, p_send), calibration(yf_test[sent_test], p_safe[sent_test]))
rows.append({"model": "constant baseline", "rho": np.nan, "send log loss": log_loss(ys_test, np.full(len(test), base_send)),
             "send Brier": brier(ys_test, np.full(len(test), base_send)),
             "safe|sent log loss": log_loss(yf_test[sent_test], np.full(int(sent_test.sum()), base_safe)),
             "safe|sent Brier": brier(yf_test[sent_test], np.full(int(sent_test.sum()), base_safe))})
table = pd.DataFrame(rows).set_index("model")
table["combined log loss"] = table["send log loss"] + table["safe|sent log loss"]
print(f"constant baselines: training send rate {base_send:.4f}, training safe rate among sends {base_safe:.4f}")
print("S-only's send column is a separate send probit on the same covariates (the naive model has no send equation)")
print(table.to_string(float_format="{:.5f}".format))
for name, (cs, cf) in calib.items():
    print(f"\ncalibration by predicted quintile, {name}: send decision (all 2025 rows) | safe among 2025 sends")
    print(pd.concat([cs.add_prefix("send "), cf.add_prefix("safe ")], axis=1).to_string(float_format="{:.3f}".format))
t = stage("train and test", t)


def coef_table(m, f, kind="sel"):
    rows = []
    for eq, names, est, sg in [("send", f["zn"], f["gamma"], np.zeros(len(f["zn"]))), ("safe", f["xn"], f["beta"], f["sgx"])]:
        for nm, e, s in zip(names, est, sg):
            sd = m["scale"][nm][1] if nm in m["scale"] else 1.0
            rows.append({"equation": eq, "term": nm, "estimate": e / sd, "per_sd": e if nm in m["scale"] else np.nan,
                         "constraint": {1: ">= 0", -1: "<= 0"}.get(int(s), "none"), "center": m["scale"][nm][0] if nm in m["scale"] else np.nan})
    if kind == "sel":
        rows.append({"equation": "rho", "term": "rho", "estimate": f["rho"], "per_sd": np.nan, "constraint": "none", "center": np.nan})
    return pd.DataFrame(rows)


print("\nsafe-equation coefficients, training fits (per unit; constrained terms marked):")
ct = {n: coef_table(models[n], fits[n], "sel" if n != "S-only" else "s").query("equation != 'send'").set_index("term")["estimate"] for n in models}
show = pd.DataFrame(ct)
all_signs = {**X_SPECS["linear"]["signs"], **X_SPECS["race"]["signs"]}
show["constraint"] = [{1: ">= 0", -1: "<= 0"}.get(all_signs.get(term, 0), "none") for term in show.index]
print(show.to_string(float_format="{:.5f}".format))
uc = coef_table(models["SEL-linear-unconstrained"], fits["SEL-linear-unconstrained"]).set_index(["equation", "term"])["estimate"]
cons = coef_table(models["SEL-linear"], fits["SEL-linear"]).set_index(["equation", "term"])
flips = []
for (eq, term), r in cons.iterrows():
    if r["constraint"] != "none":
        want = 1 if r["constraint"] == ">= 0" else -1
        if np.sign(uc[(eq, term)]) != want:
            flips.append(f"{term}: unconstrained {uc[(eq, term)]:+.5f}, constrained {r['estimate']:+.5f} ({r['constraint']})")
print("sign changes when the constraints are removed (SEL-linear-unconstrained against the constraint direction): "
      + ("; ".join(flips) if flips else "none"))

print("\n=== final fits on 2023-2025 and selection diagnostics")
final_models = {"SEL-linear": prepare("linear", df), "SEL-race": prepare("race", df), "S-only": prepare("linear", df)}
final = {n: (fit_sel(m, df) if n != "S-only" else fit_sonly(m, df)) for n, m in final_models.items()}
for n in ["SEL-linear", "SEL-race"]:
    print(f"{n}: rho {final[n]['rho']:+.4f}")
t = stage("final fits", t)

games = np.array(sorted(df["gid"].unique()))
rows_of_game = pd.Series(np.arange(len(df))).groupby(df["gid"].to_numpy()).apply(np.array)
boot_rng = np.random.default_rng(SEED + 1)
picks = [boot_rng.choice(games, size=len(games), replace=True) for _ in range(N_BOOT)]
boot = {}
for n in ["SEL-linear", "SEL-race"]:
    m, f = final_models[n], final[n]
    reps = []
    for pick in picks:
        d_b = df.iloc[np.concatenate([rows_of_game[g] for g in pick])].reset_index(drop=True)
        fb = fit_sel(m, d_b, x0=f["w"])
        reps.append(coef_table(m, fb)["estimate"].to_numpy())
    boot[n] = np.array(reps)
    lo, hi = np.percentile(boot[n][:, -1], [2.5, 97.5])
    print(f"{n}: rho {f['rho']:+.4f}, game bootstrap ({N_BOOT} replicates) 95% interval [{lo:+.4f}, {hi:+.4f}], bootstrap SE {boot[n][:, -1].std(ddof=1):.4f}")
t = stage("bootstrap", t)

q = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
sent_all = df["sent"].to_numpy()
uncond = {}
for n in ["SEL-linear", "SEL-race"]:
    _, _, uncond[n] = predict_sel(final_models[n], final[n], df)
Xs, _, _, _, _, _ = matrices(final_models["S-only"], df)
uncond["S-only"] = ndtr(Xs @ final["S-only"]["beta"])
qt = {}
for n in ["SEL-linear", "SEL-race"]:
    qt[f"{n} sent"] = np.quantile(uncond[n][sent_all], q)
    qt[f"{n} held"] = np.quantile(uncond[n][~sent_all], q)
qt["S-only held"] = np.quantile(uncond["S-only"][~sent_all], q)
qt["S-only sent"] = np.quantile(uncond["S-only"][sent_all], q)
qtab = pd.DataFrame(qt, index=[f"p{int(x * 100)}" for x in q]).T
qtab["mean"] = [uncond[k.rsplit(" ", 1)[0]][sent_all if k.endswith("sent") else ~sent_all].mean() for k in qtab.index]
qtab["n"] = [int(sent_all.sum()) if k.endswith("sent") else int((~sent_all).sum()) for k in qtab.index]
print("\nunconditional P(safe | X) = Phi(X beta), final fits, 2023-2025, for sent and held plays (S-only: the sends-only probit applied to the same plays):")
print(qtab.to_string(float_format="{:.3f}".format))

re = pd.read_csv(RE).set_index(["base_code", "outs"])["re"]
re_of = lambda b, o: 0.0 if o >= 3 else float(re.loc[(b, o)])  # noqa: E731
p_star = {}
for o in range(3):
    gain, loss, hold = re_of(1, o) + 1, re_of(1, o + 1), re_of(5, o)
    p_star[o] = (hold - loss) / (gain - loss)
    print(f"break-even, {o} out{'s' if o != 1 else ''}: p* = (RE(1_3, {o}) - RE(1__, {o + 1})) / (RE(1__, {o}) + 1 - RE(1__, {o + 1})) = "
          f"({hold:.4f} - {loss:.4f}) / ({gain:.4f} - {loss:.4f}) = {p_star[o]:.4f}")
outs = df["outs"].to_numpy()
pstar_row = np.array([p_star[o] for o in outs])
sanity = pd.DataFrame({"plays": df.groupby("outs").size(), "coaches' send rate": df.groupby("outs")["sent"].mean(), "p*": pd.Series(p_star)})
for n in ["SEL-linear", "SEL-race", "S-only"]:
    sanity[f"{n}: share with P(safe|X) > p*"] = pd.Series(uncond[n] > pstar_row).groupby(outs).mean()
print("\nsanity check: share of plays where the model says sending gains runs, against the coaches' send rate, by outs:")
print(sanity.to_string(float_format="{:.3f}".format))
out_home = (df["runner_result"] == "out at home").to_numpy()
print(f"\nmodel-implied safe probability on the {int(out_home.sum())} plays where the runner was out at home, against all sends:")
oh = []
for n in ["SEL-linear", "SEL-race", "S-only"]:
    row = {"model": n, "out at home: mean P(safe|X)": uncond[n][out_home].mean(), "out at home: median": np.median(uncond[n][out_home]),
           "all sends: mean P(safe|X)": uncond[n][sent_all].mean(), "all sends: median": np.median(uncond[n][sent_all])}
    if n != "S-only":
        _, cond, _ = predict_sel(final_models[n], final[n], df)
        row["out at home: mean P(safe|sent)"] = cond[out_home].mean()
        row["all sends: mean P(safe|sent)"] = cond[sent_all].mean()
    oh.append(row)
print(pd.DataFrame(oh).set_index("model").to_string(float_format="{:.3f}".format))
t = stage("diagnostics", t)

gain = table.loc["SEL-linear", "combined log loss"] - table.loc["SEL-race", "combined log loss"]
chosen = "SEL-race" if gain > SWITCH_GAIN else "SEL-linear"
print(f"\n=== selection: combined 2025 log loss (send + safe among sends), SEL-linear {table.loc['SEL-linear', 'combined log loss']:.5f}, "
      f"SEL-race {table.loc['SEL-race', 'combined log loss']:.5f}; gain of SEL-race over SEL-linear {gain:+.5f} (rule: switch above {SWITCH_GAIN})")
print(f"chosen specification: {chosen}. S-only is the naive comparison only.")

m_f, f_f = final_models[chosen], final[chosen]
coef = coef_table(m_f, f_f)
coef["se"] = boot[chosen].std(axis=0, ddof=1)
print(f"\n=== final {chosen} on 2023-2025 (game bootstrap SEs, {N_BOOT} replicates)")
print(coef[["equation", "term", "estimate", "se", "constraint", "center"]].to_string(index=False, float_format="{:.5f}".format))
p_send_all, p_safe_all, _ = predict_sel(m_f, f_f, df)
in_sample = {"send_log_loss": log_loss(sent_all, p_send_all), "send_brier": brier(sent_all, p_send_all),
             "safe_given_sent_log_loss": log_loss(df["safe"].to_numpy()[sent_all], p_safe_all[sent_all]),
             "safe_given_sent_brier": brier(df["safe"].to_numpy()[sent_all], p_safe_all[sent_all])}
print("in-sample, all three seasons:", {k: round(v, 5) for k, v in in_sample.items()})
# Profile likelihood for rho: refit with rho fixed on the grid, all other parameters free.
ll_free = sel_loglik(m_f, f_f, df)
profile = []
for r in PROFILE_GRID:
    fr = fit_sel(m_f, df, x0=f_f["w"], rho_fixed=float(r))
    profile.append((float(r), sel_loglik(m_f, fr, df)))
profile = pd.DataFrame(profile, columns=["rho", "log_likelihood"])
profile["lr"] = 2 * (ll_free - profile["log_likelihood"])
inside = profile[profile["lr"] < PROFILE_CUTOFF]
contiguous = bool(len(inside)) and bool((inside.index.to_series().diff().fillna(1) == 1).all())
pl_lo, pl_hi = float(inside["rho"].min()), float(inside["rho"].max())
edge_note = []
if pl_lo <= PROFILE_GRID[0]:
    edge_note.append(f"lower end at the grid edge {PROFILE_GRID[0]}")
if pl_hi >= PROFILE_GRID[-1]:
    edge_note.append(f"upper end at the grid edge {PROFILE_GRID[-1]}")
print(f"\nprofile likelihood for rho ({chosen}, 2023-2025): grid {PROFILE_GRID[0]:.2f} to {PROFILE_GRID[-1]:.2f} step 0.01, all other parameters "
      f"free; free maximum log likelihood {ll_free:.3f} at rho {f_f['rho']:.4f}")
print(profile.iloc[::5].to_string(index=False, float_format="{:.3f}".format))
boot_lo, boot_hi = np.percentile(boot[chosen][:, -1], [2.5, 97.5])
print(f"interval where 2 x (log likelihood drop) < {PROFILE_CUTOFF}: [{pl_lo:.2f}, {pl_hi:.2f}] (grid points inside: {len(inside)}, "
      f"contiguous: {contiguous}" + (f"; {'; '.join(edge_note)}" if edge_note else "") + f"); bootstrap 95% interval [{boot_lo:.4f}, {boot_hi:.4f}]")

written = {}
path = OUT / "send_model_v1_coefficients.csv"
coef[["equation", "term", "estimate", "se", "constraint", "center"]].to_csv(path, index=False)
written[path] = len(coef)
rho_ci = {n: [float(x) for x in np.percentile(boot[n][:, -1], [2.5, 97.5])] for n in boot}
meta = {
    "specification": chosen,
    "model": "bivariate probit with sample selection: send ~ Z (all plays), safe ~ X (sends), correlated errors rho",
    "rho": {n: {"estimate": final[n]["rho"], "interval_95": rho_ci[n], "bootstrap_se": float(boot[n][:, -1].std(ddof=1))} for n in boot},
    "selection_rule": f"SEL-race replaces SEL-linear only if it lowers the summed 2025 log loss (send + safe among sends) by more than {SWITCH_GAIN}",
    "gain_of_race_over_linear": float(gain),
    "seasons": {"evaluation_train": TRAIN, "evaluation_test": TEST, "final_fit": ALL},
    "metrics": {"test_2025": table.reset_index().replace({np.nan: None}).to_dict("records"), "final_in_sample_2023_2025": in_sample},
    "break_even_by_outs": {str(k): v for k, v in p_star.items()},
    "fill_values": json.loads(FILLS.read_text()),
    "launch_fill": "missing launch speed or angle: the season mean for the batted-ball type",
    "spray_angle_convention": SPRAY_CONVENTION,
    "factor_levels": {"fielder_pos": ["LF (reference)", "CF", "RF"], "bb_type": ["ground_ball (reference)", "line_drive", "fly_ball (includes popup)"],
                      "bat_side": ["R (reference)", "L"], "outs": ["0 (reference)", "1", "2"],
                      "score_diff": "clipped to -4..4, 0 is the reference", "inning_group": ["1-3 (reference)", "4-6", "7-8", "9+"],
                      "late_close": "inning 7 or later and score difference within one run"},
    "units": UNITS,
    "centering": {c: {"mean": v[0], "sd": v[1]} for c, v in m_f["scale"].items()},
    "constraints": {k: (">= 0" if v > 0 else "<= 0") for k, v in X_SPECS[chosen.split("-")[1]]["signs"].items()},
    "ridge": RIDGE,
    "bivariate_normal_cdf": {"method": f"Drezner-Wesolowsky, r = sin t, {GL_NODES}-point Gauss-Legendre", "max_abs_error_vs_scipy": max_err},
    "bootstrap": {"replicates": N_BOOT, "unit": "game", "seed": SEED + 1},
    "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__},
}
boot_cols = [f"{e}:{t_}" for e, t_ in zip(coef["equation"], coef["term"]) if e != "rho"] + ["rho"]
boot_df = pd.DataFrame(boot[chosen], columns=boot_cols)
boot_df.insert(0, "replicate", np.arange(1, len(boot_df) + 1))
meta["rho_profile_likelihood"] = {"specification": chosen, "interval_95": [pl_lo, pl_hi], "cutoff": PROFILE_CUTOFF,
                                  "grid": f"rho fixed from {PROFILE_GRID[0]:.2f} to {PROFILE_GRID[-1]:.2f} in steps of 0.01, all other parameters free",
                                  "free_maximum": {"rho": f_f["rho"], "log_likelihood": ll_free}, "grid_points_inside": int(len(inside)),
                                  "contiguous": contiguous, "edge": edge_note, "bootstrap_interval_95": rho_ci[chosen]}
meta["launch_fill_values"] = {f"{s_} {b_}": {"launch_speed": float(v["launch_speed"]), "launch_angle": float(v["launch_angle"])}
                              for (s_, b_), v in launch_fills.iterrows()}
meta["bootstrap_file"] = {"file": "send_model_v1_bootstrap.csv", "replicates": int(len(boot_df)),
                          "columns": "replicate, then equation:term for the send and safe equations in coefficient-file order, then rho; "
                                     "estimates per unit as in send_model_v1_coefficients.csv"}
path = OUT / "send_model_v1_meta.json"
path.write_text(json.dumps(meta, indent=2))
written[path] = 1
path = OUT / "send_model_v1_bootstrap.csv"
boot_df.to_csv(path, index=False)
written[path] = len(boot_df)
print(f"bootstrap file: {path.relative_to(ROOT)}, shape {boot_df.shape} (replicates x columns, including the replicate column)")
t = stage("export", t)

print("\n=== transition tables (Retrosheet population, runner on second only)")
pop = pd.read_parquet(POP)
prim = pop[(pop["base_code"] == 2) & pop["runner_result"].isin(["scored", "out at home", "held at third"])].copy()
folded = prim["batter_advance"].isin(["third", "home"])
prim["batter"] = prim["batter_advance"].replace({"third": "second", "home": "second"})
print(f"batter advances beyond second folded into 'second': {int(folded.sum())} ({prim.loc[folded, 'batter_advance'].value_counts().to_dict()})")
ba = pd.crosstab([prim["runner_result"], prim["outs"]], prim["batter"]).reindex(columns=["first", "second", "out"], fill_value=0)
n = ba.sum(axis=1)
ba = ba.div(n, axis=0)
ba["n"] = n
ba = ba.reindex(["scored", "out at home", "held at third"], level=0)
print("batter's advance by runner result and outs (shares sum to 1 per row):")
print(ba.to_string(float_format="{:.3f}".format))
print(f"check: shares sum to 1 in every row: {bool(np.allclose(ba[['first', 'second', 'out']].sum(axis=1), 1))}")
path = OUT / "send_batter_advance_v1.csv"
ba.reset_index().rename(columns={"runner_result": "runner_result", "outs": "outs"}).to_csv(path, index=False, float_format="%.6f")
written[path] = len(ba)
sec = pop[(pop["base_code"] == 3) & pop["runner_result"].isin(["scored", "out at home", "held at third"])]
tr = pd.crosstab([sec["runner_result"], sec["outs"]], sec["trailing_advance"])
tn = tr.sum(axis=1)
tr = tr.div(tn, axis=0)
tr["n"] = tn
print("\ntrailing runner (from first) by runner result and outs, runners on first and second (printed, not saved):")
print(tr.reindex(["scored", "out at home", "held at third"], level=0).to_string(float_format="{:.3f}".format))
t = stage("transition tables", t)

print("\n=== diagnostics (not part of the production export)")


def above(pr, pstar):
    return pd.Series(pr > np.array([pstar[o] for o in outs])).groupby(outs).mean()


print("\n1. rho sensitivity: SEL-linear on 2023-2025 with rho fixed (all other parameters free); P(safe | X) = Phi(X beta); "
      "share of plays with P(safe | X) above the run break-even p* (0.977, 0.761, 0.412)")
m_lin = final_models["SEL-linear"]
settings = {"rho fixed 0.49": fit_sel(m_lin, df, x0=final["SEL-linear"]["w"], rho_fixed=0.49), "rho free": final["SEL-linear"],
            "rho fixed 0.98": fit_sel(m_lin, df, x0=final["SEL-linear"]["w"], rho_fixed=0.98)}
sens = []
for label, f in settings.items():
    _, _, pr = predict_sel(m_lin, f, df)
    row = {"setting": label, "rho": f["rho"], "log likelihood": sel_loglik(m_lin, f, df)}
    for grp, mask in [("sent", sent_all), ("held", ~sent_all)]:
        for qq, v in zip([10, 25, 50, 75, 90], np.quantile(pr[mask], [0.1, 0.25, 0.5, 0.75, 0.9])):
            row[f"{grp} p{qq}"] = v
        row[f"{grp} mean"] = pr[mask].mean()
    for o, v in above(pr, p_star).items():
        row[f"send > p*, {o} outs"] = v
    sens.append(row)
coach = df.groupby("outs")["sent"].mean()
sens.append({"setting": "coaches' send rate", **{f"send > p*, {o} outs": v for o, v in coach.items()}})
sens = pd.DataFrame(sens).set_index("setting")
print(sens.T.to_string(float_format="{:.3f}".format, na_rep=""))

print("\n2. run break-even with the batter's advance (send_batter_advance_v1.csv and the run expectancy table)")
adv = pd.read_csv(OUT / "send_batter_advance_v1.csv").set_index(["runner_result", "outs"])
re2 = lambda b, o: 0.0 if o >= 3 else float(re.loc[(b, o)])  # noqa: E731
p_adv, vals = {}, []
for o in range(3):
    sc, oh_, hd = adv.loc[("scored", o)], adv.loc[("out at home", o)], adv.loc[("held at third", o)]
    v_scored = 1 + sc["first"] * re2(1, o) + sc["second"] * re2(2, o) + sc["out"] * re2(0, o + 1)
    v_out = oh_["first"] * re2(1, o + 1) + oh_["second"] * re2(2, o + 1) + oh_["out"] * re2(0, o + 2)
    v_held = hd["first"] * re2(5, o) + hd["second"] * re2(6, o) + hd["out"] * re2(4, o + 1)
    p_adv[o] = (v_held - v_out) / (v_scored - v_out)
    vals.append({"outs": o, "scored (runs)": v_scored, "out at home (runs)": v_out, "held (runs)": v_held,
                 "p* with batter's advance": p_adv[o], "p* simple": p_star[o]})
print("  scored: 1 + RE of the batter's state (first: 1__, second: _2_, out: bases empty with one more out); out at home: RE of the batter's "
      "state with one more out (two if the batter is also out; 0 at three outs); held: RE(1_3) if the batter stays at first, RE(_23) if the batter "
      "takes second, RE(__3) with one more out if the batter is out")
print(pd.DataFrame(vals).set_index("outs").to_string(float_format="{:.4f}".format))
san2 = pd.DataFrame({"plays": df.groupby("outs").size(), "coaches' send rate": coach, "p* with advance": pd.Series(p_adv)})
for n_ in ["SEL-linear", "SEL-race", "S-only"]:
    san2[f"{n_}: share with P(safe|X) > p*"] = above(uncond[n_], p_adv)
print("sanity table with the batter's-advance break-even:")
print(san2.to_string(float_format="{:.3f}".format))

print("\n3. win-probability break-even, bottom of the 8th, with the same batter's-advance shares (wp_after, new plate appearance)")
wp_rows = []
for label, bat, fld in [("tied", 3, 3), ("trailing by one", 2, 3)]:
    for o in range(3):
        st = GameState(season=2025, game_type="regular", inning=8, half=1, bat_score=bat, fld_score=fld, outs=o, balls=0, strikes=0,
                       runner2="runner_on_second", batter="batter", batter_hand="R", pitcher="pitcher", pitcher_hand="R", catcher="catcher")
        w_ = lambda base, outs_post, runs: wp_after(st, base if outs_post < 3 else 0, min(outs_post, 3), runs, new_pa=True)  # noqa: E731
        sc, oh_, hd = adv.loc[("scored", o)], adv.loc[("out at home", o)], adv.loc[("held at third", o)]
        v_scored = sc["first"] * w_(1, o, 1) + sc["second"] * w_(2, o, 1) + sc["out"] * w_(0, o + 1, 1)
        v_out = oh_["first"] * w_(1, o + 1, 0) + oh_["second"] * w_(2, o + 1, 0) + oh_["out"] * w_(0, o + 2, 0)
        v_held = hd["first"] * w_(5, o, 0) + hd["second"] * w_(6, o, 0) + hd["out"] * w_(4, o + 1, 0)
        wp_rows.append({"state": f"bottom 8th, {label}", "outs": o, "WP scored": v_scored, "WP out at home": v_out, "WP held": v_held,
                        "p* (win probability)": (v_held - v_out) / (v_scored - v_out), "p* (runs, with advance)": p_adv[o]})
wpt = pd.DataFrame(wp_rows).set_index(["state", "outs"])
wpt["win-probability p* minus run p*"] = wpt["p* (win probability)"] - wpt["p* (runs, with advance)"]
print(wpt.to_string(float_format="{:.4f}".format))

print("\n4. coefficient stability, SEL-linear safe equation: fitted on 2023-2024 against 2023-2025 (per unit; the intercept is at each "
      "fit's own centers, so it is not comparable)")
c_tr = coef_table(models["SEL-linear"], fits["SEL-linear"]).query("equation != 'send'").set_index("term")
c_all = coef_table(final_models["SEL-linear"], final["SEL-linear"])
c_all["se"] = boot["SEL-linear"].std(axis=0, ddof=1)
c_all = c_all.query("equation != 'send'").set_index("term")
stab = pd.DataFrame({"2023-2024": c_tr["estimate"], "2023-2025": c_all["estimate"], "SE (2023-2025)": c_all["se"]})
stab["change"] = stab["2023-2025"] - stab["2023-2024"]
stab["change / SE"] = stab["change"] / stab["SE (2023-2025)"]
stab["constraint"] = c_all["constraint"]
stab = stab.reindex(c_all.index)
print(stab.to_string(float_format="{:.5f}".format))
moved = stab.drop(index="intercept")
print("terms that moved by more than one bootstrap SE: " + (", ".join(f"{t_} ({v:+.2f} SE)" for t_, v in moved["change / SE"].items() if abs(v) > 1) or "none"))
t = stage("diagnostics (sensitivity)", t)

print("\nfiles:")
for path, n in written.items():
    print(f"  {path.relative_to(ROOT)}: {n:,} rows" + (" (one JSON object)" if path.suffix == ".json" else ""))
print("runtime by stage:")
for k, v in STAGES.items():
    print(f"  {k}: {v:.1f} s")
print(f"  total: {time.time() - T0:.1f} s")
