from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PLAYS = ROOT / "data" / "derived" / "plays_regular_2016_2025.parquet"
OUT = ROOT / "models" / "win_probability"

BASE_ORDER = [0, 1, 2, 4, 3, 5, 6, 7]
BASE_LABEL = {0: "___", 1: "1__", 2: "_2_", 4: "__3", 3: "12_", 5: "1_3", 6: "_23", 7: "123"}
MAX_R = 10  # r = 10 is the bin for 10 or more runs
D_MAX = 20  # d is clamped to -20..20
CAP = 25  # last inning simulated; a tie after its bottom half counts as 0.5
EMPTY = 0  # cell index base_code * 3 + outs: bases empty, 0 outs
GHOST = 2 * 3 + 0  # runner on second, 0 outs
D = np.arange(-D_MAX, D_MAX + 1)

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)

plays = pd.read_parquet(PLAYS)


# Step 1: run distributions, same filter as research/run_expectancy.py.
def run_dist(first, last):
    df = plays[plays["season"].between(first, last) & plays["inning"].between(1, 8)]
    ends = df.groupby("half_id", sort=False)["outs_post"].last()
    df = df[df["half_id"].isin(ends.index[ends == 3])]
    df = df[df["event"] != "NP"]
    changed = (df["base_pre"] != df["base_post"]) | (df["outs_pre"] != df["outs_post"]) | (df["runs"] > 0)
    df = df[changed]
    r = df["runs_rest"].clip(upper=MAX_R)
    counts = pd.crosstab([df["top_bot"], df["base_pre"], df["outs_pre"]], r).reindex(columns=range(MAX_R + 1), fill_value=0)
    full = pd.MultiIndex.from_product([[0, 1], range(8), range(3)], names=["half", "base_code", "outs"])
    counts.index.names = ["half", "base_code", "outs"]
    counts = counts.reindex(full, fill_value=0)
    n = counts.sum(axis=1)
    p = counts.div(n, axis=0)
    print(f"\nrun distributions {first}-{last}: {len(df):,} event rows, {df['half_id'].nunique():,} complete half-innings, smallest cell n = {n.min():,}")
    # P array indexed [half, base_code * 3 + outs, r]
    return p.to_numpy().reshape(2, 24, MAX_R + 1), p, n


def print_dist(p, n):
    t = p.copy()
    t.columns = [str(c) if c < MAX_R else f"{MAX_R}+" for c in t.columns]
    t = t.map("{:.3f}".format)
    t["mean"] = (p * np.arange(MAX_R + 1)).sum(axis=1).map("{:.3f}".format)
    t["n"] = n.map("{:,}".format)
    t = t.reset_index()
    t["order"] = t["base_code"].map(BASE_ORDER.index)
    t["bases"] = t["base_code"].map(BASE_LABEL)
    t["half"] = t["half"].map({0: "top", 1: "bot"})
    t = t.sort_values(["half", "order", "outs"], ascending=[False, True, True])
    print(t.drop(columns=["base_code", "order"]).set_index(["half", "bases", "outs"]).to_string())


# Step 2: win probability recursion. W[inning, half, d + D_MAX, cell] is the probability that
# the team batting second wins, where d = its runs minus the runs of the team batting first.
def solve(P, ghost):
    W = np.zeros((CAP + 2, 2, len(D), 24))
    start = np.zeros((CAP + 2, 2, len(D)))
    start[CAP + 1, 0, :] = 0.5
    r = np.arange(MAX_R + 1)
    for inning in range(CAP, 0, -1):
        cell0 = GHOST if (ghost and inning >= 10) else EMPTY
        # bottom half
        d_after = np.clip(D[:, None] + r[None, :], -D_MAX, D_MAX)
        nxt = start[inning + 1, 0]
        if inning >= 9:
            after = np.where(d_after > 0, 1.0, np.where(d_after < 0, 0.0, nxt[D_MAX]))
        else:
            after = nxt[d_after + D_MAX]
        W[inning, 1] = after @ P[1].T
        start[inning, 1] = W[inning, 1][:, cell0]
        # top half
        d_after = np.clip(D[:, None] - r[None, :], -D_MAX, D_MAX)
        nxt = start[inning, 1]
        after = nxt[d_after + D_MAX]
        if inning >= 9:
            after = np.where(d_after > 0, 1.0, after)
        W[inning, 0] = after @ P[0].T
        start[inning, 0] = W[inning, 0][:, cell0]
    return W, start


def wp_bat(W, inning, half, diff, base, outs):
    inning = np.minimum(inning, CAP)
    d = np.clip(np.where(half == 0, -diff, diff), -D_MAX, D_MAX)
    w = W[inning, half, d + D_MAX, base * 3 + outs]
    return np.where(half == 0, 1 - w, w)


def convergence(start, label):
    tie = start[10:CAP + 1, 0, D_MAX]
    step = np.abs(np.diff(start[10:CAP + 1, 0], axis=0)).max(axis=1)
    print(f"  {label}: team batting second at start of tied 10th = {tie[0]:.6f}, tied 24th = {tie[-2]:.6f}; "
          f"max change between consecutive extra innings (any d): 10->11 {step[0]:.2e}, 23->24 {step[-2]:.2e}")


def wp_table(W):
    inning, half, diff, base, outs = np.meshgrid(np.arange(1, 13), [0, 1], np.arange(-10, 11), BASE_ORDER, range(3), indexing="ij")
    t = pd.DataFrame({"inning": inning.ravel(), "half": half.ravel(), "diff": diff.ravel(), "base_code": base.ravel(), "outs": outs.ravel()})
    t["wp_bat"] = wp_bat(W, t["inning"].to_numpy(), t["half"].to_numpy(), t["diff"].to_numpy(), t["base_code"].to_numpy(), t["outs"].to_numpy())
    return t


def print_start_grid(W, ghost, label):
    rows = {}
    for inning in range(1, 13):
        cell_base = 2 if (ghost and inning >= 10) else 0
        for half in (0, 1):
            vals = wp_bat(W, np.full(11, inning), np.full(11, half), np.arange(-5, 6), np.full(11, cell_base), np.zeros(11, dtype=int))
            rows[f"{'top' if half == 0 else 'bot'} {inning}{' (_2_)' if cell_base else ''}"] = vals
    g = pd.DataFrame(rows, index=[f"{d:+d}" for d in range(-5, 6)]).T.map("{:.3f}".format)
    print(f"\n{label}: wp_bat at the start of each half-inning, by batting team score difference")
    print(g.to_string())


P_val, p_val, n_val = run_dist(2023, 2024)
print_dist(p_val, n_val)
P_fin, p_fin, n_fin = run_dist(2023, 2025)
print_dist(p_fin, n_fin)

W_val, start_val = solve(P_val, ghost=True)
W_reg, start_reg = solve(P_fin, ghost=True)
W_post, start_post = solve(P_fin, ghost=False)
print("\nconvergence of extra innings")
convergence(start_val, "regular, 2023-2024 tables")
convergence(start_reg, "regular, 2023-2025 tables")
convergence(start_post, "postseason, 2023-2025 tables")
print_start_grid(W_reg, True, "regular season, 2023-2025 tables")
print_start_grid(W_post, False, "postseason, 2023-2025 tables")

# Step 3: validation on 2025, tables from 2023-2024.
g25 = plays[plays["season"] == 2025]
team_runs = g25.groupby(["gid", "batteam"])["runs"].sum().unstack()
final = g25.groupby("gid").tail(1)
final_v = final["score_v"] + np.where(final["vis_home"] == 0, final["runs"], 0)
final_h = final["score_h"] + np.where(final["vis_home"] == 1, final["runs"], 0)
runs_total = g25.groupby("gid")["runs"].sum()
mismatch = int((runs_total.loc[final["gid"]].to_numpy() != (final_v + final_h).to_numpy()).sum())
winner = team_runs.idxmax(axis=1)
ties = int((team_runs.max(axis=1) == team_runs.min(axis=1)).sum())
second = g25[g25["top_bot"] == 1].groupby("gid")["batteam"].first()
print(f"\n2025 validation: {len(winner):,} games; final-score check against summed runs: {mismatch} mismatches; tied games: {ties}")

test = g25[(g25["inning"] <= 12) & (g25["event"] != "NP")].copy()
test["won"] = (test["batteam"].to_numpy() == winner.loc[test["gid"]].to_numpy()).astype(int)
test["wp"] = wp_bat(W_val, test["inning"].to_numpy(), test["top_bot"].to_numpy(),
                    (test["bat_score_pre"] - test["fld_score_pre"]).to_numpy(), test["base_pre"].to_numpy(), test["outs_pre"].to_numpy())
print(f"  plays tested (innings 1-12, NP rows dropped): {len(test):,}")

test["bin"] = np.minimum((test["wp"] * 10).astype(int), 9)
cal = test.groupby("bin").agg(mean_pred=("wp", "mean"), actual=("won", "mean"), n=("won", "size"))
cal.index = [f"{b / 10:.1f}-{(b + 1) / 10:.1f}" for b in cal.index]
cal["gap"] = cal["actual"] - cal["mean_pred"]
print("\ncalibration (10 bins of predicted wp_bat)")
print(cal.to_string(formatters={"mean_pred": "{:.3f}".format, "actual": "{:.3f}".format, "gap": "{:+.3f}".format, "n": "{:,}".format}))

test["sq"] = (test["wp"] - test["won"]) ** 2
groups = pd.cut(test["inning"], [0, 3, 6, 9, 12], labels=["1-3", "4-6", "7-9", "10-12"])
print(f"\nBrier score overall: {test['sq'].mean():.4f} (n={len(test):,}); constant 0.5 forecast would score 0.2500")
for label, sub in test.groupby(groups, observed=True):
    print(f"  innings {label}: {sub['sq'].mean():.4f} (n={len(sub):,})")

second_won = (second == winner.loc[second.index]).mean()
model_start = start_val[1, 0, D_MAX]
print(f"\nstart of game, team batting second: model {model_start:.4f}, actual 2025 {second_won:.4f} (n={len(second):,} games)")
extra = g25[g25["inning"] >= 10]["gid"].unique()
extra_second_won = (second.loc[extra] == winner.loc[extra]).mean()
print(f"start of tied 10th, team batting second: model {start_val[10, 0, D_MAX]:.4f}, actual 2025 {extra_second_won:.4f} (n={len(extra):,} extra-inning games)")

# Step 4: spot checks, final regular-season version.
checks = [
    ("top 1st, tied, ___, 0 outs", 1, 0, 0, 0, 0),
    ("bottom 9th, tied, ___, 0 outs", 9, 1, 0, 0, 0),
    ("bottom 9th, tied, _2_, 0 outs", 9, 1, 0, 2, 0),
    ("bottom 9th, down 1, 1__, 0 outs", 9, 1, -1, 1, 0),
    ("top 9th, down 1, 1__, 1 out", 9, 0, -1, 1, 1),
    ("bottom 7th, tied, 1__, 1 out", 7, 1, 0, 1, 1),
    ("top 10th, tied, _2_, 0 outs", 10, 0, 0, 2, 0),
    ("bottom 10th, down 1, _2_, 0 outs", 10, 1, -1, 2, 0),
]
print("\nspot checks (regular season, 2023-2025 tables)")
for i, (label, inning, half, diff, base, outs) in enumerate(checks, 1):
    print(f"  {i}. {label:<34} wp_bat = {float(wp_bat(W_reg, inning, half, diff, base, outs)):.4f}")

# Step 5: save.
OUT.mkdir(parents=True, exist_ok=True)
dist = p_fin.stack().rename("p").reset_index().rename(columns={"runs_rest": "r"})
dist["n"] = n_fin.loc[list(zip(dist["half"], dist["base_code"], dist["outs"]))].to_numpy()
dist["p"] = dist["p"].round(6)
dist.to_csv(OUT / "run_dist_v1.csv", index=False)
for name, W in [("wp_regular_v1.csv", W_reg), ("wp_postseason_v1.csv", W_post)]:
    t = wp_table(W)
    t["wp_bat"] = t["wp_bat"].round(6)
    t.to_csv(OUT / name, index=False)
    print(f"saved models/win_probability/{name} ({len(t):,} rows)")
print(f"saved models/win_probability/run_dist_v1.csv ({len(dist):,} rows; r = {MAX_R} means {MAX_R} or more)")
