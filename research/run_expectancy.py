from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PLAYS = ROOT / "data" / "derived" / "plays_regular_2016_2025.parquet"
OUT = ROOT / "models" / "run_expectancy" / "re24_v1.csv"

BASE_ORDER = [0, 1, 2, 4, 3, 5, 6, 7]
BASE_LABEL = {0: "___", 1: "1__", 2: "_2_", 4: "__3", 3: "12_", 5: "1_3", 6: "_23", 7: "123"}
N_BOOT = 500
SEED = 2026

pd.set_option("display.width", 200)

plays = pd.read_parquet(
    PLAYS,
    columns=["season", "gid", "inning", "half_id", "event", "pa", "outs_pre", "outs_post", "base_pre", "base_post", "runs", "runs_rest"],
)


def prepare(first, last):
    df = plays[plays["season"].between(first, last) & plays["inning"].between(1, 8)]
    # Complete half-innings only, judged on the last row before NP rows are dropped.
    ends = df.groupby("half_id", sort=False)["outs_post"].last()
    df = df[df["half_id"].isin(ends.index[ends == 3])]
    n_halves = df["half_id"].nunique()
    df = df[df["event"] != "NP"]
    changed = (df["base_pre"] != df["base_post"]) | (df["outs_pre"] != df["outs_post"]) | (df["runs"] > 0)
    print(f"{first}-{last}: {n_halves:,} complete half-innings in innings 1-8, {len(df):,} rows after dropping NP, {int(changed.sum()):,} event rows, {int((df['pa'] == 1).sum()):,} PA rows")
    return df[changed], df[df["pa"] == 1]


def re_table(df):
    t = df.groupby(["base_pre", "outs_pre"])["runs_rest"].agg(re="mean", n="size")
    t.index.names = ["base_code", "outs"]
    return t


def grid(series, fmt):
    g = series.unstack("outs").reindex(BASE_ORDER)
    g.index = [BASE_LABEL[b] for b in g.index]
    g.index.name = "bases"
    return g.map(fmt)


events, pas = prepare(2023, 2025)
events_old, _ = prepare(2016, 2019)

re = re_table(events)
re_pa = re_table(pas)
re_old = re_table(events_old)

print("\nevent-based vs PA-only, 2023-2025 (RE, event minus PA)")
diff_pa = re["re"] - re_pa["re"]
print(grid(diff_pa, "{:+.3f}".format).to_string())
worst = diff_pa.abs().idxmax()
print(f"largest difference: {diff_pa[worst]:+.4f} runs at {BASE_LABEL[worst[0]]}, {worst[1]} outs "
      f"(event {re.loc[worst, 're']:.4f}, n={re.loc[worst, 'n']:,}; PA {re_pa.loc[worst, 're']:.4f}, n={re_pa.loc[worst, 'n']:,})")

# Bootstrap: per-game sums and counts for each of the 24 cells, then resample games with replacement.
cells = [(b, o) for b in BASE_ORDER for o in range(3)]
cell_idx = {c: i for i, c in enumerate(cells)}


def bootstrap_se(df):
    cell = np.array([cell_idx[c] for c in zip(df["base_pre"], df["outs_pre"])])
    game_codes, games = pd.factorize(df["gid"])
    sums = np.zeros((len(games), 24))
    counts = np.zeros((len(games), 24))
    np.add.at(sums, (game_codes, cell), df["runs_rest"].to_numpy())
    np.add.at(counts, (game_codes, cell), 1)
    rng = np.random.default_rng(SEED)
    boot = np.empty((N_BOOT, 24))
    for i in range(N_BOOT):
        w = np.bincount(rng.integers(0, len(games), len(games)), minlength=len(games))
        boot[i] = (w @ sums) / (w @ counts)
    se = pd.Series(boot.std(axis=0, ddof=1), index=pd.MultiIndex.from_tuples(cells, names=["base_code", "outs"]))
    return se, len(games)


se, n_games = bootstrap_se(events)
se_old, _ = bootstrap_se(events_old)
print(f"\nbootstrap standard errors, 2023-2025 ({N_BOOT} replicates over {n_games:,} games, seed {SEED})")
print(grid(se, "{:.4f}".format).to_string())

print("\n2016-2019 run expectancy")
print(grid(re_old["re"], "{:.3f}".format).to_string())
print("\ndifference, 2023-2025 minus 2016-2019, then as z = difference / sqrt(se_2023-2025^2 + se_2016-2019^2), both bootstrapped by game")
diff_era = re["re"] - re_old["re"]
print(grid(diff_era, "{:+.3f}".format).to_string())
print(grid(diff_era / np.sqrt(se**2 + se_old**2), "{:+.1f}".format).to_string())

print("\n2023-2025 run expectancy (RE / n)")
print(grid(re["re"], "{:.3f}".format).to_string())
print(grid(re["n"], "{:,}".format).to_string())

out = re.join(se.rename("se")).reset_index()
out["base_label"] = out["base_code"].map(BASE_LABEL)
out["order"] = out["base_code"].map(BASE_ORDER.index)
out = out.sort_values(["order", "outs"])[["base_code", "base_label", "outs", "re", "se", "n"]]
out["re"] = out["re"].round(4)
out["se"] = out["se"].round(4)
OUT.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT, index=False)
print(f"\nsaved {OUT.relative_to(ROOT)} ({len(out)} rows)")
print(out.to_string(index=False))
