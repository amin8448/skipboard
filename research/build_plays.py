import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

SEASONS = range(2016, 2026)
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "retrosheet"
OUT = ROOT / "data" / "derived" / "plays_regular_2016_2025.parquet"

FLAGS = [
    "single", "double", "triple", "hr", "walk", "hbp", "k", "iw", "gdp", "othdp", "tp",
    "sh", "sf", "bunt", "ground", "fly", "line", "wp", "pb", "bk",
    "sb2", "sb3", "sbh", "cs2", "cs3", "csh", "pko1", "pko2", "pko3",
]
KEEP = [
    "gid", "date", "inning", "top_bot", "batteam", "pitteam", "batter", "pitcher",
    "bathand", "pithand", "balls", "strikes", "pitches", "nump", "pa", "event",
    "outs_pre", "outs_post", "br1_pre", "br2_pre", "br3_pre", "br1_post", "br2_post", "br3_post",
    "score_v", "score_h", "runs",
] + FLAGS
# vis_home is kept because score_v/score_h are visitor/home, and in a few games the home
# team bats in the top half, so top_bot alone cannot say whose score is whose.
READ = KEEP + ["vis_home", "gametype"]

frames = []
for year in SEASONS:
    with zipfile.ZipFile(RAW / f"{year}plays.zip") as z:
        with z.open(f"{year}plays.csv") as f:
            df = pd.read_csv(f, usecols=READ, low_memory=False)
    df = df[df["gametype"] == "regular"].drop(columns="gametype")
    df["file_season"] = year
    frames.append(df)

# Concatenating in season order keeps Retrosheet's original row (play) order; nothing is sorted.
plays = pd.concat(frames, ignore_index=True)
plays["season"] = plays["date"] // 10000

plays["base_pre"] = (
    plays["br1_pre"].notna() * 1 + plays["br2_pre"].notna() * 2 + plays["br3_pre"].notna() * 4
)
plays["base_post"] = (
    plays["br1_post"].notna() * 1 + plays["br2_post"].notna() * 2 + plays["br3_post"].notna() * 4
)

home_bat = plays["vis_home"] == 1
plays["bat_score_pre"] = np.where(home_bat, plays["score_h"], plays["score_v"])
plays["fld_score_pre"] = np.where(home_bat, plays["score_v"], plays["score_h"])

plays["half_id"] = (
    plays["gid"] + "_" + plays["inning"].astype(str) + "_" + plays["top_bot"].astype(str)
)
half_runs = plays.groupby("half_id", sort=False)["runs"]
plays["runs_rest"] = half_runs.transform("sum") - half_runs.cumsum() + plays["runs"]

plays = plays[
    ["season"] + KEEP[:4] + ["vis_home"] + KEEP[4:]
    + ["base_pre", "base_post", "bat_score_pre", "fld_score_pre", "half_id", "runs_rest", "file_season"]
]


def report(label, fails, total):
    print(f"  {label}: {fails:,} of {total:,} ({fails / total:.4%})")


print("rows and plate appearances per season (regular season only)")
per_season = plays.groupby("season").agg(rows=("gid", "size"), pa=("pa", "sum"), games=("gid", "nunique"))
print(per_season.to_string())
print(f"total: {len(plays):,} rows, {plays['pa'].sum():,} PA, {plays['gid'].nunique():,} games")

print("\nstructure")
report("rows whose season from date differs from the file season", int((plays["season"] != plays["file_season"]).sum()), len(plays))
game_blocks = int((plays["gid"] != plays["gid"].shift()).sum())
half_blocks = int((plays["half_id"] != plays["half_id"].shift()).sum())
print(f"  games stored as one contiguous block: {game_blocks == plays['gid'].nunique()} ({game_blocks:,} blocks, {plays['gid'].nunique():,} games)")
print(f"  half-innings stored as one contiguous block: {half_blocks == plays['half_id'].nunique()} ({half_blocks:,} blocks, {plays['half_id'].nunique():,} half-innings)")
print(f"  rows where top_bot != vis_home: {int((plays['top_bot'] != plays['vis_home']).sum()):,} in {plays.loc[plays['top_bot'] != plays['vis_home'], 'gid'].nunique()} games")

prev = plays.shift()
same_game = plays["gid"] == prev["gid"]
same_half = plays["half_id"] == prev["half_id"]

print("\ncheck a: score before each play = previous score before + previous runs (within game)")
exp_v = prev["score_v"] + np.where(prev["vis_home"] == 0, prev["runs"], 0)
exp_h = prev["score_h"] + np.where(prev["vis_home"] == 1, prev["runs"], 0)
fail_a = same_game & ((plays["score_v"] != exp_v) | (plays["score_h"] != exp_h))
report("failures", int(fail_a.sum()), int(same_game.sum()))
if fail_a.any():
    print(f"  games affected: {plays.loc[fail_a, 'gid'].nunique()}, first few: {plays.loc[fail_a, 'gid'].unique()[:5].tolist()}")

print("\ncheck b: continuity within half-inning")
fail_outs = same_half & (plays["outs_pre"] != prev["outs_post"])
fail_base = same_half & (plays["base_pre"] != prev["base_post"])
report("outs_pre != previous outs_post", int(fail_outs.sum()), int(same_half.sum()))
report("base_pre != previous base_post", int(fail_base.sum()), int(same_half.sum()))
for label, fail in [("outs", fail_outs), ("base", fail_base)]:
    if fail.any():
        print(f"  {label} failures in {plays.loc[fail, 'half_id'].nunique()} half-innings, first few: {plays.loc[fail, 'half_id'].unique()[:5].tolist()}")

print("\ncheck c: how half-innings end")
end = plays[plays["half_id"] != plays["half_id"].shift(-1)].copy()
end["game_last"] = end["gid"] != end["gid"].shift(-1)
n_halves = len(end)

not_last = end[~end["game_last"]]
bad_mid = not_last[not_last["outs_post"] != 3]
report("non-final half-innings not ending with 3 outs", len(bad_mid), len(not_last))
if len(bad_mid):
    print(bad_mid[["half_id", "outs_post", "event", "bat_score_pre", "fld_score_pre", "runs"]].head(10).to_string(index=False))

last = end[end["game_last"]].copy()
short = last[last["outs_post"] < 3].copy()
print(f"  final half-innings: {len(last):,}; ending with 3 outs: {int((last['outs_post'] == 3).sum()):,}; fewer than 3 outs: {len(short):,}")
bat_post = short["bat_score_pre"] + short["runs"]
fld_post = short["fld_score_pre"]
lead_pre = np.sign(short["bat_score_pre"] - short["fld_score_pre"])
lead_post = np.sign(bat_post - fld_post)
walkoff = (short["top_bot"] == 1) & (bat_post > fld_post)
early = ~walkoff & ((short["inning"] < 9) | (lead_pre == lead_post))
short["kind"] = np.select([walkoff, early], ["walk-off", "game ended early"], "other")
for kind in ["walk-off", "game ended early", "other"]:
    report(kind, int((short["kind"] == kind).sum()), len(short))
early_rows = short[short["kind"] == "game ended early"]
if len(early_rows):
    print(f"    of which fewer than 9 innings: {int((early_rows['inning'] < 9).sum()):,}; 9+ innings with no lead change on the final play: {int((early_rows['inning'] >= 9).sum()):,}")
other = short[short["kind"] == "other"]
if len(other):
    print(other[["half_id", "outs_post", "event", "bat_score_pre", "fld_score_pre", "runs"]].head(10).to_string(index=False))
print(f"  total half-innings: {n_halves:,}")

OUT.parent.mkdir(parents=True, exist_ok=True)
plays.drop(columns="file_season").to_parquet(OUT, index=False)
print(f"\nsaved {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e6:.1f} MB, {len(plays):,} rows, {plays.shape[1] - 1} columns)")
