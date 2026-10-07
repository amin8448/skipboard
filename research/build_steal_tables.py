import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pitch_codes import BALLS, ENDS, FOUL_BUNT, FOULS, NOT_THROWN, PITCHES, STRIKES, count_after  # noqa: E402
from steal_pitches import PITCH_LIKE, add_segments, going_at, pitch_rows  # noqa: E402

T0 = time.time()
ROOT = Path(__file__).resolve().parent.parent
PLAYS = ROOT / "data" / "derived" / "plays_regular_2016_2025.parquet"
PA = ROOT / "data" / "derived" / "pa_outcomes_2016_2025.parquet"
S1 = ROOT / "data" / "derived" / "steal_attempts_2023_2025.parquet"
STD_ADV = ROOT / "models" / "pa_model" / "advancement_v1.csv"
STD_SPLIT = ROOT / "models" / "pa_model" / "single_split_v1.csv"
OUT = ROOT / "data" / "derived" / "steal"
MODEL_OUT = ROOT / "models" / "steal"
STD_FC = ROOT / "models" / "pa_model" / "fc_share_v1.csv"
RECENT = "2023-2025"
WIDE = "2016-2025 excl 2019"
WINDOW = {"2023_2025": RECENT, "2016_2025": WIDE}

GOING_BASES = (1, 5)  # runner on first, second open
RESULTS = ["ball", "ball_four", "hit_by_pitch", "strike", "strike_three", "foul", "in_play", "other"]
PITCHOUT = set("PQRY")
SWINGS = set("SFTXMQRYLO")  # swinging strike, foul, foul tip, in play, missed bunt, swings at pitchouts, foul bunts
IN_PLAY_CATS = ["1B_IF", "1B_OF", "2B", "3B", "HR", "GB_OUT", "AIR_OUT", "ROE", "FC", "OTHER"]
ADV_COLS = ["base_pre", "outs_pre", "category", "base_post", "outs_post", "runs", "k", "n", "p", "n_cell", "source"]
MIN_N = 30
LABEL = {0: "___", 1: "1__", 2: "_2_", 4: "__3", 3: "12_", 5: "1_3", 6: "_23", 7: "123"}

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 40)
OUT.mkdir(parents=True, exist_ok=True)
MODEL_OUT.mkdir(parents=True, exist_ok=True)
written = {}
print(f"windows: {RECENT}, and the wider window 2016 to 2025 with 2019 excluded because its files carry no runner-going markers, labelled '{WIDE}'")


def result_category(code, balls, strikes):
    if code in BALLS:
        return "ball_four" if balls == 3 else "ball"
    if code == "H":
        return "hit_by_pitch"
    if code in STRIKES:
        return "strike_three" if strikes == 2 else "strike"
    if code == FOUL_BUNT:
        return "strike_three" if strikes == 2 else "foul"
    if code in FOULS:
        return "foul"
    if code in ENDS:
        return "in_play"
    return "other"


def event_category(row):
    ev = row.event
    if row.cs2 == 1 and "CS2(" in ev and "E" in ev.split("CS2(", 1)[1].split(")", 1)[0]:
        return "CS2_E"
    if row.sb2 == 1:
        return "SB2"
    if row.cs2 == 1:
        return "CS2"
    if row.sbh == 1:
        return "SBH"
    if row.sb3 == 1:
        return "SB3"
    if ev.startswith("DI"):
        return "DI"
    if row.wp == 1:
        return "WP"
    if row.pb == 1:
        return "PB"
    if row.bk == 1:
        return "BK"
    if ev.startswith("OA"):
        return "OA"
    if ev.startswith("PO"):
        return "PO"
    return "other"


plays = pd.read_parquet(PLAYS)
pa_out = pd.read_parquet(PA)
pa_rows = plays.index[plays["pa"] == 1]
aligned = (plays.loc[pa_rows, "gid"].to_numpy() == pa_out["gid"].to_numpy()).all() and (plays.loc[pa_rows, "date"].to_numpy() == pa_out["date"].to_numpy()).all()
print(f"plays: {len(plays):,} rows; PA outcomes aligned with the plays table's PA rows: {aligned}")
if not aligned:
    raise SystemExit("PA outcomes do not line up with the plays table")
plays["pa_category"] = None
plays.loc[pa_rows, "pa_category"] = pa_out["category"].to_numpy()
plays["pa_excluded"] = False
plays.loc[pa_rows, "pa_excluded"] = pa_out["excluded"].to_numpy()
plays["event_cat"] = [event_category(r) for r in plays[["event", "sb2", "cs2", "sbh", "sb3", "wp", "pb", "bk"]].itertuples()]
plays["seq"] = plays["pitches"].fillna("")

n_mismatch = add_segments(plays)
plays["pa_group_excluded"] = plays.groupby("pa_group")["pa_excluded"].transform("max").astype(bool)
print(f"rows whose sequence does not extend the previous row's in the same PA (segment starts at the common prefix): {n_mismatch:,}")

rows = plays[plays["base_pre"].isin(GOING_BASES) & plays["seq"].str.len().gt(0)]
pitch = pitch_rows(rows)
row_cols = ["gid", "date", "season", "inning", "top_bot", "outs_pre", "base_pre", "batter", "bathand", "pitcher", "pithand", "f2",
            "br1_pre", "event", "event_cat", "pa_category", "pa_excluded", "pa_group_excluded", "sb2", "cs2", "seq"]
pitch = pitch.join(plays[row_cols], on="row")
pitch["result"] = [result_category(c, b, s) for c, b, s in zip(pitch["code"], pitch["balls"], pitch["strikes"])]
pitch["pitchout"] = pitch["code"].isin(PITCHOUT)
pitch["bat_side"] = np.where(pitch["bathand"] == "B", np.where(pitch["pithand"] == "R", "L", "R"), pitch["bathand"])
steal_on_end = np.select([pitch["event_cat"].isin(["SB2", "CS2", "CS2_E"])], [pitch["event_cat"]], "pa_end")
pitch["following_event"] = np.where(pitch["pa_ended"], steal_on_end, np.where(pitch["last_play"], pitch["event_cat"], "none"))
pitch["pa_outcome"] = np.where(pitch["pa_ended"], pitch["pa_category"], None)
print(f"pitches and no-pitches in rows with a runner on first and second open: {len(pitch):,}; with '>': {int(pitch['going'].sum()):,}")

# A. Per-pitch table of '>' pitches. Throw situations: the catcher threw to second on the pitch. The runner caught
# on the catcher's throw to first after the pitch is not a throw situation.
pitch["caught_on_catcher_throw"] = (pitch["going"] & pitch["catcher_throw_after"] & ~pitch["pa_ended"] & (pitch["following_event"] == "CS2")).astype(int)
pitch["throw_situation"] = (pitch["result"].isin(["ball", "strike", "strike_three"]) & pitch["following_event"].isin(["SB2", "CS2", "CS2_E"])
                            & (pitch["caught_on_catcher_throw"] == 0)).astype(int)
pitch["throw_safe"] = np.where(pitch["throw_situation"] == 1, pitch["following_event"].isin(["SB2", "CS2_E"]).astype(float), np.nan)
going = pitch[pitch["going"]].copy()
table = going.rename(columns={"top_bot": "half", "outs_pre": "outs", "base_pre": "base_code", "pithand": "pitcher_hand", "f2": "catcher",
                              "br1_pre": "runner_on_first", "code": "pitch_code", "seq": "pitches"})
table = table[["gid", "date", "season", "inning", "half", "outs", "balls", "strikes", "base_code", "batter", "bathand", "bat_side",
               "pitcher", "pitcher_hand", "catcher", "runner_on_first", "pitch_code", "result", "pitchout", "pitcher_pickoffs_before",
               "catcher_pickoffs_before", "pitch_number", "pa_ended", "pa_outcome", "following_event", "event", "pitches",
               "throw_situation", "throw_safe", "caught_on_catcher_throw"]]
path = OUT / "going_pitches_2016_2025.parquet"
table.to_parquet(path, index=False)
written[path] = len(table)

print("\n=== A. '>' pitches")
print("raw code to result category (category depends on the pre-pitch count for balls, strikes and foul bunts):")
print(pd.crosstab(going["code"], going["result"]).reindex(columns=RESULTS, fill_value=0).to_string())
print("\nraw code frequencies on '>' pitches:")
print(going["code"].value_counts().to_string())
print(f"\nconsistency: strike_three pitches that did not end the PA: {int(((going['result'] == 'strike_three') & ~going['pa_ended']).sum())}; "
      f"ball_four pitches that did not end the PA: {int(((going['result'] == 'ball_four') & ~going['pa_ended']).sum())}; "
      f"in_play pitches that did not end the PA: {int(((going['result'] == 'in_play') & ~going['pa_ended']).sum())}")
unattached = {}
for seq, start in zip(plays["seq"], plays["seg_start"]):
    for i in range(start, len(seq)):
        if seq[i] == ">":
            rest = seq[i + 1:].lstrip("*>")
            nxt = rest[:1] or "end"
            if nxt not in PITCH_LIKE:
                unattached[nxt] = unattached.get(nxt, 0) + 1
print(f"'>' markers (all base states, each counted once) not followed by a pitch or no-pitch: {sum(unattached.values())} {unattached}")
print("following event on '>' pitches (all seasons):", going["following_event"].value_counts().to_dict())

print("\n=== B. diagnostics")
print("B1. SB2 and CS2 events with a runner on first and second open: share whose steal pitch carries '>'")
steals = plays[plays["base_pre"].isin(GOING_BASES) & ((plays["sb2"] == 1) | (plays["cs2"] == 1))]
play_chars = PITCHES | set("123N")
info = []
for seq in steals["seq"]:
    idx = max((i for i, ch in enumerate(seq) if ch in play_chars), default=None)
    on_pitch = idx is not None and seq[idx] in PITCHES
    marked = on_pitch and going_at(seq, idx)
    info.append((on_pitch, marked, (not marked) and ">" in seq))
steals = steals.assign(on_pitch=[i[0] for i in info], marked=[i[1] for i in info], elsewhere=[i[2] for i in info])
b1 = steals.groupby("season").agg(events=("marked", "size"), steal_on_a_pitch=("on_pitch", "mean"), marked=("marked", "mean"), elsewhere=("elsewhere", "mean"))
b1 = b1.reindex(range(2016, 2026))
print(b1.to_string(formatters={"events": "{:,.0f}".format, "steal_on_a_pitch": "{:.3f}".format, "marked": "{:.3f}".format, "elsewhere": "{:.3f}".format}))
print("  marked = '>' on the steal pitch; elsewhere = no '>' on the steal pitch but a '>' somewhere else in the row's sequence")

print("\nB2. following event on '>' pitches with result ball or strike, by season (share of pitches)")
bs = going[going["result"].isin(["ball", "strike"])]
b2 = pd.crosstab(bs["season"], bs["following_event"], normalize="index")
b2 = b2[b2.sum().sort_values(ascending=False).index]
b2["n"] = bs.groupby("season").size()
print(b2.to_string(float_format="{:.3f}".format))

print("\nB3. swing rate by count, 2023-2025, runner on first and second open (thrown pitches; V, A and N excluded)")
recent = pitch[pitch["season"].between(2023, 2025) & ~pitch["code"].isin(NOT_THROWN | {"N"})].copy()
recent["swing"] = recent["code"].isin(SWINGS)
recent["count"] = recent["balls"].astype(str) + "-" + recent["strikes"].astype(str)
b3 = recent.groupby(["count", "going"])["swing"].agg(["mean", "size"]).unstack("going")
b3.columns = [f"{'going' if g else 'not going'} {m}" for m, g in b3.columns]
all_rate = recent.groupby("count")["swing"].agg(all_rate="mean", all_n="size")
b3 = b3.join(all_rate)[["going mean", "going size", "all_rate", "all_n"]]
b3.columns = ["'>' swing rate", "'>' n", "all pitches swing rate", "all pitches n"]
print(b3.to_string(formatters={"'>' swing rate": "{:.3f}".format, "all pitches swing rate": "{:.3f}".format, "'>' n": "{:,.0f}".format, "all pitches n": "{:,.0f}".format}))

print("\nB4. '>' pitches by base_code and season")
print(pd.crosstab(going["base_pre"], going["season"], margins=True).to_string())
g32 = going[(going["balls"] == 3) & (going["strikes"] == 2)]
print("'>' pitches at 3-2 by outs and season:")
print(pd.crosstab([g32["base_pre"], g32["outs_pre"]], g32["season"], margins=True).to_string())

print("\nB5. reconciliation with S1, 2023-2025")
mine = going[(going["base_pre"] == 1) & going["season"].between(2023, 2025) & going["result"].isin(["ball", "strike"])
             & going["following_event"].isin(["SB2", "CS2", "CS2_E"])]
s1 = pd.read_parquet(S1)
s1 = s1[s1["attempt_type"] == "steal"].copy()
parts = []
for seq in s1["pitches"].fillna(""):
    idx = max(i for i, ch in enumerate(seq) if ch in PITCHES)
    b, s = count_after(seq[:idx])
    parts.append((result_category(seq[idx], b, s), going_at(seq, idx)))
s1["steal_result"] = [x[0] for x in parts]
s1["marked"] = [x[1] for x in parts]
first_third = s1["base_pre"] == 5
strike3 = ~first_third & (s1["steal_result"] == "strike_three")
unmarked = ~first_third & ~strike3 & ~s1["marked"]
odd = ~first_third & ~strike3 & s1["marked"] & ~s1["steal_result"].isin(["ball", "strike"])
remaining = ~(first_third | strike3 | unmarked | odd)
print(f"  S1 main attempts (attempt_type 'steal'): {len(s1):,}")
print(f"    first and third (base_code 5): {int(first_third.sum()):,}")
print(f"    strike-three attempts in base_code 1 (K+SB2, K+CS2): {int(strike3.sum()):,}")
print(f"    base_code 1, steal pitch has no '>' marker: {int(unmarked.sum()):,}")
print(f"    base_code 1, marked, steal pitch result other than ball or strike: {int(odd.sum()):,}")
print(f"    remaining (base_code 1, marked, ball or strike): {int(remaining.sum()):,}")
print(f"  this table: '>' pitches in base_code 1, ball or strike, followed by SB2, CS2 or CS2 with an error: {len(mine):,}")
catcher_throw = mine["seq"].str.contains(r"\+[123]$", regex=True)
print(f"  difference (remaining minus this table): {int(remaining.sum()) - len(mine):,}")
print(f"    '>' pitch followed by a catcher pickoff throw on which the runner was caught: {int(catcher_throw.sum())} "
      f"(S1 took the throw as the moment of the play and filed these as pickoff caught stealing or not on a pitch;")
print(f"    here a '+' catcher throw belongs to the pitch before it). Remaining minus this table plus these: "
      f"{int(remaining.sum()) - len(mine) + int(catcher_throw.sum())}")
bf = going[(going["base_pre"] == 1) & going["season"].between(2023, 2025) & (going["result"] == "ball_four")]
print(f"  not in S1 by construction: '>' ball-four pitches in base_code 1 ({len(bf):,}; the walk forces the runner, so no steal is credited)")


def result_shares(df, label):
    grid = pd.MultiIndex.from_product([range(4), range(3), RESULTS], names=["balls", "strikes", "result"])
    long = df.groupby(["balls", "strikes", "result"]).size().reindex(grid, fill_value=0).rename("n").reset_index()
    total = long.groupby(["balls", "strikes"])["n"].transform("sum")
    long["share"] = np.where(total > 0, long["n"] / total.where(total > 0, 1), np.nan)
    return long


print("\n=== C. pitch-result distribution by count, base_code 1 (3-2 built from 0 and 1 out; 2-out 3-2 shown separately)")
base1 = pitch[pitch["base_pre"] == 1]
not_32_two_out = ~((base1["balls"] == 3) & (base1["strikes"] == 2) & (base1["outs_pre"] == 2))
groups = {
    "going 2023-2025": base1[base1["going"] & base1["season"].between(2023, 2025)],
    f"going {WIDE}": base1[base1["going"]],
    "not going 2023-2025": base1[~base1["going"] & base1["season"].between(2023, 2025)],
}
shares = {name: result_shares(df[not_32_two_out.loc[df.index]], name) for name, df in groups.items()}
for name, years in [("going 2023-2025", "2023_2025"), (f"going {WIDE}", "2016_2025")]:
    path = OUT / f"pitch_result_going_{years}.csv"
    shares[name].assign(share=shares[name]["share"].round(6))[["balls", "strikes", "result", "n", "share"]].to_csv(path, index=False)
    written[path] = len(shares[name])
side = None
for name, long in shares.items():
    w = long.set_index(["balls", "strikes", "result"])
    cell_n = long.groupby(["balls", "strikes"])["n"].sum()
    frame = pd.DataFrame({f"{name} share": w["share"]})
    frame[f"{name} n"] = w["n"]
    side = frame if side is None else side.join(frame, how="outer")
    thin = cell_n[cell_n < 100]
    print(f"  {name}: cells with n under 100: {', '.join(f'{b}-{s} ({n})' for (b, s), n in thin.items()) if len(thin) else 'none'}")
cells = pd.DataFrame({name: long.groupby(["balls", "strikes"])["n"].sum() for name, long in shares.items()})
print("\n  cell sizes (pitches per count):")
print(cells.astype(int).to_string())
print()
for (b, s), block in side.groupby(level=[0, 1]):
    blk = block.droplevel([0, 1])
    blk = blk[(blk[[c for c in blk.columns if c.endswith(" n")]].sum(axis=1) > 0)]
    print(f"  count {b}-{s}")
    print(blk.to_string(formatters={c: ("{:.3f}".format if c.endswith("share") else "{:,.0f}".format) for c in blk.columns}))
print("\n  3-2 with 2 outs (not in the saved tables)")
two = {name: df[(df["balls"] == 3) & (df["strikes"] == 2) & (df["outs_pre"] == 2)] for name, df in groups.items()}
t2 = pd.DataFrame({name: df["result"].value_counts(normalize=True).reindex(RESULTS, fill_value=0) for name, df in two.items()})
t2.loc["n"] = [len(df) for df in two.values()]
print(t2.to_string(float_format="{:.3f}".format))
thin2 = [f"{name} ({len(df)})" for name, df in two.items() if len(df) < 100]
print(f"  cells with n under 100: {', '.join(thin2) if thin2 else 'none'}")


print("\n=== D. runner-going advancement, base_code 1 (PAs ending on a '>' pitch put in play; innings 1-8; bunts excluded)")
std = pd.read_csv(STD_ADV)
std_split = pd.read_csv(STD_SPLIT).set_index(["base_code", "outs"])["if_share"]
end = pitch[pitch["going"] & pitch["pa_ended"] & (pitch["result"] == "in_play") & pitch["inning"].between(1, 8) & ~pitch["pa_excluded"]]
end = end.join(plays[["base_post", "outs_post", "runs", "firstf"]], on="row")
single = end["pa_category"] == "1B"
end["category"] = np.select([single & end["firstf"].between(1, 6), single & end["firstf"].between(7, 9)], ["1B_IF", "1B_OF"], end["pa_category"])
print(f"in-play PA endings on '>' pitches: {len(end):,}; singles with no first fielder (left out): {int((single & ~end['firstf'].between(1, 9)).sum())}; "
      f"categories outside the in-play list: {sorted(set(end['category']) - set(IN_PLAY_CATS))}")


def build_adv(src, base):
    src = src[src["base_pre"] == base]
    obs = src.groupby(["outs_pre", "category", "base_post", "outs_post", "runs"]).size().rename("k").reset_index()
    pool = src.assign(outs_added=src["outs_post"] - src["outs_pre"]).groupby(["category", "base_post", "outs_added", "runs"]).size().rename("k").reset_index()
    parts = []
    for outs in range(3):
        for cat in IN_PLAY_CATS:
            cell = obs[(obs["outs_pre"] == outs) & (obs["category"] == cat)]
            n = int(cell["k"].sum())
            if n >= MIN_N:
                parts.append(cell.assign(base_pre=base, n=n, n_cell=n, source="data"))
                continue
            rows = pool[pool["category"] == cat].copy()
            n_pool = int(rows["k"].sum())
            if n_pool >= MIN_N:
                rows["outs_post"] = outs + rows["outs_added"]
                ended = rows["outs_post"] >= 3
                rows.loc[ended, ["outs_post", "base_post", "runs"]] = [3, 0, 0]
                rows = rows.groupby(["base_post", "outs_post", "runs"])["k"].sum().reset_index()
                parts.append(rows.assign(base_pre=base, outs_pre=outs, category=cat, n=n_pool, n_cell=n, source="backoff"))
            else:
                fb = std[(std["base_pre"] == base) & (std["outs_pre"] == outs) & (std["category"] == cat)]
                parts.append(fb.drop(columns=["n_cell", "source"]).assign(n_cell=n, source="fallback"))
    adv = pd.concat(parts, ignore_index=True)
    adv["p"] = adv["k"] / adv["n"]
    fb = adv["source"] == "fallback"
    adv.loc[fb, "p"] = std.set_index(ADV_COLS[:6]).loc[list(map(tuple, adv.loc[fb, ADV_COLS[:6]].to_numpy())), "p"].to_numpy()
    adv["cat_order"] = adv["category"].map(IN_PLAY_CATS.index)
    return adv.sort_values(["outs_pre", "cat_order", "base_post", "outs_post", "runs"])[ADV_COLS].reset_index(drop=True)


versions = {"2023_2025": end[end["season"].between(2023, 2025)], "2016_2025": end}
tables = {}
for years, src in versions.items():
    adv = build_adv(src, 1)
    tables[years] = adv
    path = OUT / f"advancement_going_{years}.csv"
    adv.assign(p=adv["p"].round(6)).to_csv(path, index=False)
    written[path] = len(adv)
    cells = adv.groupby(["outs_pre", "category"]).agg(source=("source", "first"), n_cell=("n_cell", "first"))
    print(f"\n{WINDOW[years]}: {len(src[src['base_pre'] == 1]):,} PAs; cells by source: {cells['source'].value_counts().to_dict()}")
    print(cells.unstack("outs_pre").to_string())


def share(adv, base, outs, cat, test):
    cell = adv[(adv["base_pre"] == base) & (adv["outs_pre"] == outs) & (adv["category"] == cat)]
    if cell.empty:
        return np.nan, 0, ""
    return float(cell.loc[test(cell), "p"].sum()), int(cell["n_cell"].iloc[0]), cell["source"].iloc[0]


to_third = lambda c: ((c["base_post"] & 4) > 0) | (c["runs"] >= 1)  # noqa: E731
scores = lambda c: c["runs"] >= 1  # noqa: E731
two_outs = lambda c: (c["outs_post"] - c["outs_pre"]) >= 2  # noqa: E731
print("\ncomparison with advancement_v1 (all PAs, runner going or not), base_code 1")
print(f"{'measure':<36}{'outs':>5}  {'going 2023-2025':>24}  {'going ' + WIDE:>24}  {'advancement_v1':>22}")
for label, cat, test, outs_list in [
    ("1B outfield: runner to third or home", "1B_OF", to_third, range(3)),
    ("1B infield: runner to third or home", "1B_IF", to_third, range(3)),
    ("2B: runner scores", "2B", scores, range(3)),
    ("GB_OUT: double play", "GB_OUT", two_outs, range(2)),
    ("AIR_OUT: runner doubled off", "AIR_OUT", two_outs, range(2)),
]:
    for outs in outs_list:
        vals = []
        for adv in (tables["2023_2025"], tables["2016_2025"], std):
            p, n, src = share(adv, 1, outs, cat, test)
            vals.append(f"{p:.3f} (n={n:,}, {src})")
        print(f"{label:<36}{outs:>5}  {vals[0]:>24}  {vals[1]:>24}  {vals[2]:>22}")
for outs in range(3):
    vals = []
    for years in ["2023_2025", "2016_2025"]:
        src = versions[years]
        s = src[(src["base_pre"] == 1) & (src["outs_pre"] == outs) & src["category"].isin(["1B_IF", "1B_OF"])]
        vals.append(f"{to_third(s).mean():.3f} (n={len(s):,})")
    p_if, _, _ = share(std, 1, outs, "1B_IF", to_third)
    p_of, _, _ = share(std, 1, outs, "1B_OF", to_third)
    w = std_split.loc[(1, outs)]
    vals.append(f"{w * p_if + (1 - w) * p_of:.3f} (mixed)")
    print(f"{'1B all: runner to third or home':<36}{outs:>5}  {vals[0]:>24}  {vals[1]:>24}  {vals[2]:>22}")

print("\nbase_code 5 (first and third): cell counts only, PAs ending on a '>' pitch in play")
for years, src in versions.items():
    c5 = src[src["base_pre"] == 5].groupby(["category", "outs_pre"]).size().unstack("outs_pre").reindex(IN_PLAY_CATS).fillna(0).astype(int)
    print(f"  {WINDOW[years]}: {int(c5.values.sum()):,} PAs; cells with n >= {MIN_N}: {int((c5.values >= MIN_N).sum())} of {c5.size}")
    print(c5.to_string())

print("\n=== F. throw situations on '>' pitches")
print("throw_situation rows by season and base_code:")
print(pd.crosstab(going["base_pre"], going["season"], values=going["throw_situation"], aggfunc="sum", margins=True).fillna(0).astype(int).to_string())
r1 = going[(going["base_pre"] == 1) & going["season"].between(2023, 2025) & (going["throw_situation"] == 1)]
print(f"throw_safe by result, base_code 1, {RECENT}:")
print(r1.groupby("result")["throw_safe"].agg(mean="mean", n="size").reindex(["ball", "strike", "strike_three"]).to_string(formatters={"mean": "{:.3f}".format}))
cc = going[going["caught_on_catcher_throw"] == 1]
print(f"caught_on_catcher_throw: {len(cc)} in all seasons and base codes; base_code 1, {RECENT}: "
      f"{int(((cc['base_pre'] == 1) & cc['season'].between(2023, 2025)).sum())}")
sb_after = going[going["catcher_throw_after"] & (going["throw_situation"] == 1)]
print(f"note: throw situations where the catcher threw to first after the pitch and the runner was credited with SB2 or CS2_E: "
      f"{len(sb_after)} ({int(((sb_after['base_pre'] == 1) & sb_after['season'].between(2023, 2025)).sum())} in base_code 1, {RECENT}); "
      f"kept as throw situations by the definition")


def valid_results(balls, strikes):
    return ["ball" if balls < 3 else "ball_four", "strike" if strikes < 2 else "strike_three", "foul", "hit_by_pitch", "in_play"]


def v1_cells(df, counts, seasons):
    out = []
    for b, s in counts:
        cell = df[(df["balls"] == b) & (df["strikes"] == s)]
        vc = cell["result"].value_counts()
        res = valid_results(b, s)
        extra = set(vc.index) - set(res)
        if extra:
            raise SystemExit(f"unexpected results at {b}-{s}: {extra}")
        n = vc.reindex(res, fill_value=0)
        out.append(pd.DataFrame({"balls": b, "strikes": s, "result": res, "n": n.to_numpy(), "share": (n / n.sum()).to_numpy(), "seasons": seasons}))
    return pd.concat(out, ignore_index=True)


print("\n=== G. pitch-result table v1 (base_code 1, '>' pitches, 'other' dropped, excluded PAs removed)")
g1 = pitch[pitch["going"] & (pitch["base_pre"] == 1)]
g1 = g1[~((g1["balls"] == 3) & (g1["strikes"] == 2) & (g1["outs_pre"] == 2))]
removed = g1[g1["season"].between(2023, 2025) & g1["pa_group_excluded"] & (g1["result"] == "in_play")]
print(f"in_play '>' pitches removed because the PA is flagged excluded (bunt endings, intentional walks), {RECENT}: {len(removed)}")
print("  by count:", {f"{b}-{s}": n for (b, s), n in removed.groupby(["balls", "strikes"]).size().items()})
print(f"  all '>' pitches removed by the flag, {RECENT}: {int((g1['season'].between(2023, 2025) & g1['pa_group_excluded']).sum())}; "
      f"'other' pitches dropped: {int((g1['season'].between(2023, 2025) & (g1['result'] == 'other')).sum())}")
g1 = g1[~g1["pa_group_excluded"] & (g1["result"] != "other")]
all_counts = [(b, s) for b in range(4) for s in range(3)]
pr = v1_cells(g1[g1["season"].between(2023, 2025)], [c for c in all_counts if c != (3, 0)], RECENT)
pr = pd.concat([pr, v1_cells(g1, [(3, 0)], WIDE)], ignore_index=True).sort_values(["balls", "strikes"], kind="stable").reset_index(drop=True)
path = MODEL_OUT / "pitch_result_going_v1.csv"
pr.assign(share=pr["share"].round(6)).to_csv(path, index=False)
written[path] = len(pr)
wide_pr = pr.assign(count=pr["balls"].astype(str) + "-" + pr["strikes"].astype(str)).pivot_table(index="count", columns="result", values="share", sort=False)
wide_pr = wide_pr[["ball", "ball_four", "strike", "strike_three", "foul", "hit_by_pitch", "in_play"]]
cell_info = pr.groupby(["balls", "strikes"]).agg(n=("n", "sum"), seasons=("seasons", "first"), share_sum=("share", "sum"))
wide_pr["n"] = cell_info["n"].to_numpy()
wide_pr["seasons"] = cell_info["seasons"].to_numpy()
print(wide_pr.to_string(float_format="{:.3f}".format, na_rep="-"))
print(f"check: shares sum to 1 within every count: {bool(np.allclose(cell_info['share_sum'], 1))} (max deviation {abs(cell_info['share_sum'] - 1).max():.1e})")
print("3-2 uses 0 and 1 out only; 3-0 is the wider-window cell")

print("\n=== H. advancement table v1 (base_code 1, '>' pitch in play, innings 1-8, bunts and intentional walks excluded)")
adv_recent = end[end["season"].between(2023, 2025) & (end["base_pre"] == 1)]
adv_wide = end[end["base_pre"] == 1]
STD_LABEL = "2023-2025 (all PAs)"


def cell_rows(src, outs, cat):
    c = src[(src["outs_pre"] == outs) & (src["category"] == cat)]
    return c.groupby(["base_post", "outs_post", "runs"]).size().rename("k").reset_index()


def pooled_rows(src, outs, cat):
    c = src[src["category"] == cat]
    rows = c.assign(outs_added=c["outs_post"] - c["outs_pre"]).groupby(["base_post", "outs_added", "runs"]).size().rename("k").reset_index()
    rows["outs_post"] = outs + rows["outs_added"]
    ended = rows["outs_post"] >= 3
    rows.loc[ended, ["outs_post", "base_post", "runs"]] = [3, 0, 0]
    return rows.groupby(["base_post", "outs_post", "runs"])["k"].sum().reset_index()


v1_parts = []
for outs in range(3):
    for cat in IN_PLAY_CATS:
        own = cell_rows(adv_recent, outs, cat)
        n_cell = int(own["k"].sum())
        wide_cell = cell_rows(adv_wide, outs, cat)
        pool = pooled_rows(adv_wide, outs, cat)
        base = dict(base_pre=1, outs_pre=outs, category=cat, n_cell=n_cell)
        if n_cell >= MIN_N:
            part = own.assign(**base, n=n_cell, source="data", seasons=RECENT)
        elif wide_cell["k"].sum() >= MIN_N:
            part = wide_cell.assign(**base, n=int(wide_cell["k"].sum()), source="data", seasons=WIDE)
        elif pool["k"].sum() >= MIN_N:
            part = pool.assign(**base, n=int(pool["k"].sum()), source="backoff", seasons=WIDE)
        elif cat in ("HR", "OTHER"):
            # HR: everyone scores. OTHER is catcher's interference: batter to first, runner to second.
            post, runs = (0, 2) if cat == "HR" else (3, 0)
            part = pd.DataFrame({"base_post": [post], "outs_post": [outs], "runs": [runs], "k": [n_cell]}).assign(**base, n=n_cell, source="rule", seasons="rule")
        else:
            fb = std[(std["base_pre"] == 1) & (std["outs_pre"] == outs) & (std["category"] == cat)]
            part = fb[["base_post", "outs_post", "runs", "k", "n", "p"]].assign(**base, source="fallback", seasons=STD_LABEL)
        v1_parts.append(part)
v1 = pd.concat(v1_parts, ignore_index=True)
v1["p"] = np.where(v1["source"] == "rule", 1.0, np.where(v1["source"] == "fallback", v1["p"], v1["k"] / v1["n"]))
v1["cat_order"] = v1["category"].map(IN_PLAY_CATS.index)
v1 = v1.sort_values(["outs_pre", "cat_order", "base_post", "outs_post", "runs"])[ADV_COLS + ["seasons"]].reset_index(drop=True)
path = MODEL_OUT / "advancement_going_v1.csv"
v1.assign(p=v1["p"].round(6)).to_csv(path, index=False)
written[path] = len(v1)
grid = v1.groupby(["category", "outs_pre"], sort=False).agg(source=("source", "first"), seasons=("seasons", "first"), n=("n", "first"), n_cell=("n_cell", "first"))
grid["cell"] = grid["source"] + " | " + grid["seasons"] + " | n=" + grid["n"].astype(int).astype(str) + " (own " + grid["n_cell"].astype(str) + ")"
print(grid["cell"].unstack("outs_pre").reindex(IN_PLAY_CATS).to_string())
print(f"cells by source: {grid['source'].value_counts().to_dict()}")
print(f"rule cells: {[f'{c} {o} out' for (c, o), r in grid.iterrows() if r['source'] == 'rule']}")
print(f"fallback cells: {[f'{c} {o} out' for (c, o), r in grid.iterrows() if r['source'] == 'fallback']}")
sums = v1.groupby(["outs_pre", "category"])["p"].sum()
print(f"check: probabilities sum to 1 in every cell: {bool(np.allclose(sums, 1, atol=1e-5))}")

print("\ncomparison with advancement_v1 (all PAs, runner going or not), base_code 1, with the v1 composite")
print(f"{'measure':<36}{'outs':>5}  {'going 2023-2025':>24}  {'going ' + WIDE:>26}  {'advancement_v1':>22}  {'v1 composite':>34}")


def share_v1(outs, cat, test):
    cell = v1[(v1["outs_pre"] == outs) & (v1["category"] == cat)]
    return f"{float(cell.loc[test(cell), 'p'].sum()):.3f} ({cell['source'].iloc[0]}, {cell['seasons'].iloc[0]}, n={int(cell['n'].iloc[0]):,})"


for label, cat, test, outs_list in [
    ("1B outfield: runner to third or home", "1B_OF", to_third, range(3)),
    ("1B infield: runner to third or home", "1B_IF", to_third, range(3)),
    ("2B: runner scores", "2B", scores, range(3)),
    ("GB_OUT: double play", "GB_OUT", two_outs, range(2)),
    ("AIR_OUT: runner doubled off", "AIR_OUT", two_outs, range(2)),
]:
    for outs in outs_list:
        vals = []
        for adv in (tables["2023_2025"], tables["2016_2025"], std):
            p, n, src = share(adv, 1, outs, cat, test)
            vals.append(f"{p:.3f} (n={n:,}, {src})")
        print(f"{label:<36}{outs:>5}  {vals[0]:>24}  {vals[1]:>26}  {vals[2]:>22}  {share_v1(outs, cat, test):>34}")
for outs in range(3):
    vals = []
    for years in ["2023_2025", "2016_2025"]:
        src = versions[years]
        s1b = src[(src["base_pre"] == 1) & (src["outs_pre"] == outs) & src["category"].isin(["1B_IF", "1B_OF"])]
        vals.append(f"{to_third(s1b).mean():.3f} (n={len(s1b):,})")
    p_if, _, _ = share(std, 1, outs, "1B_IF", to_third)
    p_of, _, _ = share(std, 1, outs, "1B_OF", to_third)
    w = std_split.loc[(1, outs)]
    vals.append(f"{w * p_if + (1 - w) * p_of:.3f} (mixed)")
    own = adv_recent[(adv_recent["outs_pre"] == outs) & adv_recent["category"].isin(["1B_IF", "1B_OF"])]
    w_going = (own["category"] == "1B_IF").mean()
    c_if = v1[(v1["outs_pre"] == outs) & (v1["category"] == "1B_IF")]
    c_of = v1[(v1["outs_pre"] == outs) & (v1["category"] == "1B_OF")]
    comp = w_going * c_if.loc[to_third(c_if), "p"].sum() + (1 - w_going) * c_of.loc[to_third(c_of), "p"].sum()
    vals.append(f"{comp:.3f} (mixed by going {RECENT} IF share {w_going:.3f})")
    print(f"{'1B all: runner to third or home':<36}{outs:>5}  {vals[0]:>24}  {vals[1]:>26}  {vals[2]:>22}  {vals[3]:>34}")

print("\n=== I. in-play outcomes by count (all PAs, runner going or not, 2023-2025, innings 1-8, excluded flag off)")
INPLAY_CODES = ENDS - {"H"}
pa_all = plays[(plays["pa"] == 1) & plays["season"].between(2023, 2025) & plays["inning"].between(1, 8) & ~plays["pa_excluded"]].copy()
last_idx = [max((i for i, ch in enumerate(sq) if ch in PITCHES), default=-1) for sq in pa_all["seq"]]
pa_all["last_code"] = [sq[i] if i >= 0 else "" for sq, i in zip(pa_all["seq"], last_idx)]
counts_before = [count_after(sq[:i]) if i >= 0 else (0, 0) for sq, i in zip(pa_all["seq"], last_idx)]
pa_all["balls"] = [c[0] for c in counts_before]
pa_all["strikes"] = [c[1] for c in counts_before]
inplay = pa_all[pa_all["last_code"].isin(INPLAY_CODES)].copy()
sgl = inplay["pa_category"] == "1B"
inplay["category"] = np.select([sgl & inplay["firstf"].between(1, 6), sgl & inplay["firstf"].between(7, 9)], ["1B_IF", "1B_OF"], inplay["pa_category"])
other = inplay[inplay["category"] == "OTHER"]
print(f"PAs ending on an in-play pitch code (X, Y): {len(inplay):,}; categories: {inplay['category'].value_counts().to_dict()}")
print(f"OTHER on an in-play pitch code: {len(other)}; events: {other['event'].str.extract(r'^([A-Z]+/?[A-Z0-9]*)')[0].value_counts().to_dict()}")
print("  these are catcher's interference (C/E2), recorded with an X pitch code, not real balls in play: OTHER is dropped from this table")
inplay = inplay[inplay["category"] != "OTHER"]
unsplit = inplay["category"] == "1B"
print(f"  singles with no first fielder (cannot be split into 1B_IF and 1B_OF), dropped: {int(unsplit.sum())} "
      f"({int((unsplit & inplay['base_pre'].isin(GOING_BASES)).sum())} with a runner on first and second open)")
inplay = inplay[~unsplit]
OUTCOME_CATS = ["1B_IF", "1B_OF", "2B", "3B", "HR", "GB_OUT", "FC", "ROE", "AIR_OUT"]


def outcome_table(df):
    t = df.groupby(["balls", "strikes", "category"]).size().reindex(
        pd.MultiIndex.from_product([range(4), range(3), OUTCOME_CATS], names=["balls", "strikes", "category"]), fill_value=0).rename("n").reset_index()
    t["share"] = t["n"] / t.groupby(["balls", "strikes"])["n"].transform("sum")
    return t


io_first = outcome_table(inplay[inplay["base_pre"].isin(GOING_BASES)])
io_any = outcome_table(inplay)
for label, t in [("runner on first and second open (base_code 1 or 5)", io_first), ("any base-out state", io_any)]:
    w = t.assign(count=t["balls"].astype(str) + "-" + t["strikes"].astype(str)).pivot_table(index="count", columns="category", values="share", sort=False)[OUTCOME_CATS]
    w["n"] = t.groupby(["balls", "strikes"])["n"].sum().to_numpy()
    print(f"\n  {label}")
    print(w.to_string(float_format="{:.3f}".format))
cmp = io_first.merge(io_any, on=["balls", "strikes", "category"], suffixes=("_first", "_any"))
cell_n = cmp.groupby(["balls", "strikes"])["n_first"].transform("sum")
big = cmp[(cell_n >= 500) & ((cmp["share_first"] - cmp["share_any"]).abs() > 0.01)]
print(f"\n  cells (counts with n >= 500 in the runner-on-first version) where the two differ by more than 0.01: {len(big)}")
if len(big):
    print(big.assign(diff=big["share_first"] - big["share_any"])[["balls", "strikes", "category", "share_first", "share_any", "diff"]].to_string(index=False, float_format="{:.3f}".format))
print("  the runner-on-first version (1__ or 1_3) is saved, because the steal decision exists only in those states")
path = MODEL_OUT / "inplay_outcomes_by_count_v1.csv"
io_out = io_first.assign(share=io_first["share"].round(6), seasons=RECENT, states="1__ or 1_3")[["balls", "strikes", "category", "n", "share", "seasons", "states"]]
io_out.to_csv(path, index=False)
written[path] = len(io_out)
io_sums = io_first.groupby(["balls", "strikes"])["share"].sum()
print(f"  check: shares sum to 1 within every count in {path.relative_to(ROOT)}: {bool(np.allclose(io_sums, 1))} (max deviation {abs(io_sums - 1).max():.1e}); "
      f"{len(io_out)} rows, {int(io_first['n'].sum()):,} PAs")

print(f"\n  check: going in-play outcome mix (base_code 1, {RECENT}, D's population, OTHER dropped) beside the all-PA mix")
go_mix = adv_recent[adv_recent["category"] != "OTHER"]
ref_first = inplay[inplay["base_pre"].isin(GOING_BASES)]
rows_out = {"overall": (go_mix, inplay, ref_first)}
for (b, s), g in go_mix.groupby(["balls", "strikes"]):
    if len(g) >= 200:
        rows_out[f"{b}-{s}"] = (g, inplay[(inplay["balls"] == b) & (inplay["strikes"] == s)], ref_first[(ref_first["balls"] == b) & (ref_first["strikes"] == s)])
for label, (g, a, f) in rows_out.items():
    t = pd.DataFrame({"going": g["category"].value_counts(normalize=True), "all PAs, any state": a["category"].value_counts(normalize=True),
                      "all PAs, base_code 1 or 5": f["category"].value_counts(normalize=True)}).reindex(OUTCOME_CATS).fillna(0)
    t.loc["n"] = [len(g), len(a), len(f)]
    print(f"  {label}")
    print(t.T.to_string(float_format="{:.3f}".format))

print("\n=== J. going FC share (FC over GB_OUT plus FC, '>' in-play endings, base_code 1)")
std_fc = pd.read_csv(STD_FC)
fc_tabs = {}
for years, src in [("2023_2025", adv_recent), ("2016_2025", adv_wide)]:
    gb = src[src["category"].isin(["GB_OUT", "FC"])]
    t = gb.groupby("outs_pre")["category"].agg(n_fc=lambda c: int((c == "FC").sum()), n_gb="size").reindex(range(3), fill_value=0).reset_index()
    t = t.rename(columns={"outs_pre": "outs"})
    t.insert(0, "base_code", 1)
    t.insert(1, "base_label", LABEL[1])
    t["fc_share"] = t["n_fc"] / t["n_gb"]
    t["seasons"] = WINDOW[years]
    fc_tabs[years] = t
    print(f"  {WINDOW[years]}")
    print(t.to_string(index=False, float_format="{:.4f}".format))
print("  fc_share_v1 (all PAs, 2023-2025), base_code 1:")
print(std_fc[std_fc["base_code"] == 1].to_string(index=False, float_format="{:.4f}".format))
path = MODEL_OUT / "fc_share_going_v1.csv"
out_fc = fc_tabs["2016_2025"][list(std_fc.columns) + ["seasons"]]
out_fc.assign(fc_share=out_fc["fc_share"].round(6)).to_csv(path, index=False)
written[path] = len(out_fc)

print("\n=== E. files and runtime")
for path, n in written.items():
    print(f"  {path.relative_to(ROOT)}: {n:,} rows")
print(f"  runtime: {time.time() - T0:.1f} s")
