import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pitch_codes import BALLS, FOUL_BUNT, FOULS, PITCHES, STRIKES, count_after  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PLAYS = ROOT / "data" / "derived" / "plays_regular_2016_2025.parquet"
CHADWICK = ROOT / "data" / "raw" / "chadwick"
SAVANT = ROOT / "data" / "raw" / "savant"
OUT = ROOT / "data" / "derived" / "steal_attempts_2023_2025.parquet"
SEASONS = [2023, 2024, 2025]
PLAY_CHARS = PITCHES | set("123N")  # a pitch, a pickoff throw, or a no-pitch: the moment a play can happen on

pd.set_option("display.width", 250)

plays = pd.read_parquet(PLAYS)
plays = plays[plays["season"].isin(SEASONS)].reset_index(drop=True)

# Crosswalk: Retrosheet ID to MLBAM ID from the Chadwick register.
people = pd.concat(
    [pd.read_csv(f, usecols=["key_retro", "key_mlbam"], dtype=str) for f in sorted(CHADWICK.glob("people-*.csv"))],
    ignore_index=True,
)
people = people.dropna(subset=["key_retro", "key_mlbam"])
dup = people["key_retro"].duplicated(keep=False)
print(f"Chadwick register: {len(people):,} people with both Retrosheet and MLBAM IDs; Retrosheet IDs with more than one MLBAM ID: {int(dup.sum())}")
to_mlbam = people.drop_duplicates("key_retro", keep=False).set_index("key_retro")["key_mlbam"].astype("int64")

print("\nRetrosheet to MLBAM match rate, 2023-2025 regular season")
roles = {
    "runners on first": plays["br1_pre"].dropna(),
    "catchers": plays["f2"].dropna(),
    "pitchers": plays["pitcher"].dropna(),
}
for role, ids in roles.items():
    distinct = ids.unique()
    matched = pd.Index(distinct).isin(to_mlbam.index)
    rows = ids.isin(to_mlbam.index)
    print(f"  {role:<17} distinct IDs {matched.sum():,} of {len(distinct):,} ({matched.mean():.2%}); plays {rows.mean():.4%}")
    if (~matched).any():
        print(f"    unmatched: {sorted(distinct[~matched])[:10]}")

# Candidate rows: a runner on first, second base open, and either a steal of second or defensive indifference.
open_second = plays["br1_pre"].notna() & plays["br2_pre"].isna()
is_steal = (plays["sb2"] == 1) | (plays["cs2"] == 1)
is_di = plays["event"].str.startswith("DI")
att = plays[open_second & (is_steal | is_di)].copy()


def steal_moment(seq):
    # The play happens on the last pitch, pickoff throw or no-pitch in the sequence. If that is a pitch,
    # the count at the steal pitch is the count after everything before it; otherwise the play was on a
    # throw or no-pitch and the count is the count after all pitches so far.
    idx = max((i for i, ch in enumerate(seq) if ch in PLAY_CHARS), default=None)
    if idx is None:
        return 0, 0, False, False, 0
    on_pitch = seq[idx] in PITCHES
    before = seq[:idx]
    balls, strikes = count_after(before)
    prev = max((i for i, ch in enumerate(before) if ch in PITCHES), default=-1)
    marker = on_pitch and ">" in seq[prev + 1:idx]
    pickoffs = sum(1 for i, ch in enumerate(before) if ch == "1" and (i == 0 or before[i - 1] != "+"))  # pitcher throws only
    return balls, strikes, on_pitch, marker, pickoffs


moments = [steal_moment(s) for s in att["pitches"].fillna("")]
att["balls_at_steal"] = [m[0] for m in moments]
att["strikes_at_steal"] = [m[1] for m in moments]
att["on_pitch"] = [m[2] for m in moments]
att["runner_going_marker"] = [m[3] for m in moments]
att["pickoffs_before"] = [m[4] for m in moments]

att["attempt_type"] = np.select(
    [
        att["event"].str.startswith("POCS"),
        att["event"].str.startswith("DI"),
        (att["sbh"] == 1) | (att["csh"] == 1),
        ~att["on_pitch"],
    ],
    ["pickoff_caught_stealing", "defensive_indifference", "double_steal", "not_on_pitch"],
    "steal",
)
att["cs_error"] = (att["cs2"] == 1) & att["event"].str.contains(r"CS2\([^)]*E", regex=True)  # caught stealing negated by an error
att["success"] = (att["sb2"] == 1) | att["cs_error"] | (att["attempt_type"] == "defensive_indifference")
att["pa_ending"] = att["pa"] == 1
att["score_diff"] = att["bat_score_pre"] - att["fld_score_pre"]

print(f"\nsteal-of-second candidates (runner on first, second open): {len(att):,}")
print(pd.crosstab(att["attempt_type"], att["season"], margins=True).to_string())
print(f"caught stealing negated by an error (runner safe, counted as success): {int(att['cs_error'].sum())}")

main = att["attempt_type"] == "steal"
check = att[main]
retro = (check["balls"] == check["balls_at_steal"]) & (check["strikes"] == check["strikes_at_steal"])
print("\ncount at the steal pitch")
print("  method: the steal happens on the last pitch in the row's sequence (the sequence runs from the start of the")
print("  plate appearance, so earlier segments before a '.' are included). The count is rebuilt from every pitch before")
print("  it with the same rules as the plate-appearance tables. A strikeout plus a steal is the same case: the steal")
print("  pitch is the strike-three pitch, so its count is the count before it.")
print(f"  main attempts: {len(check):,}; matches Retrosheet's balls and strikes columns (count before the final pitch of the play): {int(retro.sum()):,} ({retro.mean():.3%})")
print(f"  steal pitch preceded by the runner-going marker '>': {check['runner_going_marker'].mean():.2%}; on a PA-ending pitch (e.g. K+SB2): {int(check['pa_ending'].sum()):,}")
if (~retro).any():
    print(check.loc[~retro, ["event", "pitches", "balls", "strikes", "balls_at_steal", "strikes_at_steal"]].head(5).to_string(index=False))


def walk(seq):
    steps, b, s = [], 0, 0
    idx = max(i for i, ch in enumerate(seq) if ch in PLAY_CHARS)
    for i, ch in enumerate(seq[:idx]):
        if ch in PITCHES:
            if ch in BALLS:
                b += 1
            elif ch in STRIKES or ch == FOUL_BUNT or (ch in FOULS and s < 2):
                s += 1
            steps.append(f"{ch}->{b}-{s}")
        else:
            steps.append(f"{ch}(marker)")
    return " ".join(steps) + f" | steal pitch '{seq[idx]}' at {b}-{s}"


print("\nworked examples")
examples = [
    ("plain steal with the runner-going marker", check[(~check["pa_ending"]) & (~check["pitches"].str.contains(r"[.1]", regex=True)) & check["runner_going_marker"] & (check["pitches"].str.len() >= 5)]),
    ("strikeout plus steal on the same pitch", check[check["pa_ending"] & check["event"].str.startswith("K+SB2")]),
    ("steal after pickoff throws and an earlier play in the same PA", check[check["pitches"].str.contains(".", regex=False) & (check["pickoffs_before"] >= 1)]),
]
for label, pool in examples:
    r = pool.iloc[0]
    print(f"  {label}: {r['gid']} inning {r['inning']}, event {r['event']}, pitches '{r['pitches']}'")
    print(f"    {walk(r['pitches'])}; pickoff throws before: {r['pickoffs_before']}; Retrosheet columns {r['balls']}-{r['strikes']}")

att["runner_mlbam"] = att["br1_pre"].map(to_mlbam).astype("Int64")
att["catcher_mlbam"] = att["f2"].map(to_mlbam).astype("Int64")
att["pitcher_mlbam"] = att["pitcher"].map(to_mlbam).astype("Int64")

sprint = pd.concat([pd.read_csv(SAVANT / f"sprint_speed_{y}.csv").assign(season=y) for y in SEASONS])
sprint = sprint[["season", "player_id", "sprint_speed", "competitive_runs"]].rename(columns={"player_id": "runner_mlbam"})
pop = pd.concat([pd.read_csv(SAVANT / f"catcher_pop_time_{y}.csv").assign(season=y) for y in SEASONS])
pop = pop[["season", "entity_id", "pop_2b_sba", "pop_2b_sba_count"]].rename(columns={"entity_id": "catcher_mlbam"})
for df, key in [(sprint, "runner_mlbam"), (pop, "catcher_mlbam")]:
    if df.duplicated(["season", key]).any():
        raise SystemExit(f"duplicate {key} rows in a Savant season file")
    df[key] = df[key].astype("Int64")
att = att.merge(sprint, on=["season", "runner_mlbam"], how="left").merge(pop, on=["season", "catcher_mlbam"], how="left")

m = att[att["attempt_type"] == "steal"]
print("\njoins (main attempts)")
print(f"  runner MLBAM ID: {m['runner_mlbam'].notna().mean():.2%}; catcher MLBAM ID: {m['catcher_mlbam'].notna().mean():.2%}; pitcher MLBAM ID: {m['pitcher_mlbam'].notna().mean():.2%}")
print(f"  sprint speed: {m['sprint_speed'].notna().mean():.2%} ({int(m['sprint_speed'].notna().sum()):,} of {len(m):,})")
print(f"  pop time to second: {m['pop_2b_sba'].notna().mean():.2%} ({int(m['pop_2b_sba'].notna().sum()):,})")
print(f"  both: {(m['sprint_speed'].notna() & m['pop_2b_sba'].notna()).mean():.2%} ({int((m['sprint_speed'].notna() & m['pop_2b_sba'].notna()).sum()):,})")

cols = [
    "date", "season", "gid", "inning", "top_bot", "outs_pre", "base_pre", "score_diff",
    "br1_pre", "runner_mlbam", "f2", "catcher_mlbam", "pitcher", "pitcher_mlbam", "pithand", "batter",
    "balls_at_steal", "strikes_at_steal", "on_pitch", "runner_going_marker", "pickoffs_before",
    "attempt_type", "success", "cs_error", "pa_ending", "event", "pitches", "balls", "strikes",
    "sprint_speed", "competitive_runs", "pop_2b_sba", "pop_2b_sba_count",
]
out = att[cols].rename(columns={"outs_pre": "outs", "br1_pre": "runner", "f2": "catcher", "pithand": "pitcher_hand",
                                "balls": "retro_balls", "strikes": "retro_strikes"})
OUT.parent.mkdir(parents=True, exist_ok=True)
out.to_parquet(OUT, index=False)
print(f"\nsaved {OUT.relative_to(ROOT)} ({len(out):,} rows, {out.shape[1]} columns; main attempts have attempt_type 'steal')")


def rate_table(df, by, label):
    t = df.groupby(by, observed=False)["success"].agg(attempts="size", success_rate="mean")
    print(f"\nsuccess rate by {label}")
    print(t.to_string(formatters={"attempts": "{:,}".format, "success_rate": "{:.3f}".format}))


m = out[out["attempt_type"] == "steal"].copy()
rate_table(m, "season", "season (main attempts)")
print(f"  all seasons: {len(m):,} attempts, success rate {m['success'].mean():.3f}")
m["sprint_bin"] = pd.cut(m["sprint_speed"], [0, 26, 27, 28, 29, 99], right=False, labels=["below 26", "26-27", "27-28", "28-29", "29 and up"])
m["pop_bin"] = pd.cut(m["pop_2b_sba"], [0, 1.90, 1.95, 2.00, 9], right=False, labels=["below 1.90", "1.90-1.95", "1.95-2.00", "2.00 and up"])
rate_table(m.assign(sprint_bin=m["sprint_bin"].cat.add_categories("no sprint speed").fillna("no sprint speed")), "sprint_bin", "runner sprint speed (ft/s)")
rate_table(m.assign(pop_bin=m["pop_bin"].cat.add_categories("no pop time").fillna("no pop time")), "pop_bin", "catcher pop time to second (s)")
m["count"] = m["balls_at_steal"].astype(str) + "-" + m["strikes_at_steal"].astype(str)
rate_table(m, "count", "count at the steal pitch")
m["pickoffs"] = np.where(m["pickoffs_before"] >= 2, "2 or more", m["pickoffs_before"].astype(str))
rate_table(m, "pickoffs", "pickoff throws to first earlier in the plate appearance")
