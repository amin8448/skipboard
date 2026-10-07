import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pa_matchup_model import state_ratio_table  # noqa: E402
from pitch_codes import NOT_THROWN, PITCHES  # noqa: E402
from steal_pitches import add_segments, pitch_rows  # noqa: E402

T0 = time.time()
ROOT = Path(__file__).resolve().parent.parent
SAVANT = ROOT / "data" / "raw" / "savant"
CHADWICK = ROOT / "data" / "raw" / "chadwick"
PLAYS = ROOT / "data" / "derived" / "plays_regular_2016_2025.parquet"
GOING = ROOT / "data" / "derived" / "steal" / "going_pitches_2016_2025.parquet"
OUT = ROOT / "data" / "derived" / "steal"
SEASONS = [2023, 2024, 2025]
GOING_BASES = (1, 5)
THROWN = PITCHES - NOT_THROWN  # pitches actually thrown: no automatic balls or strikes, no no-pitches

# Attributes used on the fit table, by role: (attribute table column, fit table column)
ATTRS = {
    "runner": [("runner_sprint_speed", "sprint_speed"), ("runner_lead_gained_att", "runner_lead_gained"), ("runner_lead_gained_opp", "runner_lead_gained_opp"),
               ("runner_agg_shrunk", "runner_aggressiveness")],
    "catcher": [("catcher_pop_2b", "pop_time"), ("catcher_exchange_2b", "exchange"), ("catcher_arm_2b", "arm_strength")],
    "pitcher": [("pitcher_primary_lead", "pitcher_primary_lead"), ("pitcher_lead_gained", "pitcher_lead_gained")],
}
CONTINUOUS = [fit for role in ATTRS.values() for _, fit in role]

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
written = {}


def check_season(df, year, columns, name):
    for col in columns:
        seen = sorted(pd.to_numeric(df[col]).unique().tolist())
        if seen != [year]:
            raise SystemExit(f"season check failed for {name}: {col} = {seen}, expected {year}")


# Chadwick register: MLBAM ID to Retrosheet ID, as in S1 (ambiguous links dropped).
people = pd.concat(
    [pd.read_csv(f, usecols=["key_retro", "key_mlbam"], dtype=str) for f in sorted(CHADWICK.glob("people-*.csv"))],
    ignore_index=True,
).dropna()
ambiguous = people["key_mlbam"].duplicated(keep=False) | people["key_retro"].duplicated(keep=False)
to_retro = people[~ambiguous].assign(key_mlbam=lambda d: d["key_mlbam"].astype("int64")).set_index("key_mlbam")["key_retro"]
print(f"Chadwick register: {len(people):,} people with both IDs; ambiguous links dropped: {int(ambiguous.sum())}")


def load(stem, id_col, cols, season_cols):
    frames = []
    for y in SEASONS:
        name = f"{stem}_{y}.csv"
        df = pd.read_csv(SAVANT / name)
        if season_cols:
            check_season(df, y, season_cols, name)
        df = df[[id_col] + list(cols)].rename(columns={id_col: "mlbam_id", **cols})
        for c in cols.values():
            df[c] = pd.to_numeric(df[c], errors="coerce")
        frames.append(df.assign(season=y))
    df = pd.concat(frames, ignore_index=True)
    df["mlbam_id"] = pd.to_numeric(df["mlbam_id"]).astype("int64")
    df["retro_id"] = df["mlbam_id"].map(to_retro)
    return df


boards = {
    "sprint speed (S1 file; no season column)": load("sprint_speed", "player_id", {
        "sprint_speed": "runner_sprint_speed", "competitive_runs": "runner_competitive_runs"}, []),
    "runner basestealing, steals of second": load("runner_basestealing_2b", "player_id", {
        "n_init": "runner_opportunities", "n_sb": "runner_sb", "n_cs": "runner_cs", "rate_sbx": "runner_attempt_rate",
        "net_act_plus": "runner_net_adv_plus", "net_act_minus": "runner_net_adv_minus",
        "r_primary_lead": "runner_primary_lead", "r_sec_minus_prim_lead": "runner_lead_gained_opp",
        "r_sec_minus_prim_lead_sbx": "runner_lead_gained_att"}, ["start_year", "end_year"]),
    "catcher pop time (page data)": load("catcher_poptime_page", "catcher_id", {
        "pop_2b_sba": "catcher_pop_2b", "pop_2b_sba_count": "catcher_pop_2b_n", "exchange_2b_sba": "catcher_exchange_2b",
        "exchange_2b_sba_count": "catcher_exchange_2b_n", "maxeff_arm_2b_sba": "catcher_arm_2b",
        "maxeff_arm_2b_sba_count": "catcher_arm_2b_n"}, ["year"]),
    "pitcher running game, steals of second (S1 file)": load("pitcher_running_game_2b", "player_id", {
        "n_init": "pitcher_opportunities", "n_pk": "pitcher_pickoffs", "n_sb": "pitcher_sb", "n_cs": "pitcher_cs",
        "r_primary_lead": "pitcher_primary_lead", "r_secondary_lead": "pitcher_secondary_lead",
        "r_sec_minus_prim_lead": "pitcher_lead_gained"}, ["start_year", "end_year"]),
}
print("\n=== a. player-season attributes")
print("season checks passed for every file with a season column; sprint speed has none (S1 file, content differs by season)")
print("MLBAM to Retrosheet mapping by leaderboard:")
for name, df in boards.items():
    print(f"  {name:<50} rows {len(df):>5,}  mapped {df['retro_id'].notna().mean():.2%}  unmapped {int(df['retro_id'].isna().sum())}")
runner_board = boards["runner basestealing, steals of second"]
runner_board["runner_attempts"] = runner_board["runner_sb"] + runner_board["runner_cs"]
runner_board["runner_net_bases_gained"] = runner_board["runner_net_adv_plus"] + runner_board["runner_net_adv_minus"]

# Runner aggressiveness: pitches thrown with the runner on first and second open, and those with '>'.
plays = pd.read_parquet(PLAYS)
plays = plays[plays["season"].isin(SEASONS)].reset_index(drop=True)
plays["seq"] = plays["pitches"].fillna("")
n_mismatch = add_segments(plays)
rows = plays[plays["base_pre"].isin(GOING_BASES) & plays["seq"].str.len().gt(0)]
pitches = pitch_rows(rows)
pitches = pitches[pitches["code"].isin(THROWN)].join(plays[["season", "br1_pre", "outs_pre"]], on="row")
# At 3-2 with 2 outs the runner goes automatically, so those pitches say nothing about aggressiveness.
auto = (pitches["balls"] == 3) & (pitches["strikes"] == 2) & (pitches["outs_pre"] == 2)
n_auto, n_auto_going = int(auto.sum()), int(pitches.loc[auto, "going"].sum())
pitches = pitches[~auto]
agg = pitches.groupby(["br1_pre", "season"])["going"].agg(runner_agg_opportunities="size", runner_agg_going="sum").reset_index()
agg = agg.rename(columns={"br1_pre": "retro_id"})
agg["runner_agg_rate"] = agg["runner_agg_going"] / agg["runner_agg_opportunities"]
league = agg.groupby("season")[["runner_agg_going", "runner_agg_opportunities"]].sum()
league_rate = league["runner_agg_going"] / league["runner_agg_opportunities"]
counts = pd.DataFrame({"going": agg["runner_agg_going"], "not_going": agg["runner_agg_opportunities"] - agg["runner_agg_going"]})
m_prior = float(state_ratio_table(counts)["m"]["going"])
n_opp = agg["runner_agg_opportunities"]
agg["runner_agg_shrunk"] = (agg["runner_agg_going"] + m_prior * agg["season"].map(league_rate)) / (n_opp + m_prior)

frames = [agg.set_index(["retro_id", "season"])]
for df in boards.values():
    keep = df.dropna(subset=["retro_id"]).drop(columns=["mlbam_id"]).set_index(["retro_id", "season"])
    frames.append(keep)
attrs = pd.concat(frames, axis=1).reset_index()
attrs["mlbam_id"] = attrs["retro_id"].map(pd.Series(to_retro.index, index=to_retro.values)).astype("Int64")
lead = ["retro_id", "mlbam_id", "season"]
attrs = attrs[lead + [c for c in attrs.columns if c not in lead]].sort_values(["season", "retro_id"]).reset_index(drop=True)
path = OUT / "player_season_attributes_2023_2025.parquet"
attrs.to_parquet(path, index=False)
written[path] = len(attrs)
print(f"\nattribute table: {len(attrs):,} player-seasons; player-seasons with each attribute:")
show = ["runner_sprint_speed", "runner_lead_gained_att", "runner_lead_gained_opp", "runner_opportunities", "runner_attempts", "runner_net_bases_gained",
        "runner_agg_opportunities", "runner_agg_shrunk", "catcher_pop_2b", "catcher_exchange_2b", "catcher_arm_2b",
        "pitcher_primary_lead", "pitcher_secondary_lead", "pitcher_lead_gained", "pitcher_opportunities"]
print(attrs.groupby("season")[show].count().T.to_string())
print(f"catcher arm strength to second rests on few throws: median {attrs['catcher_arm_2b_n'].median():.0f} max-effort throws per catcher-season (max {attrs['catcher_arm_2b_n'].max():.0f})")

print("\n=== runner aggressiveness")
print(f"plays rows whose sequence does not extend the previous row's: {n_mismatch}; thrown pitches with a runner on first and second open: {len(pitches):,} "
      f"(after removing {n_auto:,} pitches at 3-2 with 2 outs, {n_auto_going:,} of them with '>')")
print(f"prior strength m (method of moments across {len(agg):,} runner-seasons, as for the state ratios): {m_prior:,.1f}")
print("league rate by season (going pitches / pitches):")
print(pd.DataFrame({"going": league["runner_agg_going"], "pitches": league["runner_agg_opportunities"], "rate": league_rate}).to_string(formatters={"rate": "{:.4f}".format}))
top = agg[agg["runner_agg_opportunities"] >= 200].sort_values("runner_agg_shrunk", ascending=False).head(10)
print("ten most aggressive runner-seasons with at least 200 opportunities:")
print(top[["retro_id", "season", "runner_agg_opportunities", "runner_agg_going", "runner_agg_rate", "runner_agg_shrunk"]].to_string(index=False, float_format="{:.4f}".format))

print("\n=== b. fit table")
going = pd.read_parquet(GOING)
fit = going[(going["throw_situation"] == 1) & (going["base_code"] == 1) & going["season"].isin(SEASONS)].copy()
fit["pickoff_throws"] = np.where(fit["pitcher_pickoffs_before"] >= 2, "2+", fit["pitcher_pickoffs_before"].astype(str))
fit = fit.rename(columns={"runner_on_first": "runner_id", "catcher": "catcher_id", "pitcher": "pitcher_id", "result": "pitch_result"})
for role, pairs in ATTRS.items():
    sub = attrs[["retro_id", "season"] + [a for a, _ in pairs]].rename(columns={"retro_id": f"{role}_id", **dict(pairs)})
    fit = fit.merge(sub, on=[f"{role}_id", "season"], how="left")
season_means = {}
for role, pairs in ATTRS.items():
    for a, f in pairs:
        means = attrs.groupby("season")[a].mean()  # over player-seasons, not throws
        season_means[f] = means
        fit[f"{f}_missing"] = fit[f].isna().astype(int)
        fit[f] = fit[f].fillna(fit["season"].map(means))
cols = ["throw_safe", "gid", "date", "season", "inning", "half", "pitch_result", "balls", "strikes", "outs", "pickoff_throws", "pitchout",
        "pitcher_hand", "bat_side", "runner_id", "sprint_speed", "runner_lead_gained", "runner_lead_gained_opp", "runner_aggressiveness",
        "catcher_id", "pop_time", "exchange", "arm_strength", "pitcher_id", "pitcher_primary_lead", "pitcher_lead_gained"]
cols += [f"{f}_missing" for f in CONTINUOUS]
fit = fit[cols].reset_index(drop=True)
path = OUT / "steal_fit_table_2023_2025.parquet"
fit.to_parquet(path, index=False)
written[path] = len(fit)
print("fill values (season league mean over player-seasons):")
print(pd.DataFrame(season_means).T.to_string(float_format="{:.3f}".format))

print("\n=== report")
print("fit table rows and mean throw_safe by season:")
print(fit.groupby("season")["throw_safe"].agg(rows="size", throw_safe="mean").to_string(formatters={"throw_safe": "{:.3f}".format}))
print(f"all: {len(fit):,} rows, throw_safe {fit['throw_safe'].mean():.3f}")
print("\nmean throw_safe by pitch result:")
print(fit.groupby("pitch_result")["throw_safe"].agg(["mean", "size"]).to_string(float_format="{:.3f}".format))
print("\nmean throw_safe by pickoff throws earlier in the PA:")
print(fit.groupby("pickoff_throws")["throw_safe"].agg(["mean", "size"]).to_string(float_format="{:.3f}".format))
print("\npickoff_throws by season (rows):")
print(pd.crosstab(fit["pickoff_throws"], fit["season"], margins=True).to_string())

print("\nmissingness among throw situations (share of rows), by season:")
miss = fit.groupby("season")[[f"{f}_missing" for f in CONTINUOUS]].mean().T
miss.index = [f"{role}: {f}" for role, pairs in ATTRS.items() for _, f in pairs]
miss["all seasons"] = fit[[f"{f}_missing" for f in CONTINUOUS]].mean().to_numpy()
print(miss.to_string(float_format="{:.3f}".format))

print("\ncontinuous covariates on the fit table (missing values filled with the season mean):")
stats = fit[CONTINUOUS].agg(["mean", "std", "min", "max"]).T
stats["observed"] = [int((fit[f"{f}_missing"] == 0).sum()) for f in CONTINUOUS]
print(stats.to_string(float_format="{:.3f}".format))
observed = fit[CONTINUOUS].where(fit[[f"{f}_missing" for f in CONTINUOUS]].to_numpy() == 0)
print("\ncorrelation matrix (observed values only, pairwise):")
print(observed.corr().to_string(float_format="{:.2f}".format))
print(f"runner_lead_gained_opp (opportunities version): correlation with sprint speed {observed['runner_lead_gained_opp'].corr(observed['sprint_speed']):.3f}; "
      f"with the attempts version {observed['runner_lead_gained_opp'].corr(observed['runner_lead_gained']):.3f}")
print("\npitcher primary lead and lead gained by pitcher hand (observed values):")
print(observed.assign(pitcher_hand=fit["pitcher_hand"]).groupby("pitcher_hand")[["pitcher_primary_lead", "pitcher_lead_gained"]].agg(["mean", "count"]).to_string(float_format="{:.3f}".format))

print("\nmean throw_safe by sprint speed bin within catcher arm strength terciles (observed values; bins right-closed: '26 and under' is <= 26.0, '29 and up' is > 29.0)")
both = fit[(fit["sprint_speed_missing"] == 0) & (fit["arm_strength_missing"] == 0)].copy()
both["sprint_bin"] = pd.cut(both["sprint_speed"], [-np.inf, 26, 27, 28, 29, np.inf], labels=["26 and under", "26 to 27", "27 to 28", "28 to 29", "29 and up"])
both["arm_tercile"] = pd.qcut(both["arm_strength"], 3, labels=["low arm", "middle arm", "high arm"])
edges = both.groupby("arm_tercile", observed=True)["arm_strength"].agg(["min", "max"])
print("  arm tercile ranges (mph): " + "; ".join(f"{t} {r['min']:.1f}-{r['max']:.1f}" for t, r in edges.iterrows()))
tab = both.groupby(["sprint_bin", "arm_tercile"], observed=False)["throw_safe"].agg(["mean", "size"]).unstack("arm_tercile")
tab.columns = [f"{t} {s}" for s, t in tab.columns]
print(tab[[f"{t} {s}" for t in ["low arm", "middle arm", "high arm"] for s in ["mean", "size"]]].to_string(float_format="{:.3f}".format))

print("\njoin check:")
for role in ["runner", "catcher", "pitcher"]:
    ids = fit[f"{role}_id"]
    in_attrs = ids.isin(attrs["retro_id"])
    print(f"  {role}: rows with no id {int(ids.isna().sum())}; rows whose id has no row in the attribute table {int((ids.notna() & ~in_attrs).sum())}; "
          f"distinct ids {ids.nunique():,}")

print("\nfiles:")
for path, n in written.items():
    print(f"  {path.relative_to(ROOT)}: {n:,} rows")
print(f"runtime: {time.time() - T0:.1f} s")
