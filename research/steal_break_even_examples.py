import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_profiles import plate_appearances, profiles  # noqa: E402
from pa_matchup_model import MODEL  # noqa: E402
from skipboard.plate_appearance import PlayerProfile  # noqa: E402
from skipboard.state import GameState  # noqa: E402
from skipboard.steal_decision import steal_decision  # noqa: E402
from skipboard.steal_model import CatcherProfile, PitcherProfile, RunnerProfile, load_steal_model  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ATTRS = ROOT / "data" / "derived" / "steal" / "player_season_attributes_2023_2025.parquet"
FIT = ROOT / "data" / "derived" / "steal" / "steal_fit_table_2023_2025.parquet"
PLAYS = ROOT / "data" / "derived" / "plays_regular_2016_2025.parquet"
SEASON = 2025
# Linear weights (runs relative to an out) to rank hitters; only used to pick examples.
WEIGHTS = {"BB": 0.69, "HBP": 0.72, "1B": 0.89, "2B": 1.27, "3B": 1.62, "HR": 2.10}
MIN_WEIGHTED_PA = 2000  # established players only
SITUATIONS = [("0-0, 0 outs", 0, 0, 0), ("1-1, 1 out", 1, 1, 1)]

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)

pa = plate_appearances()
cand = pa[(pa["season"] == SEASON) & (pa["base_pre"] == 1) & (pa["outs_pre"] == 0)].reset_index(drop=True)
bat = profiles([(b, "batter", d) for b, d in zip(cand["batter"], cand["date"])])
pit = profiles([(p, "pitcher", d) for p, d in zip(cand["pitcher"], cand["date"])])
cand["bat_value"] = sum(bat[c] * w for c, w in WEIGHTS.items())
cand["pit_k"] = pit["K"]
ok = (bat["weighted_pa"] >= MIN_WEIGHTED_PA) & (pit["weighted_pa"] >= MIN_WEIGHTED_PA)
print(f"{SEASON} plate appearances with a runner on first only and 0 outs: {len(cand):,}; with established batter and pitcher: {int(ok.sum()):,}")

bv = cand["bat_value"].where(ok)
pk = cand["pit_k"].where(ok)
mid_p = (pk - pk.median()).abs()
mid_b = (bv - bv.median()).abs()
picks = {
    "strong hitter vs average pitcher": ((mid_p.rank() <= 0.2 * ok.sum()) & ok, bv),
    "weak hitter vs average pitcher": ((mid_p.rank() <= 0.2 * ok.sum()) & ok, -bv),
    "average hitter vs high-K pitcher": ((mid_b.rank() <= 0.2 * ok.sum()) & ok, pk),
    "strong hitter vs high-K pitcher": ((pk >= pk.quantile(0.9)) & ok, bv),
    "weak hitter vs high-K pitcher": ((pk >= pk.quantile(0.9)) & ok, -bv),
}
chosen = {name: order.where(mask).idxmax() for name, (mask, order) in picks.items()}

# Runner on first and catcher for each chosen plate appearance, from the plays table.
plays = pd.read_parquet(PLAYS, columns=["gid", "season", "inning", "top_bot", "batter", "pitcher", "pa", "base_pre", "outs_pre", "br1_pre", "f2"])
plays = plays[(plays["season"] == SEASON) & (plays["pa"] == 1) & (plays["base_pre"] == 1) & (plays["outs_pre"] == 0)]
attrs = pd.read_parquet(ATTRS)
attrs = attrs[attrs["season"] == SEASON].set_index("retro_id")
model = load_steal_model()
print(f"steal success model: data/derived/steal (fitted seasons {model.fitted_seasons}); fill values for {SEASON}:")
for hand in ("L", "R"):
    print(f"  pitcher hand {hand}: " + ", ".join(f"{k} {v:.3f}" for k, v in model.fill_values(SEASON, hand).items()))

# League-average attempt: empirical safe rate by pitch result and count on the fit table (2023-2025, runner on first only).
fit = pd.read_parquet(FIT, columns=["throw_safe", "pitch_result", "balls", "strikes", "pitchout"])
n_pitchout = int(fit["pitchout"].sum())
fit = fit[~fit["pitchout"]]
rates = fit.groupby(["pitch_result", "balls", "strikes"])["throw_safe"].agg(["mean", "size"])
used = sorted({(res, b, s) for _, b, s, _ in SITUATIONS for res in ("ball", "strike", "strike_three") if res != "strike_three" or s == 2})
used = [k for k in used if k in rates.index]
print(f"\nempirical throw success used for the league-average attempt column (fit table, {len(fit):,} attempts; {n_pitchout} pitchouts excluded; all pickoff counts):")
print(rates.loc[used].rename(columns={"mean": "safe rate", "size": "n"}).to_string(formatters={"safe rate": lambda x: f"{x:.4f}"}))


def empirical_p_safe(pitch_result, balls, strikes, *rest):
    if (pitch_result, balls, strikes) not in rates.index:
        raise KeyError(f"no attempts on the fit table for {pitch_result} at {balls}-{strikes}")
    return float(rates.at[(pitch_result, balls, strikes), "mean"])


attempt_model = SimpleNamespace(p_safe=empirical_p_safe)
print("state: bottom of the 7th, tied 3-3, runner on first only, pickoff throws 0; runs values do not depend on the inning or score\n")


def attr(retro_id, column):
    if retro_id not in attrs.index or pd.isna(attrs.at[retro_id, column]):
        return None
    return float(attrs.at[retro_id, column])


def fmt(x, digits=4):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{digits}f}"


for name, i in chosen.items():
    r = cand.loc[i]
    play = plays[(plays["gid"] == r["gid"]) & (plays["inning"] == r["inning"]) & (plays["top_bot"] == r["top_bot"])
                 & (plays["batter"] == r["batter"]) & (plays["pitcher"] == r["pitcher"])].iloc[0]
    runner_id, catcher_id = play["br1_pre"], play["f2"]
    batter_pa = PlayerProfile(r["batter"], "batter", tuple(bat.loc[i, MODEL].astype(float)), bat.at[i, "hand"], float(bat.at[i, "weighted_pa"]))
    pitcher_pa = PlayerProfile(r["pitcher"], "pitcher", tuple(pit.loc[i, MODEL].astype(float)), pit.at[i, "hand"], float(pit.at[i, "weighted_pa"]))
    runner = RunnerProfile(sprint_speed=attr(runner_id, "runner_sprint_speed"), aggressiveness=attr(runner_id, "runner_agg_shrunk"))
    catcher = CatcherProfile(pop_time=attr(catcher_id, "catcher_pop_2b"))
    hold = PitcherProfile(hand=r["pithand"], primary_lead=attr(r["pitcher"], "pitcher_primary_lead"), lead_gained=attr(r["pitcher"], "pitcher_lead_gained"))
    filled = {
        f"runner {runner_id}": [k for k, v in [("sprint_speed", runner.sprint_speed), ("aggressiveness", runner.aggressiveness)] if v is None],
        f"catcher {catcher_id}": [k for k, v in [("pop_time", catcher.pop_time)] if v is None],
        f"pitcher {r['pitcher']}": [k for k, v in [("primary_lead", hold.primary_lead), ("lead_gained", hold.lead_gained)] if v is None],
    }
    print("=" * 140)
    print(f"{name}: batter {r['batter']} ({r['bathand']}) vs pitcher {r['pitcher']} ({r['pithand']}), runner {runner_id}, catcher {catcher_id}, game {r['gid']} {int(r['date'])}")
    print(f"  steal profiles: runner sprint {fmt(runner.sprint_speed, 1)} ft/s, aggressiveness {fmt(runner.aggressiveness, 3)}; catcher pop {fmt(catcher.pop_time, 2)} s; "
          f"pitcher primary lead {fmt(hold.primary_lead, 2)} ft, lead gained {fmt(hold.lead_gained, 2)} ft")
    print("  filled with the season mean (pitcher attributes: the mean for the pitcher's hand): " + "; ".join(f"{who}: {', '.join(v) if v else 'none'}" for who, v in filled.items()))
    for label, balls, strikes, outs in SITUATIONS:
        state = GameState(season=SEASON, game_type="regular", inning=7, half=1, bat_score=3, fld_score=3, outs=outs, balls=balls, strikes=strikes,
                          runner1=runner_id, batter=r["batter"], batter_hand=r["bathand"], pitcher=r["pitcher"], pitcher_hand=r["pithand"],
                          catcher=catcher_id, pickoff_throws=0)
        mine = steal_decision(state, batter_pa, pitcher_pa, runner=runner, catcher=catcher, pitcher_hold=hold, model=model)
        base = steal_decision(state, None, None, runner=RunnerProfile(), catcher=CatcherProfile(), pitcher_hold=PitcherProfile(hand=r["pithand"]), model=model)
        att = steal_decision(state, None, None, runner=RunnerProfile(), catcher=CatcherProfile(), pitcher_hold=PitcherProfile(hand=r["pithand"]), model=attempt_model)
        summary = pd.DataFrame({
            "matchup wp": [mine.hold_wp, mine.run_wp, mine.run_wp - mine.hold_wp, mine.effective_p_safe, mine.break_even_wp],
            "baseline wp": [base.hold_wp, base.run_wp, base.run_wp - base.hold_wp, base.effective_p_safe, base.break_even_wp],
            "attempt wp": [att.hold_wp, att.run_wp, att.run_wp - att.hold_wp, att.effective_p_safe, att.break_even_wp],
            "matchup runs": [mine.hold_runs, mine.run_runs, mine.run_runs - mine.hold_runs, mine.effective_p_safe, mine.break_even_runs],
            "baseline runs": [base.hold_runs, base.run_runs, base.run_runs - base.hold_runs, base.effective_p_safe, base.break_even_runs],
            "attempt runs": [att.hold_runs, att.run_runs, att.run_runs - att.hold_runs, att.effective_p_safe, att.break_even_runs],
        }, index=["not this pitch (hold)", "run on this pitch", "run minus hold", "effective throw success", "break-even throw success"])
        print(f"\n  {label} (baseline: league-average batter and pitcher, profiles at the {SEASON} fill values for a {r['pithand']} pitcher; "
              "attempt: league-average batter and pitcher, empirical throw success by pitch result and count)")
        print(summary.to_string(formatters={c: (lambda x: fmt(x)) for c in summary.columns}))
        rows = []
        for bm, bb in zip(mine.branches, base.branches):
            rows.append({"pitch result": bm.result, "share": bm.share, "p_safe matchup": bm.p_safe, "p_safe baseline": bb.p_safe,
                         "wp matchup": bm.value_wp, "wp baseline": bb.value_wp, "runs matchup": bm.value_runs, "runs baseline": bb.value_runs})
        print("  branches:")
        print(pd.DataFrame(rows).set_index("pitch result").to_string(formatters={c: (lambda x: fmt(x)) for c in rows[0] if c != "pitch result"}))
    print()
