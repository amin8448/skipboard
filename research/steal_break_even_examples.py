import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_profiles import plate_appearances, profiles  # noqa: E402
from pa_matchup_model import MODEL  # noqa: E402
from skipboard.plate_appearance import PlayerProfile  # noqa: E402
from skipboard.state import GameState  # noqa: E402
from skipboard.valuation import value_runs, value_wp  # noqa: E402
from skipboard.value import run_expectancy, wp  # noqa: E402

# Linear weights (runs relative to an out) to rank hitters; only used to pick examples.
WEIGHTS = {"BB": 0.69, "HBP": 0.72, "1B": 0.89, "2B": 1.27, "3B": 1.62, "HR": 2.10}
MIN_WEIGHTED_PA = 2000  # established players only

pa = plate_appearances()
cand = pa[(pa["season"] == 2025) & (pa["base_pre"] == 1) & (pa["outs_pre"] == 0)].reset_index(drop=True)
bat = profiles([(b, "batter", d) for b, d in zip(cand["batter"], cand["date"])])
pit = profiles([(p, "pitcher", d) for p, d in zip(cand["pitcher"], cand["date"])])
cand["bat_value"] = sum(bat[c] * w for c, w in WEIGHTS.items())
cand["bat_k"], cand["pit_k"] = bat["K"], pit["K"]
ok = (bat["weighted_pa"] >= MIN_WEIGHTED_PA) & (pit["weighted_pa"] >= MIN_WEIGHTED_PA)
print(f"2025 plate appearances with a runner on first only and 0 outs: {len(cand):,}; with established batter and pitcher: {int(ok.sum()):,}")

bv = cand["bat_value"].where(ok)
pk = cand["pit_k"].where(ok)
mid_p = (pk - pk.median()).abs()
mid_b = (bv - bv.median()).abs()
picks = {
    "strong hitter vs average pitcher": (mid_p.rank() <= 0.2 * ok.sum()) & ok,
    "weak hitter vs average pitcher": (mid_p.rank() <= 0.2 * ok.sum()) & ok,
    "average hitter vs high-K pitcher": (mid_b.rank() <= 0.2 * ok.sum()) & ok,
    "strong hitter vs high-K pitcher": (pk >= pk.quantile(0.9)) & ok,
    "weak hitter vs high-K pitcher": (pk >= pk.quantile(0.9)) & ok,
}
order = {
    "strong hitter vs average pitcher": bv,
    "weak hitter vs average pitcher": -bv,
    "average hitter vs high-K pitcher": pk,
    "strong hitter vs high-K pitcher": bv,
    "weak hitter vs high-K pitcher": -bv,
}
chosen = {name: order[name].where(mask).idxmax() for name, mask in picks.items()}


def break_even(value, first, second, fail):
    return (value(first) - value(fail)) / (value(second) - value(fail))


def make(r, base, outs, **game):
    return GameState(
        season=2025, game_type="regular", outs=outs, balls=0, strikes=0,
        runner1="r1" if base & 1 else None, runner2="r2" if base & 2 else None,
        batter=r["batter"], batter_hand=r["bathand"], pitcher=r["pitcher"], pitcher_hand=r["pithand"], catcher="c", **game,
    )


rows = []
for name, i in chosen.items():
    r = cand.loc[i]
    bp = PlayerProfile(r["batter"], "batter", tuple(bat.loc[i, MODEL].astype(float)), bat.at[i, "hand"], float(bat.at[i, "weighted_pa"]))
    pp = PlayerProfile(r["pitcher"], "pitcher", tuple(pit.loc[i, MODEL].astype(float)), pit.at[i, "hand"], float(pit.at[i, "weighted_pa"]))
    out = {"matchup": name, "batter": f"{r['batter']} ({r['bathand']})", "pitcher": f"{r['pitcher']} ({r['pithand']})", "date": int(r["date"]),
           "bat K": bat.at[i, "K"], "bat HR": bat.at[i, "HR"], "pit K": pit.at[i, "K"]}
    for label, value, table, game in [
        ("runs", lambda s: value_runs(s, bp, pp), run_expectancy, dict(inning=1, half=0, bat_score=0, fld_score=0)),
        ("bot 7 tied", lambda s: value_wp(s, bp, pp), wp, dict(inning=7, half=1, bat_score=2, fld_score=2)),
    ]:
        first, second, fail = make(r, 1, 0, **game), make(r, 2, 0, **game), make(r, 0, 1, **game)
        out[f"{label}: table"] = break_even(table, first, second, fail)
        out[f"{label}: matchup"] = break_even(value, first, second, fail)
    rows.append(out)

t = pd.DataFrame(rows).set_index("matchup")
for label in ["runs", "bot 7 tied"]:
    t[f"{label}: shift (pp)"] = (t[f"{label}: matchup"] - t[f"{label}: table"]) * 100
pd.set_option("display.width", 250)
print("\nsteal of second, runner on first, 0 outs, 0-0 count: break-even success rate")
print("table = league run expectancy / win probability tables; matchup = anchored value with both players' profiles as of the game date\n")
print(t[["batter", "pitcher", "date", "bat K", "bat HR", "pit K"]].to_string(float_format="{:.3f}".format))
print()
print(t[["runs: table", "runs: matchup", "runs: shift (pp)", "bot 7 tied: table", "bot 7 tied: matchup", "bot 7 tied: shift (pp)"]].to_string(float_format="{:.3f}".format))
print(f"\nrange of the matchup shift: runs {t['runs: shift (pp)'].min():+.2f} to {t['runs: shift (pp)'].max():+.2f} pp, "
      f"bottom 7th tied {t['bot 7 tied: shift (pp)'].min():+.2f} to {t['bot 7 tied: shift (pp)'].max():+.2f} pp")
