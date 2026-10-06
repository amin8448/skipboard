import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_profiles import plate_appearances, profiles  # noqa: E402
from pa_matchup_model import COUNT_LABEL, MODEL, CATS, count_condition, full_model, saved_league_tables  # noqa: E402
from skipboard.plate_appearance import CATEGORIES, PlayerProfile, outcome_distribution  # noqa: E402
from skipboard.state import GameState  # noqa: E402

N = 2000
SEED = 2026

assert list(CATS) == list(CATEGORIES)
pa = plate_appearances()
sample = pa[pa["season"] == 2025].sample(N, random_state=SEED).reset_index(drop=True)
rng = np.random.default_rng(SEED)
sample["count"] = [rng.choice(c.split("|")) for c in sample["counts_seen"]]  # one count each PA passed through

bat = profiles([(b, "batter", d) for b, d in zip(sample["batter"], sample["date"])])
pit = profiles([(p, "pitcher", d) for p, d in zip(sample["pitcher"], sample["date"])])
print(f"{N} random 2025 plate appearances (seed {SEED}); profiles with no history: batters {int((bat['weighted_pa'] == 0).sum())}, pitchers {int((pit['weighted_pa'] == 0).sum())}")
print("counts used: " + ", ".join(f"{c} {n}" for c, n in sample["count"].value_counts().reindex(COUNT_LABEL, fill_value=0).items()))

# Research side: the matchup script's own functions on the saved 2023-2025 tables.
lg = saved_league_tables()
b = bat[MODEL].to_numpy()
q = pit[MODEL].to_numpy()
p9 = full_model(b, q, lg, sample["matchup"], sample["state"])  # platoon and base-out state ratios on
research = count_condition(p9, lg, lg["counts"].loc[sample["count"]].to_numpy(), sample["state"].to_numpy())

# Engine side.
engine = np.zeros_like(research)
for i, r in sample.iterrows():
    balls, strikes = (int(x) for x in r["count"].split("-"))
    state = GameState(
        season=2025, game_type="regular", inning=int(r["inning"]), half=int(r["top_bot"]), bat_score=0, fld_score=0,
        outs=int(r["outs_pre"]), balls=balls, strikes=strikes,
        runner1="r1" if r["base_pre"] & 1 else None, runner2="r2" if r["base_pre"] & 2 else None, runner3="r3" if r["base_pre"] & 4 else None,
        batter=r["batter"], batter_hand=r["bathand"], pitcher=r["pitcher"], pitcher_hand=r["pithand"], catcher="c",
    )
    bp = PlayerProfile(r["batter"], "batter", tuple(b[i].tolist()), bat.at[i, "hand"], float(bat.at[i, "weighted_pa"]))
    pp = PlayerProfile(r["pitcher"], "pitcher", tuple(q[i].tolist()), pit.at[i, "hand"], float(pit.at[i, "weighted_pa"]))
    engine[i] = list(outcome_distribution(state, bp, pp).values())

diff = np.abs(engine - research)
worst = np.unravel_index(diff.argmax(), diff.shape)
print(f"\nlargest absolute difference in any category probability: {diff.max():.3e} ({CATEGORIES[worst[1]]}, row {worst[0]})")
print(f"mean absolute difference: {diff.mean():.3e}")
print(f"below 1e-9: {diff.max() < 1e-9}")

# ROE and OTHER enter count conditioning at their league count-conditional rates and are then
# renormalized with the rest; this shows how far that leaves them from the table values.
table = lg["counts"].loc[sample["count"]]
for cat in ["ROE", "OTHER"]:
    gap = np.abs(engine[:, CATEGORIES.index(cat)] - table[cat].to_numpy())
    print(f"{cat}: largest gap from the league count-conditional rate after renormalizing: {gap.max():.2e}")
