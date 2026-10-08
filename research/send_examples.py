import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_profiles import profiles  # noqa: E402
from pa_matchup_model import MODEL  # noqa: E402
from skipboard.plate_appearance import PlayerProfile  # noqa: E402
from skipboard.send_decision import send_decision  # noqa: E402
from skipboard.send_model import BattedBall, FielderProfile, PositioningStart, load_batter_advance, load_send_model  # noqa: E402
from skipboard.state import GameState  # noqa: E402
from skipboard.steal_model import RunnerProfile  # noqa: E402
from skipboard.value import wp  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIT = ROOT / "data" / "derived" / "send" / "send_fit_table_2023_2025.parquet"
SC = ROOT / "data" / "derived" / "send" / "statcast_singles_runner_on_second_2023_2025.parquet"
ATTRS = ROOT / "data" / "derived" / "steal" / "player_season_attributes_2023_2025.parquet"
ARM = ROOT / "data" / "raw" / "savant" / "arm_strength_of_page_{}.csv"
FILLS = ROOT / "data" / "derived" / "send" / "send_fill_values_2023_2025.json"
TEAM = "TOR"
SEASON = 2025
MIN_WEIGHTED_PA = 2000  # established players, as in the other example scripts
POS_LABEL = {7: "LF", 8: "CF", 9: "RF"}

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)

fit = pd.read_parquet(FIT)
sc = pd.read_parquet(SC, columns=["game_pk", "at_bat_number", "home_team", "away_team", "bat_score", "fld_score"])
fit = fit.merge(sc, on=["game_pk", "at_bat_number"], how="left")
fit["batting_team"] = np.where(fit["half"] == 0, fit["away_team"], fit["home_team"])
fit["late_close"] = (fit["inning"] >= 7) & (fit["score_diff"].abs() <= 1)
pool = fit[(fit["season"] == SEASON) & (fit["batting_team"] == TEAM)].copy()
print(f"rule: {SEASON} plays from the send fit table with {TEAM} batting; for each outs count (0, 1, 2) take one send and one hold; within each cell "
      "prefer a late-and-close play (inning 7 or later, score within one run), then the earliest date and at-bat")
print("candidates by outs and call (late and close in brackets):")
cells = pool.groupby(["outs", "sent"]).agg(plays=("gid", "size"), late_close=("late_close", "sum"))
print(cells.rename(index={True: "send", False: "hold"}, level=1).to_string())
picks = []
for outs in range(3):
    for sent in (True, False):
        cell = pool[(pool["outs"] == outs) & (pool["sent"] == sent)]
        if len(cell):
            picks.append(cell.sort_values(["late_close", "date", "at_bat_number"], ascending=[False, True, True]).iloc[0])
picks = pd.DataFrame(picks).reset_index(drop=True)
print(f"selected: {len(picks)} plays; sends {int(picks['sent'].sum())}, holds {int((~picks['sent']).sum())}, late and close {int(picks['late_close'].sum())}")

model = load_send_model()
shares = load_batter_advance()
fills = json.loads(FILLS.read_text())
attrs = pd.read_parquet(ATTRS, columns=["retro_id", "season", "runner_sprint_speed"]).set_index(["retro_id", "season"])["runner_sprint_speed"]
arm_page = pd.read_csv(str(ARM).format(SEASON)).set_index("player_id")
bat_prof = profiles([(b, "batter", int(d)) for b, d in zip(picks["on_deck_used"], picks["date"])])
pit_prof = profiles([(p, "pitcher", int(d)) for p, d in zip(picks["pitcher"], picks["date"])])


def player(prof, i, role):
    r = prof.loc[i]
    if r["weighted_pa"] < MIN_WEIGHTED_PA:
        return None, f"{r['player_id']} (weighted PA {r['weighted_pa']:.0f}, below {MIN_WEIGHTED_PA}: league average)"
    return PlayerProfile(r["player_id"], role, tuple(float(x) for x in r[MODEL]), r["hand"], float(r["weighted_pa"])), \
        f"{r['player_id']} (weighted PA {r['weighted_pa']:.0f})"


def fmt(x, digits=4):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{digits}f}"


for i, r in picks.iterrows():
    pos = int(r["fielder_pos"])
    label = POS_LABEL[pos].lower()
    filled = []
    sprint = attrs.get((r["runner_on_second"], SEASON))
    sprint = None if sprint is None or pd.isna(sprint) else float(sprint)
    if sprint is None:
        filled.append("runner sprint speed")
    own_arm = arm_page[f"arm_{label}"].get(int(r["fielder_mlbam"])) if int(r["fielder_mlbam"]) in arm_page.index else None
    own_arm = None if own_arm is None or pd.isna(own_arm) else float(own_arm)
    if own_arm is None:
        filled.append(f"{POS_LABEL[pos]} arm strength")
    league_start = fills["positioning_start_by_season_position_side"][f"{SEASON} {POS_LABEL[pos]} {r['stand']}"]
    start_filled = abs(r["start_distance"] - league_start["start_distance"]) < 1e-9 and abs(r["start_angle"] - league_start["start_angle"]) < 1e-9
    start = PositioningStart() if start_filled else PositioningStart(start_distance=float(r["start_distance"]), start_angle=float(r["start_angle"]))
    if start_filled:
        filled.append("positioning start")
    launch_speed = None if pd.isna(r["launch_speed"]) else float(r["launch_speed"])
    launch_angle = None if pd.isna(r["launch_angle"]) else float(r["launch_angle"])
    if launch_speed is None or launch_angle is None:
        filled.append("launch speed and angle")
    on_deck, on_deck_note = player(bat_prof, i, "batter")
    pitcher, pitcher_note = player(pit_prof, i, "pitcher")
    ball = BattedBall(x_feet=float(r["x_feet"]), y_feet=float(r["y_feet"]), bb_type=r["bb_type"], fielder_pos=pos,
                      launch_speed=launch_speed, launch_angle=launch_angle)
    state = GameState(season=SEASON, game_type="regular", inning=int(r["inning"]), half=int(r["half"]), bat_score=int(r["bat_score"]),
                      fld_score=int(r["fld_score"]), outs=int(r["outs"]), balls=0, strikes=0, runner2=r["runner_on_second"], batter=r["batter"],
                      batter_hand=r["bat_side"], pitcher=r["pitcher"], pitcher_hand=r["pitcher_hand"], catcher=r["catcher"])
    common = dict(model=model, batter_advance=shares, on_deck_id=r["on_deck_used"], on_deck_hand=bat_prof.loc[i, "hand"] or "R")
    mine = send_decision(state, ball, runner=RunnerProfile(sprint_speed=sprint), fielder=FielderProfile(arm_avg=own_arm), start=start,
                         on_deck=on_deck, pitcher=pitcher, **common)
    base = send_decision(state, ball, runner=RunnerProfile(), fielder=FielderProfile(), start=PositioningStart(), on_deck=None, pitcher=pitcher, **common)
    derived = model.derived_covariates(ball, start, r["stand"], SEASON)
    opponent = r["away_team"] if r["batting_team"] == r["home_team"] else r["home_team"]
    print("\n" + "=" * 150)
    print(f"play {i + 1}: game {r['gid']}, {int(r['date'])}, {'bottom' if r['half'] == 1 else 'top'} {int(r['inning'])}, {int(r['outs'])} out"
          f"{'s' if r['outs'] != 1 else ''}, {TEAM} {int(r['bat_score'])} {opponent} {int(r['fld_score'])}{' (late and close)' if r['late_close'] else ''}")
    print(f"  batter {r['batter']} ({r['bat_side']}), runner on second {r['runner_on_second']} (sprint {fmt(sprint, 1)} ft/s), "
          f"fielder {r['fielder_id']} at {POS_LABEL[pos]} (arm {fmt(own_arm, 1)} mph), pitcher {pitcher_note}")
    print(f"  batted ball: {r['bb_type']}, launch {fmt(launch_speed, 1)} mph at {fmt(launch_angle, 0)} degrees; fielding point ({r['x_feet']:.0f}, "
          f"{r['y_feet']:.0f}) ft, {derived.fielding_distance:.0f} ft from home, {derived.spray_toward_line:.1f} degrees "
          f"{'toward the ' + POS_LABEL[pos] + ' line' if pos != 8 else 'from straightaway center'}; "
          f"fielder start {fmt(start.start_distance, 0)} ft at {fmt(start.start_angle, 0)} degrees, {derived.fielder_distance:.0f} ft to the ball")
    print(f"  on-deck batter: {on_deck_note}; filled attributes: {', '.join(filled) if filled else 'none'}")
    print(f"  win probability before the play: {wp(state):.4f}; coach's call: {'send' if r['sent'] else 'hold'}; outcome: {r['runner_result']}, "
          f"batter to {r['batter_advance']}")
    table = pd.DataFrame({
        "this play": [mine.hold_runs, mine.send_runs, mine.hold_wp, mine.send_wp, mine.p_safe, mine.p_send, mine.break_even_runs,
                      mine.break_even_wp, mine.verdict, mine.confidence, mine.close_call],
        "league-average baseline": [base.hold_runs, base.send_runs, base.hold_wp, base.send_wp, base.p_safe, base.p_send, base.break_even_runs,
                                    base.break_even_wp, base.verdict, base.confidence, base.close_call],
    }, index=["hold (runs)", "send (runs)", "hold (win probability)", "send (win probability)", "p_safe", "p_send (coaches)", "break-even (runs)",
              "break-even (win probability)", "verdict", "confidence", "close call"])
    print(table.to_string(formatters={c: (lambda x: x if isinstance(x, (str, bool, np.bool_)) else fmt(x)) for c in table.columns}))
print("\nbaseline: league runner speed, league arm for the position, league positioning start and a league-average on-deck batter; same ball, state "
      "and pitcher; confidence is the share of the 200 bootstrap replicates of p_safe on the verdict's side of the win-probability break-even")
