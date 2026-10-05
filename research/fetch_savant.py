import io
from pathlib import Path

import pandas as pd
import requests
from pybaseball import statcast_catcher_poptime, statcast_sprint_speed

YEAR = 2026
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
SAVANT = "https://baseballsavant.mlb.com/leaderboard"

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 250)


def fetch_csv(url):
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    df = pd.read_csv(io.BytesIO(resp.content))
    if df.empty:
        raise ValueError(f"no rows returned from {url}")
    return resp.content


def report(name, threshold):
    df = pd.read_csv(RAW / name)
    print(f"\n=== {name}")
    print(f"threshold: {threshold}")
    print(f"rows: {len(df)}")
    print(f"columns ({len(df.columns)}): {list(df.columns)}")
    print(df.head(3).to_string())


# Lowest threshold offered on each Savant page: sprint speed min opportunities 0,
# pop time min 2B attempts 1 / min 3B attempts 0, arm strength min throws 50,
# basestealing min pitches 1.
sprint_url = f"{SAVANT}/sprint_speed?year={YEAR}&position=&team=&min=0&csv=true"
poptime_url = f"{SAVANT}/poptime?year={YEAR}&team=&min2b=1&min3b=0&csv=true"
arm_url = f"{SAVANT}/arm-strength?type=player&year={YEAR}&minThrows=50&pos=arm_of&team=&csv=true"
steal_url = (
    f"{SAVANT}/basestealing-run-value?game_type=Regular&n=1&pitch_hand=all&runner_moved=All"
    f"&target_base=All&prior_pk=All&season_start={YEAR}&season_end={YEAR}&split=no&team=&type=Bat"
    f"&with_team_only=1&csv=true"
)

try:
    df = statcast_sprint_speed(YEAR, min_opp=0)
    if df.empty:
        raise ValueError("pybaseball returned no rows")
    df.to_csv(RAW / "sprint_speed_2026.csv", index=False)
    source = "pybaseball.statcast_sprint_speed"
except Exception as e:
    print(f"pybaseball sprint speed failed ({e}); using Savant CSV export")
    (RAW / "sprint_speed_2026.csv").write_bytes(fetch_csv(sprint_url))
    source = sprint_url
print(f"\nsource: {source}")
report("sprint_speed_2026.csv", "min opportunities = 0")

try:
    df = statcast_catcher_poptime(YEAR, min_2b_att=1, min_3b_att=0)
    if df.empty:
        raise ValueError("pybaseball returned no rows")
    df.to_csv(RAW / "catcher_pop_time_2026.csv", index=False)
    source = "pybaseball.statcast_catcher_poptime"
except Exception as e:
    print(f"pybaseball pop time failed ({e}); using Savant CSV export")
    (RAW / "catcher_pop_time_2026.csv").write_bytes(fetch_csv(poptime_url))
    source = poptime_url
print(f"\nsource: {source}")
report("catcher_pop_time_2026.csv", "min 2B attempts = 1, min 3B attempts = 0")

# pybaseball has no arm strength or basestealing function, so these use the Savant CSV export.
(RAW / "of_arm_strength_2026.csv").write_bytes(fetch_csv(arm_url))
print(f"\nsource: {arm_url}")
report("of_arm_strength_2026.csv", "min throws = 50 (position = Outfielder)")

(RAW / "runner_basestealing_2026.csv").write_bytes(fetch_csv(steal_url))
print(f"\nsource: {steal_url}")
report("runner_basestealing_2026.csv", "min pitches = 1 (runners, regular season)")
