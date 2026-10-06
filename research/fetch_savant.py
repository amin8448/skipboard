import io
import sys
from pathlib import Path

import pandas as pd
import requests
from pybaseball import statcast_catcher_poptime, statcast_sprint_speed

DEFAULT_YEAR = 2026
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
SEASON_DIR = RAW / "savant"
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


def report(path, threshold):
    df = pd.read_csv(path)
    print(f"\n=== {path.name}")
    print(f"threshold: {threshold}")
    print(f"rows: {len(df)}")
    print(f"columns ({len(df.columns)}): {list(df.columns)}")
    print(df.head(3).to_string())


# Lowest threshold offered on each Savant page: sprint speed min opportunities 0,
# pop time min 2B attempts 1 / min 3B attempts 0, arm strength min throws 50,
# basestealing and pitcher running game min pitches (SB opportunities) 1.
def sprint_speed(year, path):
    url = f"{SAVANT}/sprint_speed?year={year}&position=&team=&min=0&csv=true"
    try:
        df = statcast_sprint_speed(year, min_opp=0)
        if df.empty:
            raise ValueError("pybaseball returned no rows")
        df.to_csv(path, index=False)
        source = "pybaseball.statcast_sprint_speed"
    except Exception as e:
        print(f"pybaseball sprint speed failed ({e}); using Savant CSV export")
        path.write_bytes(fetch_csv(url))
        source = url
    print(f"\nsource: {source}")
    report(path, "min opportunities = 0")


def pop_time(year, path):
    url = f"{SAVANT}/poptime?year={year}&team=&min2b=1&min3b=0&csv=true"
    try:
        df = statcast_catcher_poptime(year, min_2b_att=1, min_3b_att=0)
        if df.empty:
            raise ValueError("pybaseball returned no rows")
        df.to_csv(path, index=False)
        source = "pybaseball.statcast_catcher_poptime"
    except Exception as e:
        print(f"pybaseball pop time failed ({e}); using Savant CSV export")
        path.write_bytes(fetch_csv(url))
        source = url
    print(f"\nsource: {source}")
    report(path, "min 2B attempts = 1, min 3B attempts = 0")


# pybaseball has no arm strength, basestealing or pitcher running game function, so these use the Savant CSV export.
def arm_strength(year, path):
    url = f"{SAVANT}/arm-strength?type=player&year={year}&minThrows=50&pos=arm_of&team=&csv=true"
    path.write_bytes(fetch_csv(url))
    print(f"\nsource: {url}")
    report(path, "min throws = 50 (position = Outfielder)")


def runner_basestealing(year, path):
    url = (
        f"{SAVANT}/basestealing-run-value?game_type=Regular&n=1&pitch_hand=all&runner_moved=All"
        f"&target_base=All&prior_pk=All&season_start={year}&season_end={year}&split=no&team=&type=Bat"
        f"&with_team_only=1&csv=true"
    )
    path.write_bytes(fetch_csv(url))
    print(f"\nsource: {url}")
    report(path, "min pitches = 1 (runners, regular season)")


def pitcher_running_game(year, path):
    # Leads allowed (primary, secondary, gained between first move and release) for steals of second.
    url = (
        f"{SAVANT}/pitcher-running-game?game_type=Regular&n=1&pitch_hand=all&runner_moved=All"
        f"&target_base=2B&prior_pk=All&season_start={year}&season_end={year}&split=no&team=&type=Pit"
        f"&with_team_only=1&csv=true"
    )
    path.write_bytes(fetch_csv(url))
    print(f"\nsource: {url}")
    report(path, "min SB opportunities = 1 (pitchers, regular season, target base 2B)")


def fetch_default():
    # The original 2026 download, unchanged: four leaderboards into data/raw/.
    sprint_speed(DEFAULT_YEAR, RAW / f"sprint_speed_{DEFAULT_YEAR}.csv")
    pop_time(DEFAULT_YEAR, RAW / f"catcher_pop_time_{DEFAULT_YEAR}.csv")
    arm_strength(DEFAULT_YEAR, RAW / f"of_arm_strength_{DEFAULT_YEAR}.csv")
    runner_basestealing(DEFAULT_YEAR, RAW / f"runner_basestealing_{DEFAULT_YEAR}.csv")


def fetch_seasons(years):
    SEASON_DIR.mkdir(parents=True, exist_ok=True)
    for year in years:
        print(f"\n########## {year}")
        sprint_speed(year, SEASON_DIR / f"sprint_speed_{year}.csv")
        pop_time(year, SEASON_DIR / f"catcher_pop_time_{year}.csv")
        runner_basestealing(year, SEASON_DIR / f"runner_basestealing_{year}.csv")
        pitcher_running_game(year, SEASON_DIR / f"pitcher_running_game_2b_{year}.csv")


if __name__ == "__main__":
    # No arguments: the 2026 download. With seasons, e.g. 2023 2024 2025: steal-related leaderboards per season.
    if len(sys.argv) > 1:
        fetch_seasons([int(y) for y in sys.argv[1:]])
    else:
        fetch_default()
