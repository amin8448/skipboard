import datetime
import io
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from pybaseball import statcast_catcher_poptime, statcast_sprint_speed

DEFAULT_YEAR = 2026
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
SEASON_DIR = RAW / "savant"
SAVANT = "https://baseballsavant.mlb.com/leaderboard"
MANIFEST = SEASON_DIR / "manifest.csv"

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


def check_season(df, year, columns, url):
    # Savant falls back to the current season when it does not recognise a parameter.
    for col in columns:
        seen = sorted(pd.to_numeric(df[col]).unique().tolist())
        if seen != [year]:
            raise SystemExit(f"season check failed for {url}: {col} = {seen}, requested {year}")


def add_to_manifest(path, url, rows):
    # Files in subfolders (positioning/) are recorded with the folder.
    new = not MANIFEST.exists()
    name = path.relative_to(SEASON_DIR).as_posix() if path.is_relative_to(SEASON_DIR) else path.name
    line = pd.DataFrame([{"file": name, "url": url, "download_date": datetime.date.today().isoformat(), "rows": rows}])
    line.to_csv(MANIFEST, mode="a", header=new, index=False)


def show_header(path):
    print(f"\n=== {path.name}: {len(pd.read_csv(path))} rows")
    print(f"columns: {list(pd.read_csv(path, nrows=0).columns)}")


def runner_basestealing_2b(year, path):
    # Steals of second only; n=1 is the lowest minimum the endpoint accepts (n=0 falls back to a smaller set).
    url = (
        f"{SAVANT}/basestealing-run-value?game_type=Regular&n=1&pitch_hand=all&runner_moved=All"
        f"&target_base=2B&prior_pk=All&season_start={year}&season_end={year}&split=no&team=&type=Bat"
        f"&with_team_only=1&csv=true"
    )
    content = fetch_csv(url)
    df = pd.read_csv(io.BytesIO(content))
    check_season(df, year, ["start_year", "end_year"], url)
    path.write_bytes(content)
    add_to_manifest(path, url, len(df))
    show_header(path)


def catcher_poptime_page(year, path):
    # The pop time CSV export has no season column; the page embeds the same leaderboard with a year field and
    # second-base exchange and arm strength. min2b=0 and min3b=0 are the lowest minimums the page accepts.
    url = f"{SAVANT}/poptime?year={year}&team=&min2b=0&min3b=0"
    resp = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    html = resp.text
    m = re.search(r"(?:var|let|const)\s+data\s*=\s*", html)
    if m is None:
        raise SystemExit(f"no embedded data found at {url}")
    start = html.find("[", m.end())
    depth = 0
    for end in range(start, len(html)):
        if html[end] == "[":
            depth += 1
        elif html[end] == "]":
            depth -= 1
            if depth == 0:
                break
    df = pd.DataFrame(json.loads(html[start:end + 1]))
    if df.empty:
        raise SystemExit(f"no rows returned from {url}")
    check_season(df, year, ["year"], url)
    df.to_csv(path, index=False)
    add_to_manifest(path, url, len(df))
    show_header(path)


def fetch_steal_fit(years):
    # Leaderboards the steal success model needs beyond the S1 pulls.
    SEASON_DIR.mkdir(parents=True, exist_ok=True)
    for year in years:
        print(f"\n########## {year}")
        runner_basestealing_2b(year, SEASON_DIR / f"runner_basestealing_2b_{year}.csv")
        catcher_poptime_page(year, SEASON_DIR / f"catcher_poptime_page_{year}.csv")


def embedded(html, name, url):
    # A JSON value assigned to a page variable, e.g. var data = [...] or const serverParams = {...}.
    m = re.search(rf"(?:var|let|const)\s+{name}\s*=\s*", html)
    if m is None:
        raise SystemExit(f"no embedded {name} found at {url}")
    start = m.end()
    opener = html[start]
    closer = {"[": "]", "{": "}"}[opener]
    depth = 0
    for end in range(start, len(html)):
        if html[end] == opener:
            depth += 1
        elif html[end] == closer:
            depth -= 1
            if depth == 0:
                break
    return json.loads(html[start:end + 1])


def arm_strength_page(year, path):
    # The arm strength CSV export has no season column; the page embeds the same rows with a year field.
    # minThrows=1 is the lowest minimum the page accepts (0 falls back to 50). One row per fielder-season,
    # with throws and average arm strength by position and the overall maximum.
    url = f"{SAVANT}/arm-strength?type=player&year={year}&minThrows=1&pos=arm_of&team="
    resp = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    params = embedded(resp.text, "serverParams", url)
    if int(params["minThrows"]) != 1:
        raise SystemExit(f"minimum throws not accepted at {url}: {params['minThrows']}")
    df = pd.DataFrame(embedded(resp.text, "data", url))
    if df.empty:
        raise SystemExit(f"no rows returned from {url}")
    check_season(df, year, ["year"], url)
    df.to_csv(path, index=False)
    add_to_manifest(path, url, len(df))
    return df


POSITIONING = "https://baseballsavant.mlb.com/visuals/position_data"
POS_LABEL = {7: "LF", 8: "CF", 9: "RF"}
RUNNERS = {0: "none", 1: "first_only", 27: "other"}  # the page's runner filter: no runners on, 1B only, other
SHADE = {0: "not_shaded", 1: "shaded"}


def positioning(year, pos, side, runners, shade, path):
    # Player-level start positions (distance from home and angle) from the fielder positioning visual's CSV
    # export. The filters are honoured only with a single batter side; with both sides the export returns
    # unlabelled splits, so each side is requested separately. attempts=1 is the lowest minimum.
    url = (f"{POSITIONING}?type=player&teamId=&firstBase={runners}&shift={shade}&batSide={side}&season={year}"
           f"&position={pos}&attempts=1&csv=true")
    resp = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    df = pd.read_csv(io.BytesIO(resp.content)) if resp.content.strip() else pd.DataFrame()
    if len(df):
        check_season(df, year, ["season"], url)
        if set(df["position"]) != {POS_LABEL[pos]}:
            raise SystemExit(f"position check failed for {url}: {set(df['position'])}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(resp.content)
    add_to_manifest(path, url, len(df))
    return df


def fetch_send_fit(years):
    # Leaderboards for the send decision: outfield arm strength and outfielder positioning.
    SEASON_DIR.mkdir(parents=True, exist_ok=True)
    shown = set()
    for year in years:
        print(f"\n########## {year}")
        arm = arm_strength_page(year, SEASON_DIR / f"arm_strength_of_page_{year}.csv")
        if "arm" not in shown:
            print(f"arm strength columns: {list(arm.columns)}")
            shown.add("arm")
        for pos, label in POS_LABEL.items():
            n = int((pd.to_numeric(arm[f"total_throws_{label.lower()}"]) > 0).sum())
            n_arm = int(pd.to_numeric(arm[f"arm_{label.lower()}"], errors="coerce").notna().sum())
            print(f"  arm strength {label}: fielder-seasons with throws {n}, with an average arm {n_arm}")
        for pos, label in POS_LABEL.items():
            for side in ("R", "L"):
                for runners, rname in RUNNERS.items():
                    for shade, sname in SHADE.items():
                        path = SEASON_DIR / "positioning" / f"positioning_{year}_{label}_{side}_{rname}_{sname}.csv"
                        df = positioning(year, pos, side, runners, shade, path)
                        if "pos" not in shown and len(df):
                            print(f"positioning columns: {list(df.columns)}")
                            shown.add("pos")
                        print(f"  positioning {label} vs {side}, runners {rname}, {sname}: {len(df)} fielder rows")
                        time.sleep(0.5)


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
    # With --steal-fit and seasons: the extra leaderboards for the steal success model.
    # With --send-fit and seasons: outfield arm strength and outfielder positioning for the send decision.
    if len(sys.argv) > 1 and sys.argv[1] == "--steal-fit":
        fetch_steal_fit([int(y) for y in sys.argv[2:]])
    elif len(sys.argv) > 1 and sys.argv[1] == "--send-fit":
        fetch_send_fit([int(y) for y in sys.argv[2:]])
    elif len(sys.argv) > 1:
        fetch_seasons([int(y) for y in sys.argv[1:]])
    else:
        fetch_default()
