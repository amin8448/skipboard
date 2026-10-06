import zipfile
from pathlib import Path

import pandas as pd
import requests

SEASONS = range(2016, 2026)
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "raw" / "retrosheet"
BASE = "https://www.retrosheet.org/downloads/plays"

OUT.mkdir(parents=True, exist_ok=True)

# Zips are saved exactly as downloaded and never extracted; CSVs are read from inside the zip.
for year in SEASONS:
    name = f"{year}plays.zip"
    path = OUT / name
    if path.exists() and zipfile.is_zipfile(path):
        print(f"\n{name}: already downloaded, skipping")
    else:
        resp = requests.get(f"{BASE}/{name}", timeout=120)
        resp.raise_for_status()
        path.write_bytes(resp.content)
        if not zipfile.is_zipfile(path):
            raise ValueError(f"{name} is not a valid zip")
        print(f"\n{name}: downloaded {len(resp.content):,} bytes")

    with zipfile.ZipFile(path) as z:
        members = z.namelist()
        print(f"  files in zip: {members}")
        with z.open(members[0]) as f:
            df = pd.read_csv(f, usecols=["gid", "gametype"], dtype=str)
    games = df.groupby("gametype", dropna=False)["gid"].nunique()
    print(f"  games: {df['gid'].nunique()} total")
    for gametype, n in games.items():
        print(f"    {gametype}: {n}")

year = SEASONS[-1]
with zipfile.ZipFile(OUT / f"{year}plays.zip") as z:
    with z.open(z.namelist()[0]) as f:
        cols = list(pd.read_csv(f, nrows=0).columns)
print(f"\ncolumns in {year}plays.csv ({len(cols)}): {cols}")
