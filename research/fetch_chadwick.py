from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "raw" / "chadwick"
REPO = "chadwickbureau/register"
SHARDS = "0123456789abcdef"

# Saves the register's people files and README unmodified, pinned to the current commit.
OUT.mkdir(parents=True, exist_ok=True)
sha = requests.get(f"https://api.github.com/repos/{REPO}/commits/master", timeout=60).json()["sha"]
for name in [f"data/people-{x}.csv" for x in SHARDS] + ["README.md"]:
    resp = requests.get(f"https://raw.githubusercontent.com/{REPO}/{sha}/{name}", timeout=120)
    resp.raise_for_status()
    (OUT / Path(name).name).write_bytes(resp.content)
    print(f"{Path(name).name}: {len(resp.content):,} bytes")
(OUT / "COMMIT").write_text(sha + "\n")
print(f"commit {sha}")
