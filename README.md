# Skipboard

An in-stadium baseball game where fans make the manager's call and are scored on decision quality.

## Setup

Requires Python 3.12.

```
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Data

Data is not stored in git. Recreate the raw Baseball Savant leaderboards in `data/raw/` with:

```
.venv/bin/python research/fetch_savant.py
```

## Scope

See [docs/SCOPE.md](docs/SCOPE.md) for the project scope.
