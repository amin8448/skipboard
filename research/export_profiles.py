import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pa_matchup_model import MODEL, TABLES, load_pa, rates_as_of, shrink  # noqa: E402

HAND_COL = {"batter": "bathand", "pitcher": "pithand"}
_pa = None


def plate_appearances():
    global _pa
    if _pa is None:
        _pa = load_pa()
    return _pa


def shrinkage(tables_dir=TABLES):
    s = pd.read_csv(tables_dir / "shrinkage_v1.csv")
    k = {role: g.set_index("category").loc[MODEL, "k"].to_numpy() for role, g in s.groupby("role")}
    league = s[s["role"] == "batter"].set_index("category").loc[MODEL, "league_rate"].to_numpy()
    return k, league


def last_hand(pa, req, role):
    # Most recent handedness seen strictly before the as-of date; None for players with no history.
    hist = pa[[role, "date", HAND_COL[role]]].rename(columns={role: "player_id", HAND_COL[role]: "hand"}).sort_values("date")
    r = req[["player_id", "date"]].reset_index().sort_values("date")
    m = pd.merge_asof(r, hist, on="date", by="player_id", allow_exact_matches=False)
    return m.set_index("index")["hand"].reindex(req.index)


def profiles(requests, tables_dir=TABLES):
    # requests: list of (player_id, role, as_of_date) with role "batter" or "pitcher" and date as YYYYMMDD
    pa = plate_appearances()
    k, league = shrinkage(tables_dir)
    req = pd.DataFrame(requests, columns=["player_id", "role", "as_of_date"])
    req["date"] = req["as_of_date"].astype(int)
    req["season"] = req["date"] // 10000
    rates = np.zeros((len(req), len(MODEL)))
    wpa = np.zeros(len(req))
    hand = pd.Series(None, index=req.index, dtype=object)
    for (role, season), sub in req.groupby(["role", "season"]):
        test = sub.rename(columns={"player_id": role})
        wcounts = rates_as_of(pa, test, role, season)
        rates[sub.index] = shrink(wcounts, k[role], league)
        wpa[sub.index] = wcounts.sum(axis=1)
        hand[sub.index] = last_hand(pa, sub, role)
    out = req[["player_id", "role", "as_of_date"]].copy()
    out["hand"] = hand.where(hand.notna(), None)
    out["weighted_pa"] = wpa
    out[MODEL] = rates
    return out


if __name__ == "__main__":
    pa = plate_appearances()
    sample = pa[pa["season"] == 2025].sample(3, random_state=1)
    reqs = [(b, "batter", d) for b, d in zip(sample["batter"], sample["date"])]
    reqs += [(p, "pitcher", d) for p, d in zip(sample["pitcher"], sample["date"])]
    reqs.append(("nohistory01", "batter", 20250601))
    print(profiles(reqs).to_string(float_format="{:.4f}".format))
