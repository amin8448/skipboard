import numpy as np
import pandas as pd

from pitch_codes import PITCHES, count_after

PITCH_LIKE = PITCHES | {"N"}  # rows are written for pitches and no-pitches
LATER_PLAY = PITCHES | {"N"}  # a pitch or no-pitch after this one means the row's event did not follow this pitch


def going_at(seq, pos):
    # '>' applies to the next pitch; only '*' (blocked) or another '>' may sit between them.
    j = pos - 1
    while j >= 0 and seq[j] in "*>":
        if seq[j] == ">":
            return True
        j -= 1
    return False


def pitcher_pickoffs(prefix):
    return sum(1 for i, ch in enumerate(prefix) if ch == "1" and (i == 0 or prefix[i - 1] != "+"))


def later_play(seq, pos):
    # Another pitch, no-pitch or pitcher pickoff throw after this pitch in the row: the row's event did not
    # happen on this pitch. Catcher throws ('+' then a digit) belong to the pitch before them.
    rest = seq[pos + 1:]
    for i, ch in enumerate(rest):
        if ch in LATER_PLAY:
            return True
        if ch in "123" and (seq[pos + i] != "+"):
            return True
    return False


def common_prefix(a, b):
    k = 0
    while k < min(len(a), len(b)) and a[k] == b[k]:
        k += 1
    return k


def add_segments(plays):
    # Plate-appearance groups: a new group starts with a new half-inning or after a PA-ending row. Within a
    # group each row's sequence repeats the earlier rows' pitches, so a row's own pitches are what it adds.
    # Expects a 'seq' column (pitch sequence, '' when missing). Adds pa_group and seg_start; returns the
    # number of rows whose sequence does not extend the previous row's (their segment starts at the common prefix).
    new_group = (plays["half_id"] != plays["half_id"].shift()) | (plays["pa"].shift() == 1)
    plays["pa_group"] = new_group.cumsum()
    prev_seq = plays.groupby("pa_group")["seq"].shift().fillna("")
    prefix_ok = np.array([s.startswith(p) for s, p in zip(plays["seq"], prev_seq)])
    plays["seg_start"] = [len(p) if ok else common_prefix(s, p) for s, p, ok in zip(plays["seq"], prev_seq, prefix_ok)]
    return int((~prefix_ok).sum())


def pitch_rows(rows):
    # One row per pitch or no-pitch in each row's own segment; 'row' is the index of the play row.
    cols = {k: [] for k in ["row", "pos", "code", "going", "balls", "strikes", "pitcher_pickoffs_before", "catcher_pickoffs_before",
                             "pitch_number", "last_play", "pa_ended", "catcher_throw_after"]}
    for r in rows[["seq", "seg_start", "pa"]].itertuples():
        seq = r.seq
        for pos in range(r.seg_start, len(seq)):
            ch = seq[pos]
            if ch not in PITCH_LIKE:
                continue
            prefix = seq[:pos]
            later = later_play(seq, pos)
            b, s = count_after(prefix)
            cols["row"].append(r.Index)
            cols["pos"].append(pos)
            cols["code"].append(ch)
            cols["going"].append(going_at(seq, pos))
            cols["balls"].append(b)
            cols["strikes"].append(s)
            cols["pitcher_pickoffs_before"].append(pitcher_pickoffs(prefix))
            cols["catcher_pickoffs_before"].append(prefix.count("+"))
            cols["pitch_number"].append(sum(c in PITCHES for c in prefix) + 1 if ch in PITCHES else np.nan)
            cols["last_play"].append(not later)
            cols["pa_ended"].append(r.pa == 1 and not any(c in PITCHES for c in seq[pos + 1:]))
            cols["catcher_throw_after"].append((not later) and "+" in seq[pos + 1:])
    return pd.DataFrame(cols)
