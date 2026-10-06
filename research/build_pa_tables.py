from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PLAYS = ROOT / "data" / "derived" / "plays_regular_2016_2025.parquet"
OUT_PA = ROOT / "data" / "derived" / "pa_outcomes_2016_2025.parquet"
OUT = ROOT / "models" / "pa_model"

BASE_ORDER = [0, 1, 2, 4, 3, 5, 6, 7]
BASE_LABEL = {0: "___", 1: "1__", 2: "_2_", 4: "__3", 3: "12_", 5: "1_3", 6: "_23", 7: "123"}
CATEGORIES = ["K", "BB", "HBP", "1B", "2B", "3B", "HR", "GB_OUT", "AIR_OUT", "ROE", "FC", "OTHER"]
COUNTS = [(b, s) for b in range(4) for s in range(3)]
COUNT_LABEL = [f"{b}-{s}" for b, s in COUNTS]

# Pitch codes from https://www.retrosheet.org/eventfile.htm ("The pitches field of the play record").
BALLS = set("BIPV")  # ball, intentional ball, pitchout, automatic/called ball (V)
STRIKES = set("ACKMQSTO")  # automatic strike, called, unknown strike, missed bunt, swing on pitchout, swinging, foul tip, foul tip on bunt
FOULS = set("FR")  # foul, foul on pitchout: a strike only with fewer than 2 strikes
FOUL_BUNT = "L"  # always a strike, so a foul bunt with 2 strikes is strike three
ENDS = set("HXY")  # hit batter, ball in play, ball in play on pitchout
UNKNOWN = "U"  # unknown or missed pitch: a pitch, effect on the count unknown
PITCHES = BALLS | STRIKES | FOULS | {FOUL_BUNT} | ENDS | {UNKNOWN}
NOT_PITCHES = set("+*.123>N")  # catcher pickoff, blocked, play not involving batter, pickoffs, runner going, no pitch
NOT_THROWN = set("VA")  # automatic ball/strike: change the count but are not thrown, and nump does not count them
DOCUMENTED = PITCHES | NOT_PITCHES

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 400)

plays = pd.read_parquet(PLAYS)

print("\n=== Step 1: pitch sequences")
print(f"pitch codes:     {''.join(sorted(PITCHES))}")
print(f"not pitches:     {''.join(sorted(NOT_PITCHES))}")
print(f"pitches that change the count but are not thrown (excluded from pitch totals): {''.join(sorted(NOT_THROWN))}")

seqs = plays["pitches"].fillna("")
chars = Counter("".join(seqs))
undocumented = {ch: n for ch, n in chars.items() if ch not in DOCUMENTED}
print(f"characters in data: {dict(chars.most_common())}")
print(f"undocumented characters: {undocumented if undocumented else 'none'}")

# Does the PA-ending row hold the full sequence when a non-PA play happened mid-PA?
nxt = plays.shift(-1)
mid = (plays["pa"] == 0) & (plays["event"] != "NP") & (nxt["half_id"] == plays["half_id"]) & (nxt["batter"] == plays["batter"])
mid_idx = np.flatnonzero(mid.to_numpy())
pa_idx = np.flatnonzero((plays["pa"] == 1).to_numpy())
end_idx = pa_idx[np.minimum(np.searchsorted(pa_idx, mid_idx), len(pa_idx) - 1)]
same_pa = (plays["half_id"].to_numpy()[end_idx] == plays["half_id"].to_numpy()[mid_idx]) & (plays["batter"].to_numpy()[end_idx] == plays["batter"].to_numpy()[mid_idx])
before = seqs.to_numpy()[mid_idx][same_pa]
after = seqs.to_numpy()[end_idx][same_pa]
is_prefix = np.array([a.startswith(b) for b, a in zip(before, after)])
next_char = Counter(a[len(b):len(b) + 1] for b, a, ok in zip(before, after, is_prefix) if ok)
print(f"\nnon-PA plays in the middle of a PA (steal, wild pitch, pickoff and so on): {len(mid_idx):,}; with the PA ending later in the same half-inning: {int(same_pa.sum()):,}")
print(f"  the non-PA row's sequence is a prefix of the PA-ending row's sequence: {int(is_prefix.sum()):,} ({is_prefix.mean():.3%})")
print(f"  character right after the prefix: {dict(next_char.most_common(5))}")
for b, a in list(zip(before[~is_prefix], after[~is_prefix]))[:5]:
    print(f"  not a prefix: {b!r} then {a!r}")


def parse(seq):
    b = s = 0
    seen = []
    thrown = 0
    valid = True
    unknown = False
    for ch in seq:
        if ch not in PITCHES:
            continue
        if b > 3 or s > 2:
            valid = False
        seen.append(min(b, 3) * 3 + min(s, 2))
        if ch not in NOT_THROWN:
            thrown += 1
        if ch in BALLS:
            b += 1
        elif ch in STRIKES or ch == FOUL_BUNT:
            s += 1
        elif ch in FOULS:
            if s < 2:
                s += 1
        elif ch == UNKNOWN:
            unknown = True
    return seen, thrown, valid, unknown


pa = plays[plays["pa"] == 1].copy()
parsed = [parse(x) for x in pa["pitches"].fillna("")]
pa["n_thrown"] = [p[1] for p in parsed]
seq_valid = np.array([p[2] for p in parsed])
seq_unknown = np.array([p[3] for p in parsed])
last = np.array([p[0][-1] if p[0] else -1 for p in parsed])
masks = np.array([sum(1 << c for c in set(p[0])) for p in parsed], dtype=np.int64)
pa["counts_seen"] = ["|".join(COUNT_LABEL[c] for c in dict.fromkeys(p[0])) for p in parsed]
count_match = last == (pa["balls"].clip(upper=3) * 3 + pa["strikes"].clip(upper=2)).to_numpy()
count_match &= (pa["balls"] <= 3).to_numpy() & (pa["strikes"] <= 2).to_numpy()
pa["count_ok"] = count_match & seq_valid & ~seq_unknown

print(f"\nplate appearances: {len(pa):,}")
print(f"  no pitch codes at all: {int((last == -1).sum()):,}")
print(f"  sequence keeps going after ball four or strike three: {int((~seq_valid).sum()):,}")
print(f"  contains an unknown pitch (U): {int(seq_unknown.sum()):,}")
print(f"  count before final pitch = balls/strikes columns: {int(count_match.sum()):,} ({count_match.mean():.4%})")
bad = pa.loc[~count_match, ["season", "event", "pitches", "balls", "strikes"]]
if len(bad):
    print(bad.head(8).to_string())
print(f"  count_ok (count matches, sequence valid, no unknown pitch): {int(pa['count_ok'].sum()):,} ({pa['count_ok'].mean():.4%})")

has_dot = pa["pitches"].fillna("").str.contains(".", regex=False)
after_dot = pa["pitches"].fillna("").str.rsplit(".", n=1).str[-1].map(lambda x: sum(ch in PITCHES and ch not in NOT_THROWN for ch in x))
full_match = pa["n_thrown"] == pa["nump"]
seg_match = after_dot == pa["nump"]
print(f"\npitch totals vs nump (pitches thrown in the full sequence, V and A not counted)")
print(f"  all PAs: {full_match.mean():.4%} match")
print(f"  PAs with no mid-PA play (no '.'): {full_match[~has_dot].mean():.4%} of {int((~has_dot).sum()):,}")
print(f"  PAs with a '.' marker: {full_match[has_dot].mean():.4%} of {int(has_dot.sum()):,}; nump matches only the part after the last '.': {seg_match[has_dot].mean():.4%}")
conv = pd.DataFrame({"season": pa["season"], "full sequence": full_match, "after last '.'": seg_match})[has_dot.to_numpy()]
print("  PAs with a '.' marker, match rate by season (nump's convention changed in 2023):")
print(conv.groupby("season").mean().T.map("{:.3f}".format).to_string())
print(f"  PAs where nump matches neither: {int((~full_match & ~seg_match).sum()):,}")

print("\n=== Step 2: outcome categories")
outcome_cols = ["single", "double", "triple", "hr", "sh", "sf", "hbp", "walk", "k", "xi", "roe", "fc", "othout", "noout"]
n_flags = pa[outcome_cols].sum(axis=1)
print(f"PAs with exactly one of Retrosheet's 14 PA-result columns set: {int((n_flags == 1).sum()):,} of {len(pa):,}")
batter_out = (pa["othout"] == 1) | (pa["sf"] == 1) | (pa["sh"] == 1)
# Bunts carry their type only in hittype (BG bunt grounder, BP bunt pop-up, BL bunt liner).
on_ground = (pa["ground"] == 1) | (pa["hittype"] == "BG")
in_air = (pa["fly"] == 1) | (pa["line"] == 1) | pa["hittype"].isin(["BP", "BL"])
conditions = [
    pa["k"] == 1,
    pa["walk"] == 1,
    pa["hbp"] == 1,
    pa["single"] == 1,
    pa["double"] == 1,
    pa["triple"] == 1,
    pa["hr"] == 1,
    pa["roe"] == 1,
    pa["fc"] == 1,
    batter_out & (pa["sf"] == 1),
    batter_out & on_ground,
    batter_out & in_air,
]
choices = ["K", "BB", "HBP", "1B", "2B", "3B", "HR", "ROE", "FC", "AIR_OUT", "GB_OUT", "AIR_OUT"]
pa["category"] = np.select(conditions, choices, "OTHER")
pa["exclude_reason"] = np.select([pa["iw"] == 1, (pa["sh"] == 1) | (pa["bunt"] == 1)], ["intentional_walk", "bunt"], "")
pa["excluded"] = pa["exclude_reason"] != ""
print("categories, all seasons (excluded PAs included):")
print(pd.crosstab(pa["category"], pa["exclude_reason"].replace("", "kept"), margins=True).reindex(CATEGORIES + ["All"]).to_string())
other = pa[pa["category"] == "OTHER"]
print(f"OTHER breakdown: interference {int((other['xi'] == 1).sum()):,}, no-out other {int((other['noout'] == 1).sum()):,}, batter out with unknown batted-ball type {int((batter_out[other.index]).sum()):,}, other {int(((other["xi"] == 0) & (other["noout"] == 0) & ~batter_out[other.index]).sum()):,}")

save_cols = ["gid", "date", "season", "inning", "top_bot", "batter", "pitcher", "bathand", "pithand",
             "base_pre", "outs_pre", "base_post", "outs_post", "runs", "category", "excluded", "exclude_reason", "counts_seen", "count_ok"]
OUT_PA.parent.mkdir(parents=True, exist_ok=True)
pa[save_cols].to_parquet(OUT_PA, index=False)
print(f"saved {OUT_PA.relative_to(ROOT)} ({len(pa):,} rows, {len(save_cols)} columns)")

print("\n=== Step 3: league tables, 2023-2025, excluded PAs removed")
league = pa[pa["season"].between(2023, 2025) & ~pa["excluded"]].copy()
league_masks = masks[(pa["season"].between(2023, 2025) & ~pa["excluded"]).to_numpy()]
print(f"PAs: {len(league):,} (count_ok: {int(league['count_ok'].sum()):,})")

# a. count-conditional outcomes, PAs with a validated count only
ok = league["count_ok"].to_numpy()
rows = []
for i, (b, s) in enumerate(COUNTS):
    through = ok & ((league_masks >> i) & 1).astype(bool)
    vc = league.loc[through, "category"].value_counts().reindex(CATEGORIES, fill_value=0)
    n = int(through.sum())
    for cat, k in vc.items():
        rows.append({"balls": b, "strikes": s, "category": cat, "k": int(k), "n": n, "p": k / n})
count_tab = pd.DataFrame(rows)
wide = count_tab.pivot(index=["balls", "strikes"], columns="category", values="p")[CATEGORIES]
wide.index = [f"{b}-{s}" for b, s in wide.index]
wide_print = wide.map("{:.3f}".format)
wide_print["n"] = count_tab.groupby(["balls", "strikes"])["n"].first().map("{:,}".format).to_numpy()
print("\na. outcome distribution among PAs that passed through each count")
print(wide_print.to_string())

# b. platoon rates; switch hitters bat opposite the pitcher
league["bat_side"] = np.where(league["bathand"] == "B", np.where(league["pithand"] == "R", "L", "R"), league["bathand"])
league["platoon"] = np.where(league["bat_side"] == league["pithand"], "same", "opposite")
league["matchup"] = league["bat_side"] + "_vs_" + league["pithand"]
plat_rows = []
for col in ["matchup", "platoon"]:
    for group, sub in league.groupby(col):
        vc = sub["category"].value_counts().reindex(CATEGORIES, fill_value=0)
        for cat, k in vc.items():
            plat_rows.append({"group_type": col, "group": group, "category": cat, "k": int(k), "n": len(sub), "p": k / len(sub)})
plat = pd.DataFrame(plat_rows)
pw = plat.pivot(index="group", columns="category", values="p")[CATEGORIES]
pw = pw.loc[["L_vs_L", "L_vs_R", "R_vs_L", "R_vs_R", "same", "opposite"]]
ratio = pw.loc["same"] / pw.loc["opposite"]
pw_print = pw.map("{:.3f}".format)
pw_print.loc["same / opposite"] = ratio.map("{:.2f}".format)
pw_print["n"] = list(plat.groupby("group")["n"].first().reindex(pw.index).map("{:,}".format)) + [""]
print("\nb. outcome distribution by matchup (batter side vs pitcher hand; switch hitters bat opposite)")
print(pw_print.to_string())
print(f"switch hitters: {int((league['bathand'] == 'B').sum()):,} PAs ({(league['bathand'] == 'B').mean():.1%})")

# c. runner advancement, innings 1-8. Singles are split by the first fielder to touch the ball.
ADV_CATEGORIES = ["K", "BB", "HBP", "1B_IF", "1B_OF", "2B", "3B", "HR", "GB_OUT", "AIR_OUT", "ROE", "FC", "OTHER"]
# Source priority per cell: data if n >= MIN_N, else the rule-based result for these categories,
# else the same base state and category pooled across outs. 3B uses data and pooling only.
RULE_CATEGORIES = ["HR", "BB", "HBP", "OTHER"]
MIN_N = 30
KEYS = ["base_pre", "outs_pre", "category"]
OUTCOME = ["base_post", "outs_post", "runs"]

adv_src = league[league["inning"].between(1, 8)].copy()
is_single = adv_src["category"] == "1B"
adv_src.loc[is_single & adv_src["firstf"].between(1, 6), "category"] = "1B_IF"
adv_src.loc[is_single & adv_src["firstf"].between(7, 9), "category"] = "1B_OF"
unclassified = is_single & ~adv_src["firstf"].between(1, 9)
all_singles = pa["category"] == "1B"
print(f"\nc. runner advancement, innings 1-8: {len(adv_src):,} PAs")
print(f"  singles split by first fielder (1-6 infield, 7-9 outfield): {int(is_single.sum()):,} singles, unclassified (first fielder unknown): {int(unclassified.sum())}")
print(f"  across all seasons and innings: {int(all_singles.sum()):,} singles, unclassified: {int((all_singles & ~pa['firstf'].between(1, 9)).sum())}")
if unclassified.any():
    print("  unclassified singles are left out of the advancement table")
    adv_src = adv_src[~unclassified]

singles = adv_src[adv_src["category"].isin(["1B_IF", "1B_OF"])]
split = singles.groupby(["base_pre", "outs_pre"]).agg(n_if=("category", lambda c: int((c == "1B_IF").sum())), n=("category", "size")).reset_index()
split["if_share"] = split["n_if"] / split["n"]
split = split.rename(columns={"base_pre": "base_code", "outs_pre": "outs"})
split.insert(1, "base_label", split["base_code"].map(BASE_LABEL))
split["order"] = split["base_code"].map(BASE_ORDER.index)
split = split.sort_values(["order", "outs"]).drop(columns="order")
print(f"\n  infield share of singles by base-out state (overall {(singles['category'] == '1B_IF').mean():.3f}, n={len(singles):,})")
sg = split.set_index(["base_label", "outs"])
print(pd.DataFrame({o: sg.xs(o, level="outs")["if_share"].map("{:.3f}".format) + " (" + sg.xs(o, level="outs")["n"].map("{:,}".format) + ")" for o in range(3)}).reindex([BASE_LABEL[b] for b in BASE_ORDER]).to_string())


def rule_result(category, base, outs):
    if category == "HR":
        return 0, outs, bin(base).count("1") + 1
    if category == "3B":
        return 4, outs, bin(base).count("1")
    # BB, HBP and OTHER (almost all catcher's interference): the batter takes first and only
    # forced runners move up one base.
    if not base & 1:
        return base | 1, outs, 0
    if not base & 2:
        return base | 3, outs, 0
    if not base & 4:
        return 7, outs, 0
    return 7, outs, 1


print("\n  agreement with rule-based results (HR everyone scores; 3B all runners score, batter on third;")
print("  BB, HBP and OTHER only forced runners advance). Rules fill cells with n below 30, except 3B.")
for cat in RULE_CATEGORIES + ["3B"]:
    sub = adv_src[adv_src["category"] == cat]
    expected = [rule_result(cat, b, o) for b, o in zip(sub["base_pre"], sub["outs_pre"])]
    match = (sub[OUTCOME].to_numpy() == np.array(expected)).all(axis=1)
    print(f"    {cat}: agreement {match.mean():.4%} ({int(match.sum()):,} of {len(sub):,}){'' if cat in RULE_CATEGORIES else ' (rule not used)'}")
    if not match.all():
        ex = sub[~match].assign(state=lambda d: d["base_pre"].map(BASE_LABEL) + " " + d["outs_pre"].astype(str))
        top = ex.groupby(["state", "base_post", "outs_post", "runs"]).size().sort_values(ascending=False).head(6)
        print("      most common exceptions (state, base_post, outs_post, runs: count): " + "; ".join(f"{k[0]} -> {k[1]},{k[2]},{k[3]}: {v}" for k, v in top.items()))

observed = adv_src.groupby(KEYS + OUTCOME).size().rename("k").reset_index()
all_cells = pd.MultiIndex.from_product([range(8), range(3), ADV_CATEGORIES], names=KEYS)
n_cell = observed.groupby(KEYS)["k"].sum().reindex(all_cells, fill_value=0)

# Back-off pool: same base state and category across all outs, stored as outs added on the play.
pool = adv_src.assign(outs_added=adv_src["outs_post"] - adv_src["outs_pre"]).groupby(["base_pre", "category", "base_post", "outs_added", "runs"]).size().rename("k").reset_index()

parts = []
backed_off = []
for (base, outs, cat), n in n_cell.items():
    if n >= MIN_N:
        rows = observed[(observed["base_pre"] == base) & (observed["outs_pre"] == outs) & (observed["category"] == cat)]
        parts.append(rows.assign(n=n, n_cell=n, source="data"))
    elif cat in RULE_CATEGORIES:
        bp, op, r = rule_result(cat, base, outs)
        parts.append(pd.DataFrame({"base_pre": [base], "outs_pre": [outs], "category": [cat], "base_post": [bp], "outs_post": [op], "runs": [r], "k": [n], "n": [n], "n_cell": [n], "source": ["rule"]}))
    else:
        rows = pool[(pool["base_pre"] == base) & (pool["category"] == cat)].copy()
        n_pool = int(rows["k"].sum())
        backed_off.append((base, outs, cat, int(n), n_pool))
        if n_pool == 0:
            continue
        rows["outs_post"] = outs + rows["outs_added"]
        # A play that reaches 3 outs ends the half-inning: bases clear and no runs are credited.
        ended = rows["outs_post"] >= 3
        rows.loc[ended, ["outs_post", "base_post", "runs"]] = [3, 0, 0]
        rows = rows.groupby(["base_post", "outs_post", "runs"])["k"].sum().reset_index()
        parts.append(rows.assign(base_pre=base, outs_pre=outs, category=cat, n=n_pool, n_cell=int(n), source="backoff"))

adv = pd.concat(parts, ignore_index=True)
adv["p"] = np.where(adv["source"] == "rule", 1.0, adv["k"] / adv["n"].where(adv["n"] > 0))
adv["cat_order"] = adv["category"].map(ADV_CATEGORIES.index)
adv = adv.sort_values(["base_pre", "outs_pre", "cat_order", "base_post", "outs_post", "runs"])
adv = adv[["base_pre", "outs_pre", "category", "base_post", "outs_post", "runs", "k", "n", "p", "n_cell", "source"]]

print(f"\n  cells backed off to the same base state and category pooled across outs (n_cell -> pooled n): {len(backed_off)}")
bo = pd.DataFrame(backed_off, columns=["base_pre", "outs_pre", "category", "n_cell", "n_pool"])
bo["state"] = bo["base_pre"].map(BASE_LABEL) + " " + bo["outs_pre"].astype(str) + " out"
for cat, sub in bo.groupby("category", sort=False):
    print(f"    {cat}: " + ", ".join(f"{st} ({a} -> {b})" for st, a, b in zip(sub["state"], sub["n_cell"], sub["n_pool"])))
still_small = bo[bo["n_pool"] < MIN_N]
print(f"  cells still below {MIN_N} after data, rule and pooling: {len(still_small)}" + (": " + ", ".join(f"{c} {st} (pooled n={n})" for c, st, n in zip(still_small["category"], still_small["state"], still_small["n_pool"])) if len(still_small) else ""))
print("    (pooled n of 0 means no rows were written for the cell)")
cell_source = adv.groupby(KEYS)["source"].first().reindex(all_cells).fillna("none (no data)")
print("\n  cells by source (24 base-out states x 13 categories = 312):")
print(cell_source.value_counts().to_string())
print(pd.crosstab(cell_source.index.get_level_values("category"), cell_source.values).reindex(ADV_CATEGORIES).fillna(0).astype(int).to_string())

print("\n=== Step 4: sanity checks")
shares = league["category"].value_counts(normalize=True).reindex(CATEGORIES)
print(f"category shares, 2023-2025, {len(league):,} PAs (excluded: {int((pa['season'].between(2023, 2025) & pa['excluded']).sum()):,}):")
print(shares.map("{:.3f}".format).to_string())
print("\ncount table rows 0-0, 0-2, 3-0, 3-2:")
print(wide_print.loc[["0-0", "0-2", "3-0", "3-2"]].to_string())


def scores_from_second(table, cat):
    cell = table[(table["base_pre"] == 2) & (table["outs_pre"] == 0) & (table["category"] == cat)]
    return cell.loc[cell["runs"] >= 1, "p"].sum(), int(cell["n"].iloc[0]), cell["source"].iloc[0]


p_if, n_if, src_if = scores_from_second(adv, "1B_IF")
p_of, n_of, src_of = scores_from_second(adv, "1B_OF")
share_if = split.set_index(["base_code", "outs"]).loc[(2, 0), "if_share"]
g = adv[(adv["base_pre"] == 1) & (adv["outs_pre"] == 0) & (adv["category"] == "GB_OUT")]
print("\nrunner on second only, 0 outs, single: runner scores")
print(f"  1B_IF: {p_if:.3f} (n={n_if:,}, {src_if}); 1B_OF: {p_of:.3f} (n={n_of:,}, {src_of}); infield share {share_if:.3f}; combined {share_if * p_if + (1 - share_if) * p_of:.3f}")
print("ball four with a runner on second only (base_post, outs_post, runs: p)")
for outs in range(3):
    cell = adv[(adv["base_pre"] == 2) & (adv["outs_pre"] == outs) & (adv["category"] == "BB")]
    print(f"  {outs} outs ({cell['source'].iloc[0]}, n={int(cell['n'].iloc[0]):,}): " + "; ".join(f"{BASE_LABEL[b]},{o},{r}: {q:.3f}" for b, o, r, q in zip(cell["base_post"], cell["outs_post"], cell["runs"], cell["p"])))
print("runner on first only, 0 outs, ground-ball out: double play (outs_post = 2)")
print(f"  {g.loc[g['outs_post'] == 2, 'p'].sum():.3f} (n={int(g['n'].iloc[0]):,}, {g['source'].iloc[0]})")

OUT.mkdir(parents=True, exist_ok=True)
count_tab.assign(p=count_tab["p"].round(6)).to_csv(OUT / "count_outcomes_v1.csv", index=False)
plat.assign(p=plat["p"].round(6)).to_csv(OUT / "platoon_rates_v1.csv", index=False)
adv.assign(p=adv["p"].round(6)).to_csv(OUT / "advancement_v1.csv", index=False)
split.assign(if_share=split["if_share"].round(6)).to_csv(OUT / "single_split_v1.csv", index=False)
print(f"\nsaved models/pa_model/count_outcomes_v1.csv ({len(count_tab)} rows), platoon_rates_v1.csv ({len(plat)} rows), advancement_v1.csv ({len(adv):,} rows), single_split_v1.csv ({len(split)} rows)")
