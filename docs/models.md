# Models

Four models score game situations: a run expectancy table (re24_v1), a win probability model (wp v1), a plate-appearance model (pa v1), and a steal-of-second model (steal v1). The first three are built from Retrosheet regular-season play-by-play data alone (see `docs/retrosheet_notice.md`). The first two are league-average models: they describe a typical team, not the teams on the field. The plate-appearance model adds the batter and pitcher. The steal model's success component also uses Baseball Savant season aggregates, which MLB permits only for non-commercial use: data files derived from Savant stay under `data/` and are not tracked, and the coefficient table documented below is a snapshot of the fitted model (see section 10, "Data sources and rights", in `docs/SCOPE.md`).

## Valuation principle

Every option in a decision is valued through the same path. A fan is scored on the difference between options, so a model gap that affects every option equally cancels in the comparison.

Decision values are anchored. The level comes from the empirical tables, and the plate-appearance model adds only the player and count effects:

- `value_wp(state, batter, pitcher)` = `wp(state)` + `continue_pa_wp(state, batter, pitcher)` - `continue_pa_wp(state at 0-0, league-average batter and pitcher with the same handedness)`.
- `value_runs` is the same with `run_expectancy` and `continue_pa_runs`.
- Both engine terms use the same platoon and base-out state settings. With league-average players at 0-0 the two engine terms are identical, so the value equals the table exactly.
- Decision modules use `value_wp` and `value_runs` (`engine/src/skipboard/valuation.py`). `continue_pa_wp` and `continue_pa_runs` stay available as the raw model.

Why anchor: the raw plate-appearance model differs from the tables by up to 0.051 runs in a single state (see "Consistency checks and tolerances"). Those gaps are not equal across the states a decision compares, so they do not fully cancel. The raw model's steal break-even rate differed from the tables' by up to 3 percentage points, which is enough to flip a verdict on an attempt near break-even. Anchored, league-average players at 0-0 reproduce the tables exactly, so every option in a decision starts from the tables' level.

Example: the steal of second. "Not this pitch" is the anchored continuation at the current count: `value_wp` (or `value_runs`) of the current state with the same batter and pitcher. "Run on this pitch" is a mixture over what the pitch does with the runner going: ball, strike, strike three, foul, hit by pitch, ball four or in play, each valued at the state it leads to. Where the catcher throws (a ball, a strike, or strike three with fewer than 2 outs), that branch splits into safe and out by the success model. See "Steal of second (steal v1)".

Player information moves the answer. Five real 2025 matchups, with a runner on first, 0 outs and a 0-0 count, the bottom of the 7th tied, and each matchup's own runner, catcher and pitcher (`research/steal_break_even_examples.py`):

| Matchup | Effective throw success | Break-even, runs | Break-even, win probability | Run minus hold, runs |
|---|---|---|---|---|
| Strong hitter vs average pitcher | 0.707 | 0.780 | 0.714 | -0.055 |
| Weak hitter vs average pitcher | 0.802 | 0.670 | 0.604 | +0.091 |
| Average hitter vs high-strikeout pitcher | 0.815 | 0.725 | 0.661 | +0.063 |
| Strong hitter vs high-strikeout pitcher (left-handed) | 0.774 | 0.762 | 0.700 | +0.010 |
| Weak hitter vs high-strikeout pitcher | 0.781 | 0.675 | 0.607 | +0.075 |
| Baseline, right-handed pitcher | 0.768 | 0.706 | 0.638 | +0.046 |
| Baseline, left-handed pitcher | 0.789 | 0.704 | 0.637 | +0.063 |
| League-average attempt | 0.811 | 0.706 | 0.638 | +0.078 |

The baselines use league-average batter and pitcher, and runner, catcher and pitcher attributes at the 2025 fill values for the pitcher's hand. The league-average attempt uses the same players, with the throw success on each throw branch set to the empirical safe rate for that pitch result and count on the fit table (2023 to 2025, pitchouts excluded): 0.830 on a ball and 0.789 on a strike at 0-0. Against a strong hitter an out on the bases costs more, so the break-even rises by up to 7.4 points over the baseline and running loses value. The league-average attempt succeeds more often than the model baseline (0.811 against 0.768), because real attempts come from faster and more aggressive runners than the league-average profile.

## Run expectancy (re24_v1)

### Purpose

Expected runs scored from a given base-out state to the end of the half-inning. It is the baseline for judging decisions that trade outs for bases, such as steals and bunts.

### Inputs

| Input | Values |
|---|---|
| Bases occupied (`base_code`) | 0 to 7 (first = 1, second = 2, third = 4) |
| Outs | 0, 1, 2 |

Output per cell: `re` (expected runs), `se` (standard error), `n` (observations).

### Data and seasons

- Regular seasons 2023 to 2025, innings 1 to 8.
- Only half-innings that end with 3 outs. Non-play (NP) rows are dropped.
- 116,584 half-innings and 511,694 plays.
- Script: `research/run_expectancy.py`. Table: `models/run_expectancy/re24_v1.csv`.

### Method

The table uses the standard event-based method. It keeps every play where the base-out state changes or a run scores. Run expectancy for a state is the average number of runs scored from that play to the end of the half-inning, over all plays starting in that state.

### Results

| Bases | 0 outs | 1 out | 2 outs |
|---|---|---|---|
| ___ | 0.505 | 0.270 | 0.103 |
| 1__ | 0.902 | 0.533 | 0.233 |
| _2_ | 1.157 | 0.693 | 0.329 |
| __3 | 1.386 | 0.959 | 0.355 |
| 12_ | 1.535 | 0.952 | 0.465 |
| 1_3 | 1.870 | 1.222 | 0.508 |
| _23 | 1.991 | 1.410 | 0.570 |
| 123 | 2.379 | 1.623 | 0.794 |

### Validation

| Check | Result |
|---|---|
| Standard errors (bootstrap by game, 500 replicates) | 0.002 (bases empty, 2 outs) to 0.054 (bases loaded, 0 outs) |
| Event-based vs plate-appearance-only table | Largest difference 0.013 runs (_23, 2 outs) |
| Change from 2016 to 2019 | Only the bases-empty row changed clearly: -0.018, -0.010, -0.005 runs (z = -4.4, -3.0, -2.4). All other cells are within about 2 standard errors. |
| Shape | Run expectancy falls with each out in every base state. |

### Known limits

- League-average only: no adjustment for batter, pitcher, park or weather.
- Innings 1 to 8 only. Ninth-inning and extra-inning play, including the automatic runner on second, is not represented.
- Rare states are less precise. Runner on third with 0 outs has 1,121 observations; bases loaded with 0 outs has a standard error of 0.054.
- Covers the seasons since the 2023 rule changes only.

## Win probability (wp v1)

### Purpose

The probability that the batting team wins, given the inning, half, score and base-out state. It is used to score the decisions fans make.

### Inputs

| Input | Values |
|---|---|
| Inning | 1 to 12 |
| Half (`half`) | 0 = top, 1 = bottom |
| Score difference (`diff`) | -10 to 10, batting team minus fielding team |
| Bases occupied (`base_code`) | 0 to 7 |
| Outs | 0, 1, 2 |

Output: `wp_bat`, the batting team's win probability. There are two versions. The regular-season version starts extra innings with a runner on second. The postseason version starts them with the bases empty.

### Data and seasons

- Run distributions come from regular seasons 2023 to 2025, innings 1 to 8, with the same play filter as re24_v1.
- For validation, a second set of tables was built from 2023 to 2024 and tested on the 2025 regular season.
- Script: `research/win_probability.py`. Files in `models/win_probability/`: `run_dist_v1.csv`, `wp_regular_v1.csv`, `wp_postseason_v1.csv`.

### Method

For each half (top or bottom) and each base-out state, the model estimates the chance of scoring 0, 1, ... 9 or 10+ more runs in the half-inning. It then steps through the rest of the game one half-inning at a time, applying baseball's ending rules: walk-offs, no bottom of the ninth when the home team leads, and extra innings until a winner. Extra innings are run to inning 25. Values settle well before that: the largest change from the 10th to the 11th is under 0.00001 in both versions.

### Validation (tables from 2023 to 2024, tested on 2025)

The test set is 189,291 plays in 2,430 games, innings 1 to 12.

| Brier score | Value | Plays |
|---|---|---|
| Overall | 0.156 | 189,291 |
| Innings 1 to 3 | 0.217 | 63,574 |
| Innings 4 to 6 | 0.158 | 64,127 |
| Innings 7 to 9 | 0.087 | 59,110 |
| Innings 10 to 12 | 0.177 | 2,480 |

A constant 50% forecast scores 0.250.

| Predicted | Mean predicted | Actual win rate | Plays |
|---|---|---|---|
| 0.0 to 0.1 | 0.035 | 0.026 | 23,095 |
| 0.1 to 0.2 | 0.149 | 0.119 | 11,549 |
| 0.2 to 0.3 | 0.249 | 0.212 | 12,155 |
| 0.3 to 0.4 | 0.347 | 0.336 | 12,947 |
| 0.4 to 0.5 | 0.456 | 0.439 | 30,336 |
| 0.5 to 0.6 | 0.558 | 0.571 | 27,959 |
| 0.6 to 0.7 | 0.650 | 0.659 | 16,979 |
| 0.7 to 0.8 | 0.749 | 0.774 | 13,920 |
| 0.8 to 0.9 | 0.849 | 0.878 | 15,314 |
| 0.9 to 1.0 | 0.963 | 0.974 | 25,037 |

| Team batting second | Model | Actual 2025 | Games |
|---|---|---|---|
| Start of game | 0.526 | 0.543 | 2,430 |
| Start of a tied 10th | 0.497 | 0.526 | 209 |

Both gaps are within normal variation for one season: about 1.7 standard errors for the start of game and under 1 for the tied 10th.

Selected values from the final regular-season version (2023 to 2025 tables):

| State | wp_bat |
|---|---|
| Top 1st, tied, bases empty, 0 outs | 0.470 |
| Bottom 9th, tied, bases empty, 0 outs | 0.644 |
| Bottom 9th, tied, runner on second, 0 outs | 0.809 |
| Bottom 9th, down 1, runner on first, 0 outs | 0.342 |
| Top 10th, tied, runner on second, 0 outs | 0.494 |

### Known limits

- **Both teams are treated as league-average.** A team that is ahead is more often the better team, so leads hold more often than an average-team model expects. This explains the underconfidence seen in the 2025 calibration. Below 0.5, actual win rates fall short of predictions by up to 0.037. Above 0.5, actual win rates run ahead of predictions by up to 0.029.
- **Half-innings are treated as independent.** Each one starts fresh from the same run distributions, regardless of what came before or who is due up.
- **Extra-inning values are rough.** They reuse the innings 1 to 8 distribution for a runner on second with 0 outs, and ignore one-run strategy such as bunting the runner over. The 2025 check rests on only 209 games.
- **Run distributions come from innings 1 to 8.** Late-inning bullpen use and strategy in the ninth and later are not modeled.

Recalibration (fitting a correction to the predicted values) was deliberately not applied, because it would hide the cause of the gap rather than fix it. The planned fix is a team-strength adjustment. It will be judged by whether it closes the calibration gap on held-out data.

## Plate appearance (pa v1)

### Purpose

The probability of each plate-appearance outcome for a given batter, pitcher and count, and what happens to the runners afterward. It has two parts: league tables, and a matchup model that adjusts them for the players involved.

### Inputs

| Input | Values |
|---|---|
| Batter and pitcher | Retrosheet player IDs, with their history up to the game date |
| Handedness | Batter L, R or B (switch); pitcher L or R |
| Count | Any of the 12 ball-strike counts |
| Base-out state | `base_code` 0 to 7 and outs 0 to 2 (for runner advancement) |

Outcome categories:

| Category | Meaning |
|---|---|
| K, BB, HBP | Strikeout, unintentional walk, hit by pitch |
| 1B, 2B, 3B, HR | Hits |
| GB_OUT | Out on a ground ball |
| AIR_OUT | Fly, line or pop-up out, including sacrifice flies |
| ROE, FC | Reached on error, fielder's choice |
| OTHER | Almost all catcher's interference |

### Data and seasons

- Plate appearances from 2016 to 2025 are in `data/derived/pa_outcomes_2016_2025.parquet` (1,721,249 rows).
- Intentional walks and plate appearances ending in a bunt stay in that file with an `excluded` flag. They are left out of all tables and fits.
- League tables use 2023 to 2025: 544,378 plate appearances. Runner advancement uses innings 1 to 8 only (491,316).
- Player rates use the current season and the three before it.
- Scripts: `research/build_pa_tables.py` (league tables) and `research/pa_matchup_model.py` (matchup model and validation). Files in `models/pa_model/`: `count_outcomes_v1.csv`, `platoon_rates_v1.csv`, `advancement_v1.csv`, `single_split_v1.csv`, `fc_share_v1.csv`, `state_ratio_v1.csv`, `shrinkage_v1.csv`.
- Engine: `engine/src/skipboard/plate_appearance.py`, `transitions.py` and `continue_pa.py` reproduce the research model to within 1e-15 (`research/check_engine_parity.py`).

### Method: league tables

- **Counts.** The count before every pitch is rebuilt from Retrosheet's pitch sequences, using the codes in Retrosheet's own documentation. The count before the final pitch matches Retrosheet's recorded count in 99.99% of plate appearances. The count table gives the outcome distribution for all plate appearances that passed through each count.
- **Platoon.** Outcome rates by batter side and pitcher hand, with switch hitters treated as batting opposite the pitcher.
- **Runner advancement.** For each base-out state and outcome, the table gives the distribution of the resulting base state, outs and runs. Singles are split into infield (first fielder 1 to 6) and outfield (7 to 9). The league infield share of singles by base-out state is saved separately.
- **Source priority.** Each advancement cell takes the first of these that applies:
  1. Data, if the cell has at least 30 plate appearances.
  2. Otherwise, a rule-based result, for HR, BB, HBP and OTHER.
  3. Otherwise, the same base state and outcome pooled across outs.

The rules:

| Outcome | Rule |
|---|---|
| HR | Everyone scores |
| BB, HBP, OTHER | Batter to first; only forced runners advance one base |
| 3B | No rule; uses data and pooling only |

- **FC share.** The share of fielder's choices among ground balls (GB_OUT plus FC) by base-out state. It is 0 with bases empty and highest for plays at the plate: 0.28 with a runner on third and 1 out, 0.34 with runners on second and third and 1 out.
- **Base-out state ratio.** For each base-out state, each outcome's share divided by its overall share. With first base open, walks rise by a factor of 1.15 to 1.43; with a runner on third and fewer than 2 outs, singles rise by a factor of 1.03 to 1.27 (shrunk values).
- **State ratio shrinkage.** Each state's share is pulled toward the overall share: (n x share + m x overall share) / (n + m). The prior strength m is estimated per outcome by method of moments across the 24 states: the variance of the state shares minus the average sampling variance. An outcome with no variance beyond sampling noise would get a ratio of 1 everywhere; none did. Triples are shrunk most (raw range 0.60 to 1.50, shrunk 0.86 to 1.09), walks and strikeouts least.

| Outcome | K | BB | HBP | 1B | 2B | 3B | HR | GB | AIR_OUT |
|---|---|---|---|---|---|---|---|---|---|
| Prior strength m | 653 | 230 | 2,481 | 500 | 3,703 | 41,816 | 4,916 | 3,207 | 2,775 |

### Method: matchup model

- **Weighting.** Each player's outcome counts are weighted 6 for the current season up to the day before the game, then 5, 4 and 3 for the three previous seasons. Same-day games are excluded.
- **Shrinkage.** Each rate is pulled toward the league rate: (weighted count + k x league rate) / (weighted PA + k), then renormalized. k is estimated for each category and role with a beta-binomial fit across players on 2022 to 2024 counts. It is fitted per actual plate appearance, then multiplied by the average weight per PA (4.0) so it matches the weighted counts.
- **Combination.** Batter and pitcher are combined with the odds-ratio method: each outcome's probability is proportional to batter rate x pitcher rate / league rate.
- **Platoon.** The result is multiplied by the league platoon ratio for the matchup's handedness, then renormalized.
- **Base-out state.** The result is multiplied by the shrunk state ratio for the state's bases and outs, then renormalized. It can be turned off with a flag (`use_state` in research, `apply_state` in the engine).
- **Count conditioning.** For a plate appearance already at count c, the league outcome distribution for c is multiplied by the matchup's ratio to league for each outcome, then renormalized.

The model covers nine outcomes: GB_OUT and FC are combined as GB, and ROE and OTHER stay at league rates. GB is split back into GB_OUT and FC using the league FC share for the base-out state.

Shrinkage strength (k per actual plate appearance; higher means more shrinkage):

| Category | Batter | Pitcher |
|---|---|---|
| K | 43 | 70 |
| BB | 129 | 179 |
| HBP | 267 | 437 |
| 1B | 243 | 382 |
| 2B | 1,704 | 1,097 |
| 3B | 546 | 3,658 |
| HR | 156 | 687 |
| GB | 98 | 92 |
| AIR_OUT | 121 | 101 |

### Validation: league tables

| Check | Result |
|---|---|
| Full pitch sequence on the PA-ending row when a steal or pickoff happens mid-PA | 99.97% of 51,411 cases |
| Count before the final pitch matches the recorded count | 99.99% |
| Rule agreement with data | HR 100%, BB 99.55%, HBP 99.98%, OTHER 99.61%, 3B 98.60% (rule not used) |
| Advancement cells by source (312 total) | 254 data, 26 rule, 29 pooled, 3 empty (fielder's choice with bases empty, which cannot happen) |
| Runner on second only, 0 outs, scores on a single | Infield single 0.119, outfield single 0.438 |
| Runner on first only, 0 outs, ground-ball out is a double play | 0.492 |

Same-side platoon rates relative to opposite-side: K 1.05, BB 0.83, HBP 1.41, HR 0.89, GB_OUT 1.06.

### Validation: matchup model (tables from 2023 to 2024, tested on 2025)

The test set is 181,084 plate appearances. Log loss is per plate appearance, over all 12 outcomes.

| Model | Log loss | Improvement over league only |
|---|---|---|
| League only | 1.8476 | |
| Batter only | 1.8289 | 1.01% |
| Batter and pitcher | 1.8184 | 1.58% |
| Batter, pitcher and platoon | 1.8179 | 1.61% |
| Batter, pitcher, platoon and base-out state (full model) | 1.8166 | 1.68% |

The state adjustment adds 0.0013. Unshrunk state ratios scored 1.81666; shrunk, 1.81659.

- **HR calibration:** within 0.003 in every decile of predicted probability.
- **K calibration:** close overall (0.227 predicted, 0.224 actual), but the top three deciles are overpredicted by 0.006 to 0.011. Without the state adjustment the overprediction was 0.005 to 0.009, so the adjustment makes it slightly worse.

| Count conditioning | Plate appearances | League count table | Count-conditioned full model | Improvement | Without state adjustment |
|---|---|---|---|---|---|
| Through 0-2 | 38,924 | 1.5750 | 1.5483 | 1.70% | 1.5486 |
| Through 3-0 | 7,132 | 1.3206 | 1.3123 | 0.63% | 1.3119 |
| Through 3-2 | 25,752 | 1.6945 | 1.6780 | 0.98% | 1.6776 |

### Consistency checks and tolerances

With league-average players at 0-0, the engine's one-step value of letting the plate appearance play out (`continue_pa`) should match the run expectancy and win probability tables. It does not match exactly, and the gap has four sources. For runs, across the 24 base-out states:

| Source of the gap | Largest | Mean | What it is |
|---|---|---|---|
| Memorylessness | 0.031 runs | 0.009 | Half-innings that reach a state score more afterward than the average from the next state (for example, a weaker pitcher is still in) |
| Steals, wild pitches, balks | 0.018 | 0.007 | Runs and advances outside plate appearances, which `continue_pa` does not model |
| Excluded bunts and intentional walks | 0.010 | 0.002 | Left out of the outcome and advancement tables |
| Left in the engine | 0.026 | 0.006 | Outcome mix and advancement smoothing, including state ratio shrinkage |
| **Total** | **0.051** | **0.015** | Largest at runners on first and third, 0 outs |

The largest win probability gap is 0.012 (runners on first and third, 0 outs, top of the 9th, tied).

Steal of second, break-even success rate at 0-0 with a runner on first only, from the tables and from the engine:

| Context | Outs | Tables | Engine | Difference |
|---|---|---|---|---|
| Runs | 0 / 1 / 2 | 0.712 / 0.729 / 0.710 | 0.727 / 0.759 / 0.708 | +1.5 / +3.0 / -0.2 points |
| Top 9th, tied | 0 / 1 / 2 | 0.600 / 0.625 / 0.623 | 0.608 / 0.651 / 0.602 | +0.8 / +2.5 / -2.2 points |
| Bottom 9th, tied | 0 / 1 / 2 | 0.572 / 0.602 / 0.587 | 0.573 / 0.611 / 0.571 | +0.1 / +0.9 / -1.6 points |
| Bottom 7th, tied | 0 / 1 / 2 | 0.642 / 0.670 / 0.644 | 0.650 / 0.685 / 0.640 | +0.8 / +1.5 / -0.5 points |

The raw model's break-even rates differ from the tables' by up to 3.0 percentage points. This is why decision values are anchored (see "Valuation principle").

These checks are diagnostics of the raw plate-appearance model (`continue_pa`), not of decision values. Decision values are anchored to the tables (see "Valuation principle") and match them exactly for league-average players at 0-0, which a separate test checks.

The diagnostic test tolerances are the largest observed gap plus a margin, set after the state ratio shrinkage: runs 0.051 + 0.005 = 0.056; win probability 0.012 + 0.002 = 0.0143; break-even rates 3.03 + 0.5 = 3.54 percentage points. A change to the model that widens any gap fails the tests.

### Known limits

- **The matchup effect is assumed the same at every count.** A high-strikeout pitcher gets the same relative boost at 3-0 as at 0-2.
- **Strikeouts are slightly overpredicted in the top bins in 2025** (by up to 0.011). Part of this is a lower strikeout environment in 2025 than in the 2023 to 2024 tables; the state adjustment adds about 0.002.
- **No park factors.**
- **No individual platoon splits.** Every player gets the league platoon ratio for their handedness.
- **Switch hitters are assumed to bat opposite the pitcher.**
- **ROE and OTHER stay at league rates** for every matchup.
- **Season weights (6, 5, 4, 3) are fixed, not tuned.**
- **Players with no history get league rates.** In 2025 this applied to 365 plate appearances on the batter side and 1,859 on the pitcher side.
- **`nump` is not used.** Retrosheet's pitch-count column follows different conventions in different seasons when a play happens mid-PA, so pitch counts come from the pitch sequence instead.
- **Some advancement cells are pooled across outs,** mostly triples, errors and fielder's choices with runners on third.
- **Base-out state ratios are league-wide.** They are the same for every matchup and every count.
- **`continue_pa` does not model steals, wild pitches or balks,** and treats half-innings as memoryless. Its values differ from the run expectancy table by up to 0.051 runs in a single state.

## Steal of second (steal v1)

### Purpose

The value of sending the runner from first on the next pitch, compared with holding. The fan picks one of two options: "run on this pitch" or "not this pitch". The model has three parts: running-game tables built from Retrosheet, a success model for the catcher's throw, and a decision module that combines them with the plate-appearance model.

The decision is offered with a runner on first only (second and third empty), at any count except 3-2 with 2 outs. There the runner goes automatically: from 2023 to 2025 the runner went on 6,155 of 6,200 such pitches (99.3%), so there is nothing to decide.

### Inputs

| Input | Values |
|---|---|
| Game state | Runner on first only, count, outs, inning, half and score (for win probability), batter side, season |
| Pickoff throws | Pitcher pickoff throws to first earlier in the plate appearance |
| Batter and pitcher | Plate-appearance profiles as in pa v1; league average if absent |
| Runner | Sprint speed (ft/s), aggressiveness |
| Catcher | Pop time to second (s) |
| Pitcher holding the runner | Hand, primary lead allowed (ft), lead gained (secondary minus primary lead allowed, ft) |

Any runner, catcher or pitcher attribute may be missing; missing values are filled (see "Fill values").

Output: the hold and run values in win probability and runs, the branches, the effective throw success (the share-weighted success probability over the throw branches), and the break-even throw success in each currency.

### Data and seasons

- Running-game tables: Retrosheet pitch sequences, regular seasons 2023 to 2025, with a few cells from a wider window, 2016 to 2025 without 2019 (see "Coverage"). 69,318 pitches with a runner on first and second open carry the runner-going marker; 58,228 of them have a runner on first only.
- Success model: 8,649 throw situations from 2023 to 2025 (2,802, 2,914 and 2,933 by season). Evaluation trains on 2023 to 2024 (5,716) and tests on 2025 (2,933). The final fit uses all three seasons.
- Player attributes: Baseball Savant season leaderboards, 2023 to 2025, linked to Retrosheet IDs through the Chadwick register (see `docs/chadwick_notice.md`).
- Scripts: `research/build_steal_attempts.py` (steal attempt table), `research/steal_pitches.py` (pitch-level helpers), `research/build_steal_tables.py` (running-game tables), `research/fetch_savant.py` and `research/fetch_chadwick.py` (downloads), `research/build_steal_fit_table.py` (player attributes and fit table), `research/fit_steal_success.py` (success model), `research/steal_break_even_examples.py` (examples).
- Tracked files in `models/steal/`: `pitch_result_going_v1.csv`, `advancement_going_v1.csv`, `inplay_outcomes_by_count_v1.csv`, `fc_share_going_v1.csv`. These use Retrosheet only.
- Not tracked, in `data/derived/steal/`: `steal_success_v1_coefficients.csv`, `steal_success_v1_meta.json`, `steal_success_v1_player_effects.csv` (header only, because the production model has no player effects), `player_season_attributes_2023_2025.parquet`, `steal_fit_table_2023_2025.parquet`.

### Method: decision

"Not this pitch" is `value_wp` (or `value_runs`) of the current state, with the same batter and pitcher (see "Valuation principle"). "Run on this pitch" is the sum, over pitch results, of the result's share at the count (`pitch_result_going_v1.csv`) times the value of its branch:

| Pitch result | Branch |
|---|---|
| Ball | Throw. Safe: runner on second, one more ball. Out: bases empty, one more out, one more ball. |
| Strike | Throw. As for a ball, with one more strike. |
| Strike three, 0 or 1 out | Throw. Safe: batter out, runner on second. Out: double play. The plate appearance ends. |
| Strike three, 2 outs | Third out, no throw. |
| Ball four, hit by pitch | Runners on first and second. The plate appearance ends. |
| Foul | One more strike below 2 strikes. At 2 strikes the state is unchanged, so the branch equals holding. |
| In play | In-play outcome mix at the count, then advancement with the runner going. The plate appearance ends. |

- **States within the plate appearance** (after a ball, a strike, a foul, or a caught stealing that is not the third out) are valued with `value_wp` and `value_runs`, with the same batter and pitcher at the new count.
- **States that end the plate appearance** are valued at the tables: win probability, or run expectancy plus runs scored, for the new base-out state. This is the value of the next plate appearance starting at 0-0 with league-average players; the on-deck batter is not modeled. A caught stealing for the third out ends the half-inning.
- **In play.** The league in-play mix at the count (`inplay_outcomes_by_count_v1.csv`) is multiplied by the matchup's ratio to league for each category (platoon on, base-out state ratio off): both kinds of single by the 1B ratio, extra-base hits and air outs by their own, ground balls by the GB ratio, and ROE stays at league. Ground balls are split into GB_OUT and FC by the going FC share. The mix is renormalized, and each category's states come from `advancement_going_v1.csv`.
- **Break-even.** The single success probability on every throw branch at which running equals holding, found by bisection on [0, 1] with 100 steps. If no probability in [0, 1] equalizes the two, there is no break-even.

### Method: running-game tables

**The runner-going marker.** Retrosheet pitch sequences mark a pitch on which the runner went with ">". The marker recovers what the steal record cannot: every pitch the runner went on, including those fouled off or put in play (which erase the attempt) and ball four (the walk forces the runner, so no steal is credited).

**Coverage.** The share of steals of second (runner on first, second open) whose steal pitch carries the marker:

| Season | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|---|---|---|
| Marked | 0.873 | 0.877 | 0.868 | 0.000 | 0.929 | 0.926 | 0.933 | 0.939 | 0.938 | 0.940 |

The 2019 files carry no markers, so 2019 is left out of every window. In every season, 93% to 94% of these steals happen on a pitch; the rest happen on a pickoff throw or a no-pitch.

**Reconciliation with the steal attempt table** (`data/derived/steal_attempts_2023_2025.parquet`, 2023 to 2025). The attempt table has 10,374 main attempts. Removing those with runners on first and third (1,711), the strike-three attempts (896) and the steal pitches with no marker (10) leaves 7,757. The marker table has 7,764 marked balls and strikes followed by a steal, a caught stealing, or a caught stealing negated by an error. The difference of 7 is pitches followed by a catcher's pickoff throw on which the runner was caught. The attempt table filed these as pickoff caught stealing or not on a pitch; here a catcher's throw belongs to the pitch before it. The marker table also has 2,151 marked ball-four pitches, which the steal record never counts as attempts.

**Swing rate.** Batters swing less when the runner goes, except at 1-2 and 3-2, where the rates match. Swing rates from 2023 to 2025 with a runner on first and second open:

| Count | 0-0 | 1-0 | 2-0 | 0-1 | 1-1 | 1-2 | 3-2 |
|---|---|---|---|---|---|---|---|
| Runner going | 0.225 | 0.283 | 0.238 | 0.431 | 0.484 | 0.589 | 0.710 |
| All pitches | 0.340 | 0.433 | 0.420 | 0.506 | 0.547 | 0.589 | 0.708 |

So what the pitch does differs when the runner goes, and the decision needs its own pitch-result table.

**Pitch-result table** (`pitch_result_going_v1.csv`). For each count, the share of marked pitches (runner on first only) by result.
- Seasons 2023 to 2025, except 3-0. That count has only 53 marked pitches in those seasons, so it comes from the wider window (152).
- 3-2 with 2 outs is excluded (the runner goes automatically); 3-2 uses 0 and 1 out.
- Plate appearances flagged excluded (bunts and intentional walks, 77 marked pitches) are removed.

| Count | Ball (ball four at 3-2) | Strike (strike three at 3-2) | Foul | In play | Hit by pitch | Pitches |
|---|---|---|---|---|---|---|
| 0-0, runner going | 0.454 | 0.379 | 0.094 | 0.072 | 0.000 | 3,107 |
| 0-0, runner not going (for comparison) | 0.380 | 0.369 | 0.128 | 0.120 | 0.003 | 108,733 |
| 3-2, runner going | 0.236 | 0.181 | 0.292 | 0.291 | 0.000 | 2,753 |

**Advancement with the runner going** (`advancement_going_v1.csv`). For plate appearances ending on a marked pitch put in play, the table gives the distribution of base state, outs and runs after each in-play category. Runner on first only, innings 1 to 8, bunts and intentional walks excluded: 4,166 plate appearances from 2023 to 2025, and 11,556 in the wider window. Each cell (category and outs) takes the first of these that applies:
1. Data from 2023 to 2025, if the cell has at least 30 plate appearances.
2. Otherwise, data from the wider window, if at least 30.
3. Otherwise, the same category pooled across outs in the wider window, if at least 30.
4. Otherwise, a rule for HR (everyone scores) or OTHER (catcher's interference: batter to first, runner to second).
5. Otherwise, the standard table (`advancement_v1`, all plate appearances, 2023 to 2025).

Of the 30 cells, 21 come from data, 3 are pooled, 3 use the rule (OTHER at every out count) and 3 use the standard table (FC at every out count). The runner going changes advancement a lot. All going values below are 2023 to 2025 data cells; `advancement_v1` covers all plate appearances, runner going or not.

| Measure | Outs | Runner going | advancement_v1 |
|---|---|---|---|
| Outfield single: runner to third or home | 0 / 1 / 2 | 0.818 / 0.847 / 0.811 | 0.291 / 0.318 / 0.407 |
| Double: runner scores | 0 / 1 / 2 | 0.627 / 0.610 / 0.770 | 0.290 / 0.327 / 0.502 |
| Ground-ball out: double play | 0 / 1 | 0.130 / 0.050 | 0.492 / 0.486 |
| Air out: runner doubled off | 0 / 1 | 0.185 / 0.187 | 0.029 / 0.031 |

**In-play outcomes by count** (`inplay_outcomes_by_count_v1.csv`).
- **Population.** The mix by count comes from all plate appearances ending on a ball in play with a runner on first and second open, runner going or not: 72,968 plate appearances from 2023 to 2025, innings 1 to 8.
- **Exclusions.** Catcher's interference (259 plate appearances recorded with an in-play pitch code) is dropped.
- **Why the runner-on-first version.** The steal decision exists only in those states, so that version is saved. Compared with the version over all base-out states, it differs by more than 0.01 in 19 cells (counts with at least 500 plate appearances). Most of these are more outfield singles (0.012 to 0.018) and fewer ground-ball outs (0.010 to 0.019).
- **Check against going balls in play.** Over the 4,150 balls in play on going pitches, the overall mix is close: outfield singles 0.197 against 0.189, ground-ball outs 0.303 against 0.295.

**Going FC share** (`fc_share_going_v1.csv`). Fielder's choices as a share of ground balls (GB_OUT plus FC) among marked in-play endings, runner on first only, wider window: 0.016, 0.009 and 0.002 at 0, 1 and 2 outs (11 of 704, 8 of 925, 4 of 1,829). The standard `fc_share_v1` has 0.035, 0.032 and 0.006. This is consistent with the lead runner rarely being retired when already running.

### Method: success model

**Fit population.** A throw situation is a marked pitch with result ball, strike or strike three, runner on first only, followed by a steal of second, a caught stealing, or a caught stealing negated by an error (counted as safe).
- **Size.** 8,649 throw situations from 2023 to 2025, with a safe rate of 0.797.
- **Left out.** A runner caught on the catcher's throw to first after the pitch (7). Ball four, fouls and balls in play, where no throw is made. Defensive indifference and other advances with no throw.
- **Consequence.** The model predicts success given a throw. Counting the 575 cases of defensive indifference on the same pitches as successes would raise the base rate from 0.797 to 0.810.

**Attributes.** Season aggregates by player, 2023 to 2025:

| Attribute | Source | Column |
|---|---|---|
| Runner sprint speed (ft/s) | Savant Sprint Speed leaderboard | `sprint_speed` |
| Catcher pop time to second (s) | Savant Catcher Pop Time leaderboard (data embedded in the page) | `pop_2b_sba` |
| Pitcher primary lead allowed (ft) | Savant Pitcher Running Game leaderboard, steals of second | `r_primary_lead` |
| Pitcher lead gained (ft) | Savant Pitcher Running Game leaderboard, steals of second | `r_sec_minus_prim_lead` |
| Runner aggressiveness | Retrosheet pitch sequences | See below |
| Candidates not used: runner lead gained on attempts and on opportunities | Savant Basestealing Run Value leaderboard, steals of second | `r_sec_minus_prim_lead_sbx`, `r_sec_minus_prim_lead` |
| Candidates not used: catcher exchange and arm strength | Savant Catcher Pop Time page | `exchange_2b_sba`, `maxeff_arm_2b_sba` |

- **Linking IDs.** Savant uses MLBAM IDs. The Chadwick register maps them to Retrosheet IDs: 24,622 people have both, with no ambiguous links, and every leaderboard row mapped.
- **Season check.** Every leaderboard file with a season column passed a check that it holds the requested season.
- **Coverage.** The attribute table has 4,327 player-seasons. Among throw situations, at most 0.3% of rows are missing any one attribute (sprint speed).

**Runner aggressiveness.** The share of thrown pitches on which the runner goes, with the runner on first and second open, by runner-season.
- **Population.** 454,040 pitches from 2023 to 2025. 3-2 with 2 outs is removed (7,377 pitches, 7,338 of them marked), because the runner goes automatically.
- **League rates.** 0.0413, 0.0457 and 0.0425 by season.
- **Shrinkage.** Each runner's rate is pulled toward the season's league rate: (going + m x league rate) / (pitches + m). The prior strength m = 15.0 pitches is estimated by method of moments across 1,924 runner-seasons, as for the base-out state ratios.

**Covariates and constraints.** A logistic regression for safe, with these terms:

| Term | Levels or form |
|---|---|
| Pitch result | Ball (reference), strike, strike three |
| 3-2 count | Indicator. Every 3-2 throw situation is a strike-three throw (490 of 490), so this is the 3-2 strike-three effect on top of strike three. |
| Pickoff throws | Pitcher throws to first earlier in the plate appearance: 0 (reference), 1, 2 or more |
| Pitchout | Indicator. The engine always evaluates with no pitchout. |
| Pitcher hand, batter side | Right-handed is the reference; switch hitters bat opposite the pitcher |
| Season | 2023 (reference), 2024, 2025 |
| Sprint speed, pop time, pitcher primary lead, pitcher lead gained | Slopes constrained to be non-negative, written as the exponential of a free parameter |
| Runner aggressiveness | Slope unconstrained |

During fitting, continuous attributes are centered on the fit-table mean and scaled by its standard deviation. Coefficients are reported per unit.

**Fitting.**
- **Estimation.** Penalized maximum likelihood (L-BFGS-B), with a ridge of 0.0001 on the fixed effects other than the intercept for numerical stability.
- **Player effects.** The specifications with player effects (B32, B-arm32) add one ridge-penalized effect per runner, catcher and pitcher.
- **Penalty search.** The three penalties are chosen by five-fold cross-validation, with folds split by game (3,243 training games). The grid has eight values from 1 to 3,162. The search starts from the best common value, then moves one penalty at a time. Chosen: runner 10, catcher 31.6, pitcher 31.6 (29 settings evaluated).
- **Selection rule.** A specification with player effects replaces A32 only if it beats A32 by more than 0.002 in 2025 log loss.
- **Benchmark.** LightGBM (C) is a benchmark only. It uses the same folds and monotone directions, with num_leaves 4, 8 or 16 and min_data_in_leaf 50, 100 or 200. The chosen setting is 16 and 100, with 45 rounds.

### Validation: success model (trained on 2023 to 2024, tested on 2025)

The test set is 2,933 throw situations.

| Model | What it is | Log loss | Brier | AUC |
|---|---|---|---|---|
| A32 | The fixed effects above | 0.50081 | 0.16172 | 0.624 |
| B32 | A32 plus runner, catcher and pitcher effects | 0.50105 | 0.16210 | 0.627 |
| B-arm32 | B32 with catcher arm strength (non-positive slope) and exchange (non-negative) in place of pop time | 0.50204 | 0.16212 | 0.619 |
| C | LightGBM | 0.51167 | 0.16573 | 0.593 |
| Constant | Training mean, 0.8023 | 0.51924 | 0.16813 | |

Calibration of A32 by decile of predicted success (about 293 throws each):

| Decile | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Mean predicted | 0.615 | 0.718 | 0.754 | 0.778 | 0.797 | 0.815 | 0.832 | 0.850 | 0.871 | 0.907 |
| Observed | 0.643 | 0.666 | 0.761 | 0.765 | 0.844 | 0.802 | 0.788 | 0.829 | 0.867 | 0.901 |

**Lagged-attribute check.** A season aggregate includes the attempts being predicted. To test for leakage, each attribute was replaced by the player's previous-season value (the previous season's league mean when missing). Each version was trained on 2024 only and tested on 2025:

| A32 | 2025 log loss |
|---|---|
| Same-season attributes, trained on 2023 to 2024 | 0.50081 |
| Same-season attributes, trained on 2024 | 0.50161 |
| All attributes lagged, trained on 2024 | 0.50511 |

Change in 2025 log loss when one attribute at a time is lagged: sprint speed +0.00008, pop time +0.00222, pitcher primary lead +0.00105, pitcher lead gained -0.00096, aggressiveness +0.00067.

A previous-season value is also a noisier measure of current skill, so a small positive gap is expected without any leak. Only pop time is above the 0.002 threshold, and only just. It is kept, and listed under "Known limits". The attempts version of runner lead gained failed the same kind of check clearly (see "Rejected candidates").

### Final model (A32, fitted on 2023 to 2025)

Logit scale, per unit. Continuous terms are centered on the fit-table mean shown. Standard errors come from a game bootstrap (200 replicates).

| Term | Estimate | SE | Center |
|---|---|---|---|
| Intercept | 1.534 | 0.077 | |
| Pitch result: strike | -0.167 | 0.065 | |
| Pitch result: strike three | 0.057 | 0.129 | |
| 3-2 count | -1.054 | 0.145 | |
| Pickoff throws: 1 | -0.075 | 0.062 | |
| Pickoff throws: 2 or more | -0.056 | 0.133 | |
| Pitchout | -0.578 | 0.367 | |
| Pitcher left-handed | 0.914 | 0.129 | |
| Batter bats left | -0.186 | 0.059 | |
| Season 2024 | -0.132 | 0.073 | |
| Season 2025 | -0.101 | 0.074 | |
| Sprint speed (per ft/s, non-negative) | 0.055 | 0.027 | 28.226 |
| Pop time (per s, non-negative) | 5.796 | 0.567 | 1.960 |
| Pitcher primary lead (per ft, non-negative) | 0.176 | 0.076 | 10.366 |
| Pitcher lead gained (per ft, non-negative) | 0.345 | 0.046 | 3.820 |
| Runner aggressiveness (per unit share) | 3.492 | 0.793 | 0.082 |

In-sample fit on all three seasons: log loss 0.48398, Brier 0.15488, AUC 0.640. Every constrained slope is inside its bound.

Average change in predicted success over the 8,603 throw situations without a pitchout:
- pop time 0.1 s slower: +7.6 points
- lead gained 1 ft larger: +4.8 points
- aggressiveness one standard deviation (0.045) higher: +2.3 points
- sprint speed 1 ft/s faster: +0.8 points

Findings:
- **The 3-2 strike-three throw is different.** The 3-2 term is -1.054 (SE 0.145), and the strike-three term on its own is about zero (0.057, SE 0.129). Observed, 490 strike-three throws at 3-2 were safe 0.584 of the time, against 0.826 for 402 at 0-2, 1-2 and 2-2. The likely cause is selection. On a full count with 0 or 1 out, the runner goes on 25.9% of pitches (against 2.8% at 0-0), because ball four forces the runner and a foul or a ball in play erases the attempt. The runner goes because of the count, not because the pitch was a good one to run on, and a strike three turns into a strike-'em-out, throw-'em-out play.
- **Player effects add nothing beyond the measured attributes.** B32 scored 0.50105 on 2025 against 0.50081 for A32. Its player effects were small: standard deviations of 0.105 (runners), 0.072 (catchers) and 0.031 (pitchers) on the logit scale. A32 is the production model.
- **Pop time is in line with MLB's published figure.** The coefficient is 5.80 per second (SE 0.57), so a pop 0.1 s slower raises predicted success by 7.6 points on average over the fit table. MLB's published figure is about ten points per 0.1 s. That figure is a raw contrast across catchers, while ours is the partial effect with the runner, the pitcher and the pitch result held fixed, so ours is expected to be smaller.

**Rejected candidates.**
- **Runner lead gained on attempts: it leaks.** It is a season average over the runner's own attempts, so it contains the outcomes being predicted. With 2024 values in place of same-season ones, its 2025 log loss was 0.5050 against 0.5002 without it.
- **Runner lead gained on opportunities:** indistinguishable from zero (0.030 per ft, SE 0.030 on all three seasons). It stays on the fit table but in no model.
- **Catcher arm strength: too thin.** It rests on a median of 3 max-effort throws per catcher-season (at most 8), and B-arm32 lost to A32 by 0.00123.
- **Missing-value indicators:** each covered 0 to 15 rows, mostly all safe, so their coefficients separated. Missing attributes use the fill values instead.

**Fill values.**
- A missing runner or catcher attribute takes the season mean over player-seasons.
- A missing pitcher primary lead or lead gained takes the season mean over pitcher-seasons of the same hand. A pitcher-season's hand is the one used most often in the Retrosheet plays. Every pitcher-season had a hand.
- A season outside 2023 to 2025 uses the 2025 season effect and fill values.

| Season | Sprint speed | Pop time | Aggressiveness | Primary lead L / R | Lead gained L / R | Pitcher-seasons L / R |
|---|---|---|---|---|---|---|
| 2023 | 27.248 | 1.971 | 0.043 | 10.208 / 10.506 | 1.720 / 3.874 | 217 / 625 |
| 2024 | 27.284 | 1.963 | 0.048 | 10.251 / 10.444 | 1.790 / 3.938 | 219 / 618 |
| 2025 | 27.320 | 1.954 | 0.046 | 10.404 / 10.592 | 1.649 / 3.850 | 212 / 649 |

### Engine

- **Modules.**
  - `engine/src/skipboard/steal_model.py`: `load_steal_model`, `StealSuccessModel.p_safe`, and the `RunnerProfile`, `CatcherProfile` and `PitcherProfile` inputs. The loader checks the terms, factor levels, constraint signs, and that pitcher fill values are given by hand. It raises `StealModelError` if any check fails.
  - `engine/src/skipboard/steal_tables.py`: `load_steal_tables` reads the four tables in `models/steal/`.
  - `engine/src/skipboard/steal_decision.py`: `steal_available` and `steal_decision`, which return the hold and run values, branches, effective throw success and break-even rates. Any object with a `p_safe` method can stand in for the success model.
- **Model directory.** The success model is read from the directory passed to `load_steal_model`; otherwise from `SKIPBOARD_STEAL_MODEL_DIR`; otherwise from `data/derived/steal/`.
- **Synthetic fixture.** `engine/tests/fixtures/steal_model/` holds round values with the structure of the exports, not estimated from any data. The unit tests run on it, so they need no Savant-derived file.
- **Parity.** A parity test recomputes the research model's prediction from the coefficients file for every throw situation without a pitchout (8,603). It compares those with the engine's `p_safe`, and the largest difference must be below 1e-10. The test is skipped when the research exports are not present.
- **Tests.** 71 tests cover the steal modules: 33 for the success model, 7 for the tables and 31 for the decision. The full engine suite has 184 tests, all passing.

### Known limits

- **Attributes are season aggregates.** A player has one value for every attempt in a season, and changes within the season are not seen. Pop time is the one attribute above the lagged-check threshold (+0.00222 against 0.002).
- **The speed slope is weakened by selection.** Slow runners attempt only when the jump or the pitch favors them, so among attempts speed adds little: +0.8 points per ft/s. Aggressiveness (+2.3 points per standard deviation) acts as the skill proxy, and it carries part of the speed effect (its correlation with sprint speed is 0.57). It describes runners who choose to go often, not what happens when a runner is told to go more.
- **Pitch-result shares are league-wide by count.** Every batter gets the same shares of fouls and balls in play on a going pitch. Only the in-play outcome mix depends on the batter.
- **Hit-and-run plays are among the going pitches.** The marker does not separate a straight steal from a hit-and-run, so the pitch-result shares and the advancement table include both.
- **Pickoff throws, not disengagements.** The model counts pitcher pickoff throws to first earlier in the plate appearance; both coefficients are small (-0.075 and -0.056). Step-offs and other disengagements count toward the disengagement limit but are not used.
- **No pitch location or type.** Only the result of the pitch is known.
- **FC advancement comes from the standard table.** With the runner going, fielder's choices are too rare (1, 5 and 1 plate appearances at 0, 1 and 2 outs from 2023 to 2025) to give their own advancement.
- **First and third is not covered.** 1,711 attempts from 2023 to 2025 came with runners on first and third, and their in-play advancement cells are thin (7 of 30 with at least 30 plate appearances). The decision is offered with a runner on first only.
- **The hand and lead-gained terms mean something only together.** The left-handed term (+0.914) offsets left-handers' smaller lead gained: 1.649 ft against 3.850 ft at the 2025 fill values, which is -0.76 on the logit scale at 0.345 per ft. Their primary lead is also slightly shorter: 10.404 ft against 10.592 ft at 0.176 per ft, about -0.03. At the 2025 fill values the net difference is +0.12 (0.802 against 0.782 for a league-average runner and catcher on a 0-0 ball). Neither term alone describes left-handers.
- **Same-season look-ahead.** An attempt early in a season uses attributes measured over that whole season, including later games. At game time only the season to date, or the previous season, is available.
- **No on-deck batter.** States that end the plate appearance get table values, as if the next batter were league average at 0-0.
- **The 3-2 residual in 2025.** With the 3-2 term fitted on 2023 to 2024, 3-2 throws in 2025 were safe 0.500 of the time against 0.610 predicted (172 throws). The final fit includes 2025, which moves the term to -1.054.
