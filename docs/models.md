# Models

Three models score game situations: a run expectancy table (re24_v1), a win probability model (wp v1), and a plate-appearance model (pa v1). All are built from Retrosheet regular-season play-by-play data (see `docs/retrosheet_notice.md`). The first two are league-average models: they describe a typical team, not the teams on the field. The plate-appearance model adds the batter and pitcher.

## Valuation principle

Every option in a decision is valued through the same path. A fan is scored on the difference between options, so a model gap that affects every option equally cancels in the comparison.

Decision values are anchored. The level comes from the empirical tables, and the plate-appearance model adds only the player and count effects:

- `value_wp(state, batter, pitcher)` = `wp(state)` + `continue_pa_wp(state, batter, pitcher)` - `continue_pa_wp(state at 0-0, league-average batter and pitcher with the same handedness)`.
- `value_runs` is the same with `run_expectancy` and `continue_pa_runs`.
- Both engine terms use the same platoon and base-out state settings. With league-average players at 0-0 the two engine terms are identical, so the value equals the table exactly.
- Decision modules use `value_wp` and `value_runs` (`engine/src/skipboard/valuation.py`). `continue_pa_wp` and `continue_pa_runs` stay available as the raw model.

Why anchor: the raw plate-appearance model differs from the tables by up to 0.051 runs in a single state (see "Consistency checks and tolerances"). Those gaps are not equal across the states a decision compares, so they do not fully cancel. The raw model's steal break-even rate differed from the tables' by up to 3 percentage points, which is enough to flip a verdict on an attempt near break-even. Anchored, league-average players reproduce the tables' break-even exactly.

Example: the steal. Each option is valued with `value_wp` (or `value_runs`) at the state it leads to, with the same batter, pitcher and count:

| Option | State valued |
|---|---|
| Hold | Current state, runner stays |
| Steal succeeds | Runner on the next base, same count and outs |
| Steal fails | Runner out and base cleared, one more out (if that is the third out, the half-inning ends and the value is the next half-inning's) |

The value of attempting is the success probability times the success value plus the failure probability times the failure value. That is compared with the value of holding.

Player information moves the answer. For five real 2025 matchups with a runner on first, 0 outs and a 0-0 count, the anchored break-even ranged from 3.5 percentage points below the league table (weak hitters) to 6.9 points above it (a strong power hitter), in runs and in win probability for the bottom of the 7th tied (`research/steal_break_even_examples.py`). Against a strong hitter, an out on the bases costs more, so an attempt needs a higher success rate.

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
