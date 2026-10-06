# Models

Two models score game situations: a run expectancy table (re24_v1) and a win probability model (wp v1). Both are built from Retrosheet regular-season play-by-play data (see `docs/retrosheet_notice.md`) and are league-average models: they describe a typical team, not the teams on the field.

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
| Top 10th, tied, runner on second, 0 outs | 0.495 |

### Known limits

- **Both teams are treated as league-average.** A team that is ahead is more often the better team, so leads hold more often than an average-team model expects. This explains the underconfidence seen in the 2025 calibration. Below 0.5, actual win rates fall short of predictions by up to 0.037. Above 0.5, actual win rates run ahead of predictions by up to 0.029.
- **Half-innings are treated as independent.** Each one starts fresh from the same run distributions, regardless of what came before or who is due up.
- **Extra-inning values are rough.** They reuse the innings 1 to 8 distribution for a runner on second with 0 outs, and ignore one-run strategy such as bunting the runner over. The 2025 check rests on only 209 games.
- **Run distributions come from innings 1 to 8.** Late-inning bullpen use and strategy in the ninth and later are not modeled.

Recalibration (fitting a correction to the predicted values) was deliberately not applied, because it would hide the cause of the gap rather than fix it. The planned fix is a team-strength adjustment. It will be judged by whether it closes the calibration gap on held-out data.
