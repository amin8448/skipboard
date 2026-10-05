# Skipboard: Product Scope

Version: 2 (replaces version 1, "Be the Manager")
Status: draft for second review
Owner: Amin Nabavi
Last updated: 2026-10-05

| | |
|---|---|
| Product and company name | Skipboard |
| Default segment name | Skipper's Call (pending name checks; clubs may rename) |
| Domain | skipboard.app |
| Conflict-of-interest disclosure filed | [date to be recorded] |

---

## 0. Changes from version 1

- Renamed from "Be the Manager", which is in use by an established soccer management game.
- Replay is the product. Live mode is removed; a narrow live-state option is listed as a later possibility (section 3).
- The pilot runs on precomputed segments and a managed vote service. The engine runs offline (section 9).
- Win probability is the single scoring currency; runs are used only to explain (section 6).
- A shared batter plate-appearance model values "swing away" and "hold" with the actual hitter and pitcher (section 7).
- The 2025 World Series Game 7 bunt is no longer the flagship demo scenario. The demo leads with steal and send situations where player-specific inputs change the league-average answer.
- Positioning is rewritten around the in-stadium buyer (section 2).
- Go-to-market lists three routes; the conversation with the Toronto contact decides between them (section 11).
- The segment is cut to four beats and designed around a real break (section 3).
- Data rights are corrected and the minor league data gap is stated plainly (section 10).
- The conflict-of-interest disclosure is filed before any demo leaves (section 12, Phase 0).

---

## 1. Product summary

### What it is

Skipboard is an in-stadium game in which fans make a manager's call about a real game situation on their phones. Between innings, the big screen presents a situation from a past game and asks a question such as "Send him?" or "Run on this pitch?". Fans vote. The big screen shows how the crowd split, what the numbers say, and what actually happened.

Fans are scored on decision quality, not on outcome. Each option is valued by win probability using models built for the specific players involved. A sound call that fails still scores well; a poor call that works does not.

### Who pays

Clubs, through the departments that own the in-stadium experience (ballpark entertainment, marketing, partnerships). Which clubs, and through which channel, is decided by the go-to-market work in section 11.

### Revenue model

- Priced from the sponsor's side. The segment is sponsor inventory ("Skipper's Call, presented by [sponsor]"), and the product's price must fit comfortably inside what that sponsorship sells for.
- Two pricing forms to test with the pilot club: a share of the segment's sponsorship revenue, or a low flat seasonal fee.
- Optional, club-controlled fan opt-in (for example, email for a sponsor offer). This is what in-stadium sponsors typically value most, and it brings privacy and contest law into scope (section 12).
- Expected reach per segment at a typical minor league game is in the hundreds to low thousands of voters. Pricing must be realistic about that.

### Current stage

Foundations and demo. The demo is a private, non-commercial research prototype built on public data (section 10).

---

## 2. Positioning

### What a ballpark buyer will compare it with

- **SQWAD:** QR-code videoboard games (predictions, live polls) sold to clubs as sponsor activations.
- **Fan Controlled Sports, Control app:** fans voted on in-game strategic decisions for the Oakland Ballers (Pioneer League) in 2024.
- **MLB Ballpark app:** supports in-stadium voting, trivia, and games in select markets.
- **Tally:** free-to-play live predictions sold to teams.
- **History:** Bill Veeck's Grandstand Managers Night (1951), where fans voted on the Browns' decisions with placards. The format is 75 years old.

### What an analyst will compare it with

- **Game Strategy Explorer (Baseball Savant, April 2026):** win probability and run expectancy for any situation from 2016 to 2025, with steals, intentional walks, and bunts. It deliberately does not identify the hitter or pitcher. MLB launched it under the headline "You play the manager".
- **Savant Games:** guessing games only (pitch type, player chart, prospect spray chart, team logo, player photo). None is a crowd decision game.
- **Statcast Basestealing Run Value and Extra Bases Taken leaderboards:** per-opportunity success probabilities for steals and advancement. These are the benchmarks for Skipboard's steal and send models.

### What sets Skipboard apart

The claim to defend: the only videoboard segment whose answer is grounded in player-level analytics and explained in one line.

1. Player-specific values: this runner against this catcher and pitcher, this hitter at the plate. Every demo scenario shows the league-average baseline next to Skipboard's number.
2. Scoring on decision quality, with the gap between a good call and a good result made visible.
3. Built for the big screen and a casual crowd, not for a solo lookup.

### What is not defensible

The format is old and easy to copy. Defensibility comes from model quality, curated segment content, operator tooling that makes the segment easy to run, and club relationships.

Wagering comparisons are kept out of all buyer-facing material.

---

## 3. Users and delivery

### Users

- **Fan:** joins once, early (pregame or first inning), by QR code in the phone browser. The session stays open all night. No app install, no account.
- **Big-screen audience:** everyone in the stadium, including people who never vote. The segment must work for them.
- **Operator:** club control-room staff. Starts segments, skips them, or switches to the fallback slate.
- **Club partnerships staff:** sponsor branding and participation reports.

### Segment flow: four beats

1. **Situation:** one line and the two or three numbers that matter (for example, runner speed and catcher pop time).
2. **Vote:** 15 to 20 seconds.
3. **Crowd versus the computer:** the crowd split and a one-line verdict with one visual, for example "Safe 71% of the time. Worth the risk."
4. **What happened:** the actual result and, where enabled, the real manager's call.

The leaderboard appears at the end of the night, not after every segment. Total segment length is fixed in agreement with the club, within the break between innings.

### In-venue requirements

- Big-screen output as a 1080p video signal from a dedicated machine, in whatever form the control room routes (HDMI, SDI, or NDI), or as a pre-rendered package.
- A scripted PA read for each segment.
- A one-button skip or fallback slate for when a break is cut short or the network fails.
- Rehearsal in the venue before any live use.
- Accessibility: a screen-reader-friendly phone page, colour-safe vote splits, and captioned big-screen text.

### Delivery modes

| Mode | Situation source | When |
|---|---|---|
| Demo | Curated past situations, precomputed | Now |
| Replay | Curated past situations, precomputed, with live voting | Pilot |

Live voting on the current game is removed. The send decision happens while the ball is in play, and steal and bunt decisions arise between pitches inside an 18-second pitch timer, leaving no time to present, vote, and close.

**Possible later option, not planned:** a live-state hypothetical during a mid-inning pitching change with a runner on first ("Should he run on the new pitcher?"), closed before the first pitch. Never offered for the send decision.

---

## 4. Decision catalog

### Decision module contract

1. Availability rule: which game states allow the decision.
2. Options offered to fans.
3. Outcome probabilities for each option, given the players involved.
4. State transitions for each outcome.
5. Value of each option in win probability (section 6).

### Version 1

| Decision | Situation | Options | Notes |
|---|---|---|---|
| Steal second | Runner on first, second open | Run on this pitch, not this pitch | Valued per pitch (section 7) |
| Send the runner home | Runner on second, single to the outfield | Send, hold at third | Shown on screen as the third-base coach's call |
| Sacrifice bunt | Runner on first, runner on second, or both; nobody out | Bunt, swing away | Swing away valued with the actual hitter and pitcher |

### Later versions

Pinch runner, pinch hitter, steal of third, double steal, tag-up on a fly ball, hit-and-run, squeeze, intentional walk, pitching change. None is designed in detail until a pilot shows fans use the product.

### Out of scope, all versions

- Defensive positioning and alignment.

---

## 5. Game state

### Principles

- The state describes the situation. Player abilities live in a separate player profile table.
- States are immutable; outcomes produce new states.
- Only fields that change a version 1 answer are designed now. Fields for later decisions are added when those decisions are built.

### Version 1 fields

| Group | Fields |
|---|---|
| Context | Level, season, game type (regular season or postseason), park |
| Inning and score | Inning, half, home and away runs |
| Count | Outs, balls, strikes |
| Bases | Runner player ID on first, second, third |
| Disengagements | Used since the last runner advance; limit for this level and season |
| Batter | Batter ID and handedness; on-deck batter ID |
| Pitcher | Pitcher ID and handedness, days of rest, pitch count |
| Fielders | Catcher ID; outfielders by position |

### Rules that the fields encode

- The disengagement count resets when a runner advances during the plate appearance. The limit is a per-level, per-season value, because minor league experimental rules differ from MLB's.
- Game type sets the extra-inning rule: the automatic runner applies in the regular season, not in the postseason.

### Deferred fields

Bench, bullpen, players already used, mound visits remaining, full lineup. Added with the decisions that need them.

---

## 6. Valuation

### One currency

Every decision is scored in win probability. Runs appear only as explanation on screen ("costs 0.15 runs, which is 1.1% win probability here"). One currency means one points curve and no threshold to defend.

### Run expectancy

- Base-out table (24 states) from Retrosheet play-by-play, 2023 to 2025, the seasons under current rules. Three seasons is ample for 24 states.
- Used for the on-screen explanation and as a cross-check.

### Win probability

- Built from a transition model rather than averaging raw outcomes per cell: plate-appearance and count transition probabilities estimated on Retrosheet 2016 to 2025, with an era adjustment or reweighting toward 2023 onward.
- Win probability for any state is derived from the transition model by Markov chain or simulation.
- Count-level values come from the same model, so a count-level table is not a separate milestone.
- Cross-checked against the Game Strategy Explorer's baseline.

---

## 7. Probability models

### Common rules

- Version 0 models are hand-set or league-average and labelled as such wherever shown.
- Each model is exported as a readable coefficient file with a version number.
- Until a club relies on the numbers, each model is documented in a short table (inputs, data, seasons, validation result, known limits). Full model cards come later.
- Validation for fitted versions: calibration on a held-out season, and log loss compared with a league-average baseline.

### Shared component: batter plate-appearance model

Used by the steal "hold" option and the bunt "swing away" option.

- Outcome distribution for the batter: single, double, triple, home run, walk or hit by pitch, strikeout, other out, and ground-ball double play rate.
- Batter rates shrunk toward league average; adjusted for the pitcher by the log5 (odds-ratio) method.
- One plate-appearance transition applied to the current state; resulting states valued by win probability.
- Data from Retrosheet, which permits commercial use (section 10).

### Steal second

| | |
|---|---|
| Options | Run on this pitch, or not on this pitch |
| Outcomes if running | Safe, caught stealing (throwing errors folded into a league-average rate) |
| Inputs | Runner sprint speed and lead, catcher pop time, pitcher time to plate, disengagements used and limit, count |
| Version 0 success model | Logistic curve; intercept at the league success rate on steals of second; runner slope fitted on 2026 runner data; catcher slope hand-set |
| Value of "not this pitch" | Continue the plate appearance with the batter model |

Known limits:
- Valuing "not this pitch" by continuing the plate appearance ignores the option to run later in the same plate appearance. This undervalues holding and biases the verdict toward running. Stated on the model table.
- League success rates sit around 79 to 80 percent, well above the break-even rate of about 70 percent implied by Statcast's run values, so most real attempts are clearly correct. Demo scenarios are chosen near break-even (two outs, late and close, strong catcher, slower runner), where the call is genuinely difficult.
- The data contains only attempts, and runners choose when to go. Partial pooling and a monotone constraint on speed (faster is never worse) keep extrapolation sensible.
- Validated at runner level against Statcast's Basestealing Run Value expectations.

### Send the runner home

| | |
|---|---|
| Outcomes | Safe at home, out at home, trailing runner advances on the throw |
| Inputs | Runner sprint speed and position, outfielder arm strength and location at the catch, hit direction and depth, outs, on-deck hitter |
| Version 0 | Race model: runner time to home against ball time to home, with a hand-set spread on the margin |
| Version 1 | Same structure, spread and timing terms fitted on observed sends |

Known limit: held runners are never tested. The race structure extends to those cases through physics. Benchmarked against Statcast's Extra Bases Taken probabilities, with differences documented.

### Sacrifice bunt

| | |
|---|---|
| Outcomes of bunting | Successful sacrifice, bunt hit, batter safe with no out (error or fielder's choice), lead runner forced, double play, popped up; foul bunts leading to two strikes and a swing-away |
| Version 0 bunt outcomes | League-average rates for the base-out situation |
| Value of swing away | Batter plate-appearance model against the actual pitcher |
| Version 1 | Batter-specific bunt rates, shrunk toward league average |

---

## 8. Scoring and game mechanics

### Points

- The cost of a choice is the win probability it gives up against the best option.
- Points = round(100 x exp(-cost / s)), where s is set so that the median cost across the curated scenarios scores 50 (s = median cost / ln 2). The value of s is recorded with the model tables.
- Points never depend on what happened on the play or on how fast a fan answered.

### Close calls

- The uncertainty in the difference between options is estimated by bootstrap or posterior simulation.
- A winner is declared when the probability that the best option is truly better is at least 0.8. Otherwise the verdict is "too close to call" and every option in that range scores 100.
- Because scenarios are curated, most should have a clear answer. The share of "too close" verdicts is tracked as a content measure.

### The real manager's call

- Shown for historical replays and opponent replays, where "the crowd versus the real call" is the strongest moment of the segment.
- Off by default for the home club's current manager. A club setting.

### Structure

- A game night is three to six segments, set per club.
- Leaderboard at the end of the night.
- Season leaderboards and fan accounts are not planned until a pilot proves fans vote.

### Vote integrity

- A signed session token issued at QR join; one vote per token per segment.
- Rate limits by IP address and token.
- The leaderboard only counts tokens that joined before the segment opened.
- Any prize requires a verified account.

---

## 9. Architecture

### Repository layout

```
skipboard/
  docs/        scope, model tables, decision records, dated Retrosheet notice
  data/        raw and derived data (not in git; fetch scripts recreate it)
  research/    plain scripts: fetch, prepare, fit, validate
  models/      exported coefficient files, by model and version
  engine/      Python package: game state, profiles, valuation, decisions,
               scoring, scenario export; with tests
  scenarios/   curated scenario definitions and exported JSON
  web/         phone view, big-screen view, operator console
```

### Pilot path

1. The engine runs offline and exports every scenario as JSON: situation, values, verdict, points table, and outcome.
2. A static front end serves the phone view, the big-screen view, and the operator console.
3. A managed real-time service handles joins, votes, and live counts. Chosen after a burst test at the expected voter count; candidates not yet evaluated.
4. The big-screen view runs on a dedicated machine with a video output.

There is no runtime engine, API server, or self-hosted database in the pilot. A runtime API is added only if live-state hypotheticals are ever built.

### Demo path

Same exported JSON, displayed by a static page shared privately. No voting service.

### Engineering standards

- Python 3.12 with type hints; typed structures for game states and profiles.
- Tests for every decision module, transition, and scoring rule; tests run automatically on every push.
- Feature branches, no direct commits to main.
- The domain is under the .app top-level domain, which browsers only load over HTTPS. Every hosted page must have a valid certificate.

---

## 10. Data sources and rights

| Source | Contents | Terms | Use |
|---|---|---|---|
| Retrosheet | Play-by-play event files, 1898 to 2025 | Any use including commercial, provided Retrosheet's notice appears prominently | Valuation, batter plate-appearance model |
| Baseball Savant (Statcast) | Sprint speed, pop time, arm strength, basestealing | MLB terms permit only personal, non-commercial home use without written permission | Private research prototype only |
| MLB Stats API | Game feeds | MLB terms | Research only |
| Sportradar | Official MLB data, including Statcast | Reported exclusive distributor of MLB official data through 2032 | Licensed route; request a quote once a club shows interest |
| Club-supplied data | Club's tracking data | Assume MLB sign-off is needed for a vendor's commercial use | Ask the club, and get the answer in writing |

### Minor league data

Public minor league Statcast covers Triple-A (from 2023) and the Florida State League (from 2021) only, and it belongs to MLB. Double-A, High-A, and most Low-A clubs have no public sprint speed, pop time, or arm data. Player-specific segments at those clubs are not possible without a data agreement.

### The demo

- Labelled as a non-commercial research prototype.
- Shared privately by link; not publicly hosted, listed, or sold.
- No team logos, colours, or MLB marks.

### Retrosheet notice

Copy the notice verbatim from retrosheet.org/notice.txt on the download date, store it with the date in `docs/`, and display it on the demo page and in big-screen credits.

### Player names and likeness

The US fantasy-sports precedent on names and statistics does not settle a sponsored segment, and it does not apply in Canada. A legal opinion is obtained before the pilot. Until then, pilot segments use the host club's own players with the club's written consent, or historical replays, and sponsor branding is kept visually separate from player names and images.

---

## 11. Go-to-market

### Three routes

| Route | What it is | What it needs | Main obstacle |
|---|---|---|---|
| A. Triple-A through Toronto | Pilot with Toronto's Triple-A affiliate, the Buffalo Bisons, introduced through the Toronto contact | Club agreement, MLB approval for data use | MLB approval |
| B. Partner and summer collegiate leagues | Clubs outside MLB's affiliated system, running replays of memorable MLB moments | A club willing to buy; licensed player inputs for MLB replays | No local tracking data; player inputs still MLB-derived |
| C. License the engine and content | Supply the decision engine and curated segments to platforms that already run in-stadium voting and hold data rights | A platform partner | Smaller share of revenue; dependence on the partner |

The Toronto conversation is used to learn which route is real.

### What the Toronto conversation should answer

1. Does the analysis hold up to a club analyst?
2. Who owns in-stadium entertainment and sponsorship at Toronto and at Buffalo?
3. Can a club supply tracking inputs to an outside vendor, and what would MLB require?
4. Would the contact make an introduction?

### Two artifacts

- **For the analyst:** a short model note covering methods, validation, limits, and the comparison with the Game Strategy Explorer.
- **For a buyer:** a 60-second video of the segment as the crowd would see it, plus a phone vote mock-up.

### Pricing approach

Ask the pilot club what a recurring in-game feature sponsorship sells for, then price as a revenue share or a flat fee that fits clearly inside it.

### Timing

The 2026 season is over. A 2027 pilot has to be agreed during the off-season, before the club sells its 2027 sponsorship inventory.

---

## 12. Roadmap

Each phase ends at a gate.

### Phase 0: Foundations

- Conflict-of-interest disclosure filed. Nothing leaves for a club before this.
- Scope version 2 and second review.
- Repository set up; data fetched.
- Name checks for the segment name; decision on a Canadian trademark application for Skipboard.

### Phase 1: Demo

- Run expectancy (base-out) and the win probability transition model from Retrosheet.
- Batter plate-appearance model.
- Version 0 steal, send, and bunt models.
- Four to six curated Toronto scenarios near break-even, each showing the league-average baseline next to Skipboard's number.
- Model note and 60-second segment video.
- Gate: an introduction to a ballpark entertainment or partnerships lead (Toronto or Buffalo), or a clear answer on which route to pursue.

### Phase 2: Replay product

- Fitted version 1 models with model tables.
- Phone view, big-screen view with video output, operator console, managed vote service.
- Burst test at expected voter count.
- Test of the four-beat segment with casual fans.
- Gate: a full replay game night runs end to end in rehearsal.

### Phase 3: Pilot

- One club on the chosen route, with data and consent agreements in writing.
- Connectivity survey at the venue during a real game: load time, vote round-trip time, and failure rate on each major carrier.
- A named sponsor for the segment.
- Gate: at least 20 percent of attendance votes in at least one segment, and the club is willing to pay for the following season.

### Not planned until a pilot succeeds

- Live-state hypotheticals.
- Later decisions and the half-inning simulator.
- Season leaderboards and fan accounts.
- Broadcast, streaming, or consumer versions.

---

## 13. Risks and open questions

### Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Conflict of interest as a federal public servant | High | Disclosure filed before any demo leaves; all work off government equipment and time |
| Data rights for a commercial version | High | Commercial use only on licensed or club-supplied data with written permission |
| MLB control of minor league data and approvals | High | Route choice informed by the Toronto conversation |
| The Toronto contact is not a buyer | High | Two artifacts; gate is an introduction, not a sale |
| Model credibility with analysts | High | Baseline shown beside every number; stated limits; close-call rule |
| Segment too long for the break | High | Four beats, fixed length agreed with the club, fallback slate |
| Stadium connectivity | High | Venue survey; tiny vote request; page cached at join; split shown from whatever votes arrive |
| Key-person risk: the product depends on one person on game night | High | Runbook and fallback slate the club can run alone |
| Capacity: one developer | High | Strict phase gates; nothing from a later phase starts early |
| Season timing | Medium | Agree a 2027 pilot during the off-season |
| No sponsor for the pilot segment | Medium | Sponsor named as part of the pilot agreement |
| Player names in a sponsored segment | Medium | Legal opinion before the pilot; host club consent or historical replays |
| Prize contests and lead capture | Medium | Canada: Criminal Code section 206 (skill-testing question) and Competition Act disclosure. Quebec's separate regime was repealed in October 2023. A US pilot falls under that state's law. Rules review before any prize or sponsor offer |
| Vote integrity | Medium | Signed session tokens, rate limits, verified accounts for prizes |
| Privacy of fan data | Medium | Minimum collection; opt-in only under club control; privacy review per market |
| Liability and insurance for an in-venue activation | Medium | Ask the pilot club what vendor insurance it requires |
| Fans vote for fun, not for the best call | Medium | One-line verdicts; crowd-versus-computer framing; casual-fan testing |

### Open questions

1. Which go-to-market route (section 11).
2. Pricing and the club's sponsorship values.
3. Segment name checks for "Skipper's Call".
4. Canadian trademark application for Skipboard: when and in which classes.
5. Managed vote service: which one, after a burst test.
6. Whether club-supplied data can be used, and on what terms.
7. Scope of the legal review: data rights, player names, contest rules.
