from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Protocol

from skipboard.plate_appearance import MATCHUP_CATEGORIES, PATables, PlayerProfile, load_pa_tables, matchup_rates
from skipboard.state import GameState
from skipboard.steal_model import CatcherProfile, PitcherProfile, RunnerProfile, load_steal_model
from skipboard.steal_tables import GOING_BASE_CODE, StealTables, is_offered, load_steal_tables
from skipboard.valuation import value_runs, value_wp
from skipboard.value import runs_after, wp_after

BISECTION_STEPS = 100
# In-play category -> matchup category whose ratio to league scales it (GB_OUT and FC share the GB ratio).
INPLAY_RATIO = {"1B_IF": "1B", "1B_OF": "1B", "2B": "2B", "3B": "3B", "HR": "HR", "AIR_OUT": "AIR_OUT", "ROE": None}


PaEnd = Callable[[int, int, int], tuple[float, float]]


class SuccessModel(Protocol):
    def p_safe(
        self, pitch_result: str, balls: int, strikes: int, pickoff_throws: int, bat_side: str, season: int,
        runner: RunnerProfile, catcher: CatcherProfile, pitcher: PitcherProfile,
    ) -> float: ...


@dataclass(frozen=True)
class Outcome:
    label: str
    p: float  # probability within the branch
    value_wp: float
    value_runs: float


@dataclass(frozen=True)
class StealBranch:
    result: str  # pitch result on the pitch the runner goes on
    share: float
    p_safe: float | None  # throw branches only
    outcomes: tuple[Outcome, ...]  # safe/out for throws, in-play categories, or a single state

    @property
    def is_throw(self) -> bool:
        return self.p_safe is not None

    def value_at(self, p: float | None = None) -> tuple[float, float]:
        # Branch value in (win probability, runs); throw branches can be valued at any success probability.
        if self.is_throw and p is not None:
            safe, out = self.outcomes
            return p * safe.value_wp + (1 - p) * out.value_wp, p * safe.value_runs + (1 - p) * out.value_runs
        return sum(o.p * o.value_wp for o in self.outcomes), sum(o.p * o.value_runs for o in self.outcomes)

    @property
    def value_wp(self) -> float:
        return self.value_at()[0]

    @property
    def value_runs(self) -> float:
        return self.value_at()[1]


@dataclass(frozen=True)
class StealDecision:
    hold_wp: float  # "not this pitch"
    hold_runs: float
    run_wp: float  # "run on this pitch"
    run_runs: float
    branches: tuple[StealBranch, ...]
    effective_p_safe: float | None  # share-weighted success probability over the throw branches
    break_even_wp: float | None  # p on every throw that equalizes the options; None if no p in [0, 1] does
    break_even_runs: float | None

    def run_wp_at(self, p: float) -> float:
        return sum(b.share * b.value_at(p)[0] for b in self.branches)

    def run_runs_at(self, p: float) -> float:
        return sum(b.share * b.value_at(p)[1] for b in self.branches)


def steal_available(state: GameState) -> bool:
    # Runner on first, second and third empty, and not 3-2 with 2 outs (the runner goes automatically).
    return state.base_code == GOING_BASE_CODE and is_offered(state.balls, state.strikes, state.outs)


def steal_decision(
    state: GameState,
    batter: PlayerProfile | None = None,
    pitcher: PlayerProfile | None = None,
    *,
    runner: RunnerProfile,
    catcher: CatcherProfile,
    pitcher_hold: PitcherProfile,
    model: SuccessModel | None = None,
    tables: StealTables | None = None,
    pa_tables: PATables | None = None,
) -> StealDecision:
    if not steal_available(state):
        raise ValueError(f"the steal decision is not offered with base_code {state.base_code}, {state.balls}-{state.strikes}, {state.outs} outs")
    if pitcher_hold.hand != state.pitcher_hand:
        raise ValueError(f"pitcher profile hand {pitcher_hold.hand} does not match the state's pitcher hand {state.pitcher_hand}")
    m = model or load_steal_model()
    t = tables or load_steal_tables()
    pa_t = pa_tables or load_pa_tables()
    b, s, outs = state.balls, state.strikes, state.outs

    def same_batter(**changes: object) -> tuple[float, float]:
        nxt = replace(state, **changes)
        return value_wp(nxt, batter, pitcher), value_runs(nxt, batter, pitcher)

    def pa_end(base_post: int, outs_post: int, runs: int = 0) -> tuple[float, float]:
        # The plate appearance (or the half-inning) ends: valued as the existing valuation values it.
        return wp_after(state, base_post, outs_post, runs, new_pa=True), runs_after(state, base_post, min(outs_post, 3), runs)

    def caught(new_balls: int, new_strikes: int) -> tuple[float, float]:
        if outs + 1 == 3:
            return pa_end(0, 3)
        return same_batter(runner1=None, outs=outs + 1, balls=new_balls, strikes=new_strikes)

    def throw(result: str, p: float, safe: tuple[float, float], out: tuple[float, float]) -> tuple[float | None, tuple[Outcome, ...]]:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"success model returned {p} for {result}")
        return p, (Outcome("safe", p, *safe), Outcome("out", 1 - p, *out))

    def p_safe(result: str) -> float:
        return m.p_safe(result, b, s, state.pickoff_throws, state.batter_hand, state.season, runner, catcher, pitcher_hold)

    hold = (value_wp(state, batter, pitcher), value_runs(state, batter, pitcher))
    branches = []
    for result, share in t.pitch_result_shares(b, s, outs).items():
        p: float | None = None
        if result in ("ball", "strike"):
            nb, ns = (b + 1, s) if result == "ball" else (b, s + 1)
            p, outcomes = throw(result, p_safe(result), same_batter(runner1=None, runner2=state.runner1, balls=nb, strikes=ns), caught(nb, ns))
        elif result in ("ball_four", "hit_by_pitch"):
            outcomes = (Outcome("runner to second, batter to first", 1.0, *pa_end(3, outs)),)
        elif result == "strike_three":
            if outs == 2:
                outcomes = (Outcome("third out, no throw", 1.0, *pa_end(0, 3)),)
            else:
                out_end = pa_end(0, 3) if outs + 2 >= 3 else pa_end(0, outs + 2)
                p, outcomes = throw(result, p_safe(result), pa_end(2, outs + 1), out_end)
        elif result == "foul":
            value = same_batter(strikes=s + 1) if s < 2 else hold
            outcomes = (Outcome("foul", 1.0, *value),)
        elif result == "in_play":
            outcomes = _in_play(state, batter, pitcher, t, pa_t, pa_end)
        else:
            raise ValueError(f"unknown pitch result {result!r} in the pitch-result table")
        branches.append(StealBranch(result, share, p, outcomes))

    branches_t = tuple(branches)
    run = (sum(br.share * br.value_wp for br in branches_t), sum(br.share * br.value_runs for br in branches_t))
    throws = [br for br in branches_t if br.is_throw]
    weight = sum(br.share for br in throws)
    effective = sum(br.share * (br.p_safe or 0.0) for br in throws) / weight if weight > 0 else None
    decision = StealDecision(hold[0], hold[1], run[0], run[1], branches_t, effective, None, None)
    return replace(decision, break_even_wp=_break_even(lambda q: decision.run_wp_at(q) - hold[0], throws),
                   break_even_runs=_break_even(lambda q: decision.run_runs_at(q) - hold[1], throws))


def _in_play(
    state: GameState, batter: PlayerProfile | None, pitcher: PlayerProfile | None, t: StealTables, pa_t: PATables, pa_end: PaEnd
) -> tuple[Outcome, ...]:
    # League in-play mix at the count, scaled by the matchup's ratio to league (platoon on, no state ratio).
    rates = matchup_rates(state, batter, pitcher, pa_t, apply_platoon=True, apply_state=False)
    ratio = dict(zip(MATCHUP_CATEGORIES, (rates / pa_t.league9).tolist()))
    base = t.inplay_shares(state.balls, state.strikes)
    weights: dict[str, float] = {}
    for category, share in base.items():
        if category in ("GB_OUT", "FC"):
            continue
        if category not in INPLAY_RATIO:
            raise ValueError(f"unknown in-play category {category!r}")
        key = INPLAY_RATIO[category]
        weights[category] = share * (ratio[key] if key else 1.0)
    ground = (base.get("GB_OUT", 0.0) + base.get("FC", 0.0)) * ratio["GB"]
    fc = t.going_fc_share(state.outs)
    weights["GB_OUT"], weights["FC"] = ground * (1 - fc), ground * fc
    total = sum(weights.values())
    outcomes = []
    for category, w in weights.items():
        options = t.going_transitions(state.outs, category)
        values = [(tr.p, pa_end(tr.base_post, tr.outs_post, tr.runs)) for tr in options]
        outcomes.append(Outcome(category, w / total, sum(q * v[0] for q, v in values), sum(q * v[1] for q, v in values)))
    return tuple(outcomes)


def _break_even(gap: Callable[[float], float], throws: list[StealBranch]) -> float | None:
    # Bisection on [0, 1] for the p that makes the run option equal to holding.
    if not throws:
        return None
    lo, hi = 0.0, 1.0
    g_lo, g_hi = gap(lo), gap(hi)
    if g_lo == 0:
        return lo
    if g_hi == 0:
        return hi
    if (g_lo > 0) == (g_hi > 0):
        return None
    for _ in range(BISECTION_STEPS):
        mid = (lo + hi) / 2
        g_mid = gap(mid)
        if (g_mid > 0) == (g_lo > 0):
            lo, g_lo = mid, g_mid
        else:
            hi = mid
    return (lo + hi) / 2
