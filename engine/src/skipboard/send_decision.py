from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Literal, Protocol

import numpy as np

from skipboard.plate_appearance import PlayerProfile
from skipboard.send_model import (
    BattedBall,
    FielderProfile,
    PositioningStart,
    SendContext,
    load_batter_advance,
    load_send_model,
)
from skipboard.state import BatterHand, GameState
from skipboard.steal_model import RunnerProfile
from skipboard.valuation import value_runs, value_wp
from skipboard.value import runs_after, wp_after

SEND_BASE_CODE = 2  # runner on second, first and third empty
OUTFIELD = (7, 8, 9)
CLOSE_CALL_THRESHOLD = 0.7
ON_DECK_ESTABLISHED_PA = 2000.0  # weighted PA; below it the on-deck batter counts as league average in the send equation
ADVANCES = ("first", "second", "out")
# Runner result -> (batter-advance table row, runs scored, outs added by the runner)
RESULTS = {"held": ("held at third", 0, 0), "safe": ("scored", 1, 0), "out": ("out at home", 0, 1)}

Advance = Literal["first", "second", "out"]
Result = Literal["held", "safe", "out"]


class SendSuccessModel(Protocol):
    def p_safe(self, ball: BattedBall, runner: RunnerProfile, fielder: FielderProfile, start: PositioningStart, context: SendContext, season: int) -> float: ...

    def p_send(self, ball: BattedBall, runner: RunnerProfile, fielder: FielderProfile, start: PositioningStart, context: SendContext, season: int) -> float: ...

    def p_safe_bootstrap(
        self, ball: BattedBall, runner: RunnerProfile, fielder: FielderProfile, start: PositioningStart, context: SendContext, season: int
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class SendOutcome:
    advance: Advance  # what the batter who singled did
    p: float  # share within the result, from the batter-advance table
    base_post: int
    outs_post: int  # 3 when the half-inning ends
    runs: int
    value_wp: float
    value_runs: float


@dataclass(frozen=True)
class SendBranch:
    result: Result  # held at third, sent and safe, sent and out
    outcomes: tuple[SendOutcome, ...]

    @property
    def value_wp(self) -> float:
        return sum(o.p * o.value_wp for o in self.outcomes)

    @property
    def value_runs(self) -> float:
        return sum(o.p * o.value_runs for o in self.outcomes)


@dataclass(frozen=True)
class SendDecision:
    hold_wp: float
    hold_runs: float
    send_wp: float
    send_runs: float
    p_safe: float  # the runner's chance of scoring if sent (unconditional)
    p_send: float  # the chance a coach sends in this situation
    break_even_wp: float | None  # p_safe that equalizes the options; None when a safe send is worth no more than an out
    break_even_runs: float | None
    branches: tuple[SendBranch, SendBranch, SendBranch]  # held, safe, out
    verdict: Literal["send", "hold"]  # the option with the higher win probability
    confidence: float  # share of bootstrap replicates of p_safe on the verdict's side of the win-probability break-even
    close_call: bool
    threshold: float

    def branch(self, result: Result) -> SendBranch:
        return next(b for b in self.branches if b.result == result)

    def send_wp_at(self, p: float) -> float:
        return p * self.branch("safe").value_wp + (1 - p) * self.branch("out").value_wp

    def send_runs_at(self, p: float) -> float:
        return p * self.branch("safe").value_runs + (1 - p) * self.branch("out").value_runs


def send_available(state: GameState, fielder_pos: int) -> bool:
    # Runner on second only, fewer than three outs, and the single fielded by an outfielder.
    return state.base_code == SEND_BASE_CODE and 0 <= state.outs <= 2 and fielder_pos in OUTFIELD


def send_decision(
    state: GameState,
    ball: BattedBall,
    *,
    runner: RunnerProfile,
    fielder: FielderProfile,
    start: PositioningStart,
    on_deck: PlayerProfile | None = None,
    pitcher: PlayerProfile | None = None,
    on_deck_id: str = "on_deck_batter",
    on_deck_hand: BatterHand | None = None,
    model: SendSuccessModel | None = None,
    batter_advance: dict[tuple[str, int], dict[str, float]] | None = None,
    season: int | None = None,
    close_call_threshold: float = CLOSE_CALL_THRESHOLD,
) -> SendDecision:
    # state: the state before the pitch (the batter at the plate hit the single, the runner is on second).
    # on_deck and pitcher: plate-appearance profiles for the next batter and the current pitcher (None is league average).
    if not send_available(state, ball.fielder_pos):
        raise ValueError(f"the send decision is not offered with base_code {state.base_code}, {state.outs} outs, fielder {ball.fielder_pos}")
    if not 0 < close_call_threshold <= 1:
        raise ValueError(f"close_call_threshold must be in (0, 1], got {close_call_threshold}")
    m = model or load_send_model()
    shares = batter_advance or load_batter_advance()
    yr = state.season if season is None else season
    next_id = on_deck.player_id if on_deck is not None else on_deck_id
    hand = on_deck_hand or (on_deck.hand if on_deck is not None and on_deck.hand in ("L", "R", "B") else "R")
    bat_side: Literal["L", "R"] = state.batter_hand if state.batter_hand in ("L", "R") else ("L" if state.pitcher_hand == "R" else "R")  # type: ignore[assignment]
    context = SendContext(outs=state.outs, score_diff=state.diff, inning=state.inning, bat_side=bat_side,
                          on_deck_quality=_on_deck_quality(state, on_deck, next_id, hand))
    args = (ball, runner, fielder, start, context, yr)
    p_safe = _probability(m.p_safe(*args), "p_safe")
    p_send = _probability(m.p_send(*args), "p_send")

    def value(base_post: int, outs_post: int, runs: int) -> tuple[float, float]:
        walk_off = state.half == 1 and state.inning >= 9 and state.bat_score + runs > state.fld_score
        if walk_off:
            return 1.0, float(runs)  # the game ends
        if outs_post >= 3:
            return wp_after(state, 0, 3, runs, new_pa=True), runs_after(state, 0, 3, runs)
        # The batter who singled and the runner occupy the bases; the on-deck batter comes up at 0-0.
        on1 = state.batter if base_post & 1 else None
        on2 = state.batter if base_post & 2 else None
        on3 = state.runner2 if base_post & 4 else None
        nxt = replace(state, runner1=on1, runner2=on2, runner3=on3, outs=outs_post, balls=0, strikes=0, bat_score=state.bat_score + runs,
                      batter=next_id, batter_hand=hand, pickoff_throws=0)
        return value_wp(nxt, on_deck, pitcher), runs + value_runs(nxt, on_deck, pitcher)  # runs on the play plus the rest of the half

    branches = tuple(_branch(result, state.outs, shares, value) for result in RESULTS)
    held, safe, out = branches
    hold = (held.value_wp, held.value_runs)
    send = (p_safe * safe.value_wp + (1 - p_safe) * out.value_wp, p_safe * safe.value_runs + (1 - p_safe) * out.value_runs)
    be_wp = _break_even(hold[0], safe.value_wp, out.value_wp)
    be_runs = _break_even(hold[1], safe.value_runs, out.value_runs)
    verdict: Literal["send", "hold"] = "send" if send[0] > hold[0] else "hold"
    reps = np.asarray(m.p_safe_bootstrap(*args), float)
    if be_wp is None:
        confidence = 1.0  # a safe send is worth no more than an out, so holding wins for any p_safe
    elif verdict == "send":
        confidence = float(np.mean(reps > be_wp))
    else:
        confidence = float(np.mean(reps <= be_wp))
    return SendDecision(hold_wp=hold[0], hold_runs=hold[1], send_wp=send[0], send_runs=send[1], p_safe=p_safe, p_send=p_send,
                        break_even_wp=be_wp, break_even_runs=be_runs, branches=(held, safe, out), verdict=verdict, confidence=confidence,
                        close_call=confidence < close_call_threshold, threshold=close_call_threshold)


def _branch(result: str, outs: int, shares: dict[tuple[str, int], dict[str, float]],
            value: Callable[[int, int, int], tuple[float, float]]) -> SendBranch:
    row, runs, runner_outs = RESULTS[result]
    split = shares[(row, outs)]
    outcomes = []
    for advance in ADVANCES:
        if result == "held":
            base_post = {"first": 5, "second": 6, "out": 4}[advance]  # runner on third plus the batter
        else:
            base_post = {"first": 1, "second": 2, "out": 0}[advance]  # the runner has scored or been put out
        outs_post = min(3, outs + runner_outs + (advance == "out"))
        v = value(base_post if outs_post < 3 else 0, outs_post, runs)
        outcomes.append(SendOutcome(advance, split[advance], base_post if outs_post < 3 else 0, outs_post, runs, v[0], v[1]))  # type: ignore[arg-type]
    return SendBranch(result, tuple(outcomes))  # type: ignore[arg-type]


def _break_even(hold: float, safe: float, out: float) -> float | None:
    # p* (V_safe - V_out) + V_out = V_hold; may fall outside [0, 1] when one option dominates.
    if safe <= out:
        return None
    return (hold - out) / (safe - out)


def _probability(p: float, name: str) -> float:
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"send model returned {name} = {p}")
    return float(p)


def _on_deck_quality(state: GameState, on_deck: PlayerProfile | None, next_id: str, hand: BatterHand) -> float | None:
    # As in the research fit: the on-deck batter's anchored run value against a league-average batter at runners on first
    # and third (the state if the runner holds), the play's outs, 0-0, against the pitcher's hand; 0 below the established threshold.
    if on_deck is None or on_deck.weighted_pa < ON_DECK_ESTABLISHED_PA:
        return None
    ref = replace(state, runner1="runner_on_first", runner2=None, runner3="runner_on_third", balls=0, strikes=0, batter=next_id,
                  batter_hand=hand, pickoff_throws=0, inning=1, half=0, bat_score=0, fld_score=0)
    return value_runs(ref, on_deck, None) - value_runs(ref, None, None)
