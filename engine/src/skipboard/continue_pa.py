from collections.abc import Callable

from skipboard.plate_appearance import PlayerProfile, outcome_distribution
from skipboard.state import GameState
from skipboard.transitions import transitions
from skipboard.value import runs_after, wp_after


def _expected(
    state: GameState,
    batter: PlayerProfile | None,
    pitcher: PlayerProfile | None,
    value: Callable[[int, int, int], float],
    apply_platoon: bool,
    apply_state: bool,
) -> float:
    total = 0.0
    for category, p in outcome_distribution(state, batter, pitcher, apply_platoon=apply_platoon, apply_state=apply_state).items():
        if p == 0:
            continue
        options = transitions(state, category)
        if not options:
            raise ValueError(f"no transitions for {category} with base_code {state.base_code} and {state.outs} outs")
        total += p * sum(t.p * value(t.base_post, t.outs_post, t.runs) for t in options)
    return total


def continue_pa_wp(
    state: GameState,
    batter: PlayerProfile | None = None,
    pitcher: PlayerProfile | None = None,
    apply_platoon: bool = True,
    apply_state: bool = True,
) -> float:
    return _expected(state, batter, pitcher, lambda b, o, r: wp_after(state, b, o, r, new_pa=True), apply_platoon, apply_state)


def continue_pa_runs(
    state: GameState,
    batter: PlayerProfile | None = None,
    pitcher: PlayerProfile | None = None,
    apply_platoon: bool = True,
    apply_state: bool = True,
) -> float:
    return _expected(state, batter, pitcher, lambda b, o, r: runs_after(state, b, o, r), apply_platoon, apply_state)
