from dataclasses import replace

from skipboard.continue_pa import continue_pa_runs, continue_pa_wp
from skipboard.plate_appearance import PlayerProfile
from skipboard.state import GameState
from skipboard.value import run_expectancy, wp

# Anchored decision values: the level comes from the empirical tables, and the plate-appearance
# model adds only the difference between this matchup and count and league-average players at 0-0
# with the same handedness. Decision modules use these; continue_pa_* stay available as the raw model.


def _anchor(state: GameState) -> GameState:
    return replace(state, balls=0, strikes=0)


def value_wp(
    state: GameState,
    batter: PlayerProfile | None = None,
    pitcher: PlayerProfile | None = None,
    apply_platoon: bool = True,
    apply_state: bool = True,
) -> float:
    effect = continue_pa_wp(state, batter, pitcher, apply_platoon, apply_state) - continue_pa_wp(
        _anchor(state), None, None, apply_platoon, apply_state
    )
    return min(1.0, max(0.0, wp(state) + effect))  # keep it a probability at the extremes


def value_runs(
    state: GameState,
    batter: PlayerProfile | None = None,
    pitcher: PlayerProfile | None = None,
    apply_platoon: bool = True,
    apply_state: bool = True,
) -> float:
    effect = continue_pa_runs(state, batter, pitcher, apply_platoon, apply_state) - continue_pa_runs(
        _anchor(state), None, None, apply_platoon, apply_state
    )
    return run_expectancy(state) + effect
