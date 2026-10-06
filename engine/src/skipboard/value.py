from dataclasses import dataclass, replace

from skipboard.state import GameState, GameType
from skipboard.tables import DIFF_MAX, WP_INNINGS, Tables, load_tables

EXTRA_INNINGS_RUNNER = 2  # regular-season extra innings start with a runner on second


@dataclass(frozen=True)
class Situation:
    game_type: GameType
    inning: int
    half: int
    diff: int  # batting team minus fielding team
    base_code: int
    outs: int
    balls: int = 0
    strikes: int = 0  # carried for count-aware tables; the v1 tables do not use the count


def situation(state: GameState) -> Situation:
    return Situation(state.game_type, state.inning, state.half, state.diff, state.base_code, state.outs, state.balls, state.strikes)


def wp_situation(s: Situation, tables: Tables | None = None) -> float:
    t = tables or load_tables()
    inning = min(s.inning, WP_INNINGS)
    diff = max(-DIFF_MAX, min(DIFF_MAX, s.diff))
    return float(t.wp[s.game_type][inning - 1, s.half, diff + DIFF_MAX, s.base_code, s.outs])


def wp(state: GameState, tables: Tables | None = None) -> float:
    return wp_situation(situation(state), tables)


def wp_after(state: GameState, base_post: int, outs_post: int, runs: int, new_pa: bool, tables: Tables | None = None) -> float:
    _check_play(state, base_post, outs_post, runs)
    bat = state.bat_score + runs
    fld = state.fld_score
    late = state.inning >= 9

    if state.half == 1 and late and bat > fld:
        return 1.0  # walk-off
    if outs_post < 3:
        balls, strikes = (0, 0) if new_pa else (state.balls, state.strikes)
        after = replace(situation(state), diff=bat - fld, base_code=base_post, outs=outs_post, balls=balls, strikes=strikes)
        return wp_situation(after, tables)

    # Third out: the half-inning ends.
    if state.half == 0 and late and fld > bat:
        return 0.0  # the team batting second leads after the top half
    if state.half == 1 and late and bat < fld:
        return 0.0
    next_inning, next_half = (state.inning, 1) if state.half == 0 else (state.inning + 1, 0)
    extra_runner = state.game_type == "regular" and next_inning >= 10
    nxt = Situation(state.game_type, next_inning, next_half, fld - bat, EXTRA_INNINGS_RUNNER if extra_runner else 0, 0)
    return 1.0 - wp_situation(nxt, tables)  # the teams switch


def run_expectancy(state: GameState, tables: Tables | None = None) -> float:
    t = tables or load_tables()
    return float(t.re24[state.base_code, state.outs])


def runs_after(state: GameState, base_post: int, outs_post: int, runs: int, tables: Tables | None = None) -> float:
    _check_play(state, base_post, outs_post, runs)
    if outs_post == 3:
        return float(runs)
    t = tables or load_tables()
    return runs + float(t.re24[base_post, outs_post])


def _check_play(state: GameState, base_post: int, outs_post: int, runs: int) -> None:
    if not 0 <= base_post <= 7:
        raise ValueError(f"base_post must be 0 to 7, got {base_post}")
    if not state.outs <= outs_post <= 3:
        raise ValueError(f"outs_post must be from {state.outs} to 3, got {outs_post}")
    if runs < 0:
        raise ValueError(f"runs cannot be negative, got {runs}")
