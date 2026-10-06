import pytest
from conftest import scaled_profile

from skipboard.value import run_expectancy, wp, wp_after
from skipboard.valuation import value_runs, value_wp

BASES = (0, 1, 2, 4, 3, 5, 6, 7)
SCORES = {"tied": (2, 2), "down 1": (1, 2)}
STEAL_CONTEXTS = [(9, 0, "top 9th, tied"), (9, 1, "bottom 9th, tied"), (7, 1, "bottom 7th, tied")]


@pytest.mark.parametrize("inning", [1, 5, 9])
@pytest.mark.parametrize("half", [0, 1])
@pytest.mark.parametrize("score", list(SCORES), ids=list(SCORES))
def test_league_players_at_0_0_reproduce_tables(state_factory, inning, half, score):
    bat, fld = SCORES[score]
    for base in BASES:
        for outs in range(3):
            s = state_factory(inning=inning, half=half, base_code=base, outs=outs, bat_score=bat, fld_score=fld)
            assert value_wp(s) == pytest.approx(wp(s), abs=1e-12)
            assert value_runs(s) == pytest.approx(run_expectancy(s), abs=1e-12)


def _break_even(v_first: float, v_second: float, v_fail: float) -> float:
    return (v_first - v_fail) / (v_second - v_fail)


@pytest.mark.parametrize("outs", range(3))
def test_steal_break_even_in_runs_equals_tables(state_factory, outs):
    first, second = state_factory(base_code=1, outs=outs), state_factory(base_code=2, outs=outs)
    fail = state_factory(outs=outs + 1) if outs < 2 else None
    fail_table = run_expectancy(fail) if fail else 0.0
    fail_value = value_runs(fail) if fail else 0.0
    table = _break_even(run_expectancy(first), run_expectancy(second), fail_table)
    anchored = _break_even(value_runs(first), value_runs(second), fail_value)
    assert anchored == pytest.approx(table, abs=1e-12)


@pytest.mark.parametrize("inning, half, label", STEAL_CONTEXTS, ids=[c[2] for c in STEAL_CONTEXTS])
@pytest.mark.parametrize("outs", range(3))
def test_steal_break_even_in_wp_equals_tables(state_factory, inning, half, label, outs):
    game = dict(inning=inning, half=half, bat_score=2, fld_score=2)
    first, second = state_factory(base_code=1, outs=outs, **game), state_factory(base_code=2, outs=outs, **game)
    if outs < 2:
        fail = state_factory(outs=outs + 1, **game)
        fail_table, fail_value = wp(fail), value_wp(fail)
    else:
        fail_table = fail_value = wp_after(first, base_post=0, outs_post=3, runs=0, new_pa=False)
    table = _break_even(wp(first), wp(second), fail_table)
    anchored = _break_even(value_wp(first), value_wp(second), fail_value)
    assert anchored == pytest.approx(table, abs=1e-12)


def test_count_moves_value(state_factory):
    at = lambda balls, strikes: value_wp(state_factory(inning=7, half=1, base_code=1, balls=balls, strikes=strikes))  # noqa: E731
    assert at(0, 2) < at(0, 0) < at(3, 0)


def test_player_effects(state_factory):
    s = state_factory(inning=5, half=0, base_code=1, outs=1)
    league = value_wp(s)
    assert value_wp(s, scaled_profile("batter", "K", 1.5)) < league
    assert value_wp(s, scaled_profile("batter", "HR", 2.0)) > league
    assert value_runs(s, scaled_profile("batter", "HR", 2.0)) > value_runs(s)


def test_value_wp_stays_a_probability(state_factory):
    slugger = scaled_profile("batter", "HR", 4.0)
    for half, bat, fld in [(1, 3, 3), (0, 9, 0), (0, 0, 9)]:
        for base in BASES:
            v = value_wp(state_factory(inning=9, half=half, base_code=base, outs=2, bat_score=bat, fld_score=fld, balls=3), slugger)
            assert 0.0 <= v <= 1.0
