from pathlib import Path

import numpy as np
import pytest
from conftest import scaled_profile

from skipboard.steal_decision import steal_available, steal_decision
from skipboard.steal_model import CatcherProfile, PitcherProfile, RunnerProfile, load_steal_model
from skipboard.steal_tables import load_steal_tables
from skipboard.value import run_expectancy, runs_after, wp, wp_after

FIXTURE = Path(__file__).parent / "fixtures" / "steal_model"
GAME = dict(inning=7, half=1, bat_score=3, fld_score=3)
SITUATIONS = [(0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 1, 0), (0, 2, 1), (3, 2, 1), (1, 0, 2)]


@pytest.fixture
def model():
    return load_steal_model(FIXTURE)


def decide(model, state, runner=None, catcher=None, pitcher_hold=None, batter=None, pitcher=None, **kw):
    return steal_decision(state, batter, pitcher, runner=runner or RunnerProfile(sprint_speed=28.0, aggressiveness=0.08),
                          catcher=catcher or CatcherProfile(pop_time=2.0),
                          pitcher_hold=pitcher_hold or PitcherProfile(hand=state.pitcher_hand, primary_lead=10.0, lead_gained=4.0),
                          model=model, **kw)


@pytest.mark.parametrize("balls, strikes, outs", SITUATIONS)
def test_probabilities_sum_to_one(model, state_factory, balls, strikes, outs):
    d = decide(model, state_factory(base_code=1, balls=balls, strikes=strikes, outs=outs, **GAME))
    assert sum(b.share for b in d.branches) == pytest.approx(1, abs=1e-12)
    for b in d.branches:
        assert sum(o.p for o in b.outcomes) == pytest.approx(1, abs=1e-12), b.result
        assert all(0 <= o.p <= 1 for o in b.outcomes)
        if b.is_throw:
            assert [o.label for o in b.outcomes] == ["safe", "out"] and 0 <= b.p_safe <= 1


@pytest.mark.parametrize("inning, half, outs", [(1, 0, 0), (5, 1, 1), (9, 1, 2), (9, 0, 0)])
def test_hold_value_is_anchored_for_league_players_at_0_0(model, state_factory, inning, half, outs):
    state = state_factory(base_code=1, inning=inning, half=half, outs=outs, bat_score=2, fld_score=2)
    d = decide(model, state)
    assert d.hold_wp == pytest.approx(wp(state), abs=1e-12)
    assert d.hold_runs == pytest.approx(run_expectancy(state), abs=1e-12)


@pytest.mark.parametrize("balls, strikes", [(0, 0), (2, 1)])
def test_caught_stealing_for_the_third_out_ends_the_half_inning(model, state_factory, balls, strikes):
    state = state_factory(base_code=1, balls=balls, strikes=strikes, outs=2, **GAME)
    d = decide(model, state)
    for b in d.branches:
        if b.result in ("ball", "strike"):
            out = b.outcomes[1]
            assert out.value_wp == pytest.approx(wp_after(state, 0, 3, 0, new_pa=True), abs=1e-12)
            assert out.value_runs == pytest.approx(runs_after(state, 0, 3, 0), abs=1e-12) == 0.0


def test_strike_three_with_two_outs_ignores_the_running_game(model, state_factory):
    state = state_factory(base_code=1, balls=1, strikes=2, outs=2, **GAME)
    profiles = [
        (RunnerProfile(sprint_speed=24.0), CatcherProfile(pop_time=1.8), PitcherProfile(hand="R", primary_lead=8.0, lead_gained=1.0)),
        (RunnerProfile(sprint_speed=30.5, aggressiveness=0.2), CatcherProfile(pop_time=2.2), PitcherProfile(hand="R", primary_lead=13.0, lead_gained=9.0)),
    ]
    values = []
    for runner, catcher, pitcher_hold in profiles:
        b = next(b for b in decide(model, state, runner, catcher, pitcher_hold).branches if b.result == "strike_three")
        assert b.p_safe is None
        values.append((b.value_wp, b.value_runs))
    assert values[0] == values[1]
    assert values[0][0] == pytest.approx(wp_after(state, 0, 3, 0, new_pa=True), abs=1e-12)


@pytest.mark.parametrize("balls, outs", [(0, 0), (2, 1), (3, 0)])
def test_foul_with_two_strikes_equals_holding(model, state_factory, balls, outs):
    d = decide(model, state_factory(base_code=1, balls=balls, strikes=2, outs=outs, **GAME))
    foul = next(b for b in d.branches if b.result == "foul")
    assert foul.value_wp == pytest.approx(d.hold_wp, abs=1e-15)
    assert foul.value_runs == pytest.approx(d.hold_runs, abs=1e-15)


@pytest.mark.parametrize("balls, strikes, outs", SITUATIONS)
def test_faster_runner_and_slower_pop_never_lower_the_run_value(model, state_factory, balls, strikes, outs):
    state = state_factory(base_code=1, balls=balls, strikes=strikes, outs=outs, **GAME)
    runs_by_speed = [decide(model, state, runner=RunnerProfile(sprint_speed=v)) for v in np.linspace(24.0, 31.0, 8)]
    runs_by_pop = [decide(model, state, catcher=CatcherProfile(pop_time=v)) for v in np.linspace(1.8, 2.2, 9)]
    for seq in (runs_by_speed, runs_by_pop):
        assert all(b.run_wp >= a.run_wp - 1e-15 for a, b in zip(seq, seq[1:]))
        assert all(b.run_runs >= a.run_runs - 1e-15 for a, b in zip(seq, seq[1:]))


def test_full_count_with_two_outs_is_not_offered(model, state_factory):
    state = state_factory(base_code=1, balls=3, strikes=2, outs=2, **GAME)
    assert not steal_available(state)
    with pytest.raises(ValueError):
        decide(model, state)
    assert steal_available(state_factory(base_code=1, balls=3, strikes=2, outs=1, **GAME))
    for base in (0, 2, 3, 5, 7):
        assert not steal_available(state_factory(base_code=base, **GAME))


@pytest.mark.parametrize("balls, strikes, outs", [(0, 0, 0), (1, 1, 1), (2, 0, 2)])
def test_break_even_reproduces_equality(model, state_factory, balls, strikes, outs):
    d = decide(model, state_factory(base_code=1, balls=balls, strikes=strikes, outs=outs, **GAME))
    assert d.break_even_wp is not None and d.break_even_runs is not None
    assert d.run_wp_at(d.break_even_wp) == pytest.approx(d.hold_wp, abs=1e-9)
    assert d.run_runs_at(d.break_even_runs) == pytest.approx(d.hold_runs, abs=1e-9)


class AlwaysSafe:
    def p_safe(self, *args: object) -> float:
        return 1.0


def test_certain_success_is_worth_running_at_0_0_with_no_outs(state_factory):
    state = state_factory(base_code=1, **GAME)
    d = steal_decision(state, runner=RunnerProfile(), catcher=CatcherProfile(), pitcher_hold=PitcherProfile(hand="R"), model=AlwaysSafe())
    assert d.effective_p_safe == 1.0
    assert d.run_runs >= d.hold_runs
    assert d.run_runs == pytest.approx(d.run_runs_at(1.0), abs=1e-15)


def test_player_effects_and_pickoffs_reach_the_decision(model, state_factory):
    state = state_factory(base_code=1, **GAME)
    league = decide(model, state)
    slugger = decide(model, state, batter=scaled_profile("batter", "HR", 2.0))
    assert slugger.hold_wp > league.hold_wp
    picked = decide(model, state_factory(base_code=1, pickoff_throws=2, **GAME))
    assert picked.effective_p_safe != league.effective_p_safe


def test_pitcher_hand_must_match_the_state(model, state_factory):
    with pytest.raises(ValueError):
        decide(model, state_factory(base_code=1, pitcher_hand="R", **GAME), pitcher_hold=PitcherProfile(hand="L"))


@pytest.mark.parametrize("outs", [0, 1, 2])
def test_inplay_scoring_transitions_are_valued_as_runs_plus_run_expectancy(model, state_factory, outs):
    # Regression: a runner-going in-play transition that scores is worth the runs scored plus the run expectancy of the resulting state.
    tables = load_steal_tables()
    state = state_factory(base_code=1, outs=outs, **GAME)
    in_play = next(b for b in decide(model, state).branches if b.result == "in_play")
    scoring = []
    for o in in_play.outcomes:
        transitions = tables.going_transitions(outs, o.label)
        expected = sum(tr.p * (tr.runs + (0.0 if tr.outs_post >= 3 else run_expectancy(state_factory(base_code=tr.base_post, outs=tr.outs_post, **GAME))))
                       for tr in transitions)
        assert o.value_runs == pytest.approx(expected, abs=1e-12), o.label
        if any(tr.runs > 0 for tr in transitions):
            scoring.append(o.label)
    assert {"2B", "HR"} <= set(scoring)
