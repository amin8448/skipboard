import math
from pathlib import Path

import numpy as np
import pytest
from conftest import scaled_profile

from skipboard.plate_appearance import PlayerProfile
from skipboard.send_decision import send_available, send_decision
from skipboard.send_model import BattedBall, FielderProfile, PositioningStart, load_batter_advance, load_send_model
from skipboard.steal_model import RunnerProfile
from skipboard.value import run_expectancy, wp, wp_after

FIXTURE = Path(__file__).parent / "fixtures" / "send_model"
GAME = dict(inning=7, half=1, bat_score=3, fld_score=3)
BALL = BattedBall(x_feet=-150.0, y_feet=200.0, bb_type="line_drive", fielder_pos=7, launch_speed=95.0, launch_angle=12.0)


@pytest.fixture
def model():
    return load_send_model(FIXTURE)


@pytest.fixture
def shares():
    return load_batter_advance()  # the real Retrosheet table in models/send/


def decide(model, shares, state, ball=BALL, runner=None, fielder=None, start=None, **kw):
    return send_decision(state, ball, runner=runner or RunnerProfile(sprint_speed=27.5), fielder=fielder or FielderProfile(arm_avg=88.0),
                         start=start or PositioningStart(start_distance=290.0, start_angle=-28.0), model=model, batter_advance=shares, **kw)


class FixedModel:
    # Stand-in success model: fixed p_safe and bootstrap replicates.
    def __init__(self, p_safe: float, replicates: list[float] | None = None, p_send: float = 0.5) -> None:
        self.value, self.reps, self.send = p_safe, np.array(replicates if replicates is not None else [p_safe]), p_send

    def p_safe(self, *args: object) -> float:
        return self.value

    def p_send(self, *args: object) -> float:
        return self.send

    def p_safe_bootstrap(self, *args: object) -> np.ndarray:
        return self.reps


@pytest.mark.parametrize("outs", [0, 1, 2])
def test_branch_probabilities_sum_to_one(model, shares, state_factory, outs):
    d = decide(model, shares, state_factory(base_code=2, outs=outs, **GAME))
    for b in d.branches:
        assert sum(o.p for o in b.outcomes) == pytest.approx(1.0, abs=1e-12)
        assert [o.advance for o in b.outcomes] == ["first", "second", "out"]
    assert d.p_safe + (1 - d.p_safe) == pytest.approx(1.0)
    assert 0 <= d.p_safe <= 1 and 0 <= d.p_send <= 1


@pytest.mark.parametrize("outs", [0, 1, 2])
@pytest.mark.parametrize("inning, half, bat, fld", [(7, 1, 3, 3), (3, 0, 1, 2), (8, 1, 2, 3)])
def test_hold_value_is_anchored_for_league_players(model, shares, state_factory, outs, inning, half, bat, fld):
    state = state_factory(base_code=2, outs=outs, inning=inning, half=half, bat_score=bat, fld_score=fld)
    d = decide(model, shares, state)
    split = shares[("held at third", outs)]
    exp_wp, exp_runs = 0.0, 0.0
    for advance, base, extra in [("first", 5, 0), ("second", 6, 0), ("out", 4, 1)]:
        outs_post = outs + extra
        if outs_post >= 3:
            v_wp, v_runs = wp_after(state, 0, 3, 0, new_pa=True), 0.0
        else:
            nxt = state_factory(base_code=base, outs=outs_post, inning=inning, half=half, bat_score=bat, fld_score=fld)
            v_wp, v_runs = wp(nxt), run_expectancy(nxt)
        exp_wp += split[advance] * v_wp
        exp_runs += split[advance] * v_runs
    assert d.hold_wp == pytest.approx(exp_wp, abs=1e-12)
    assert d.hold_runs == pytest.approx(exp_runs, abs=1e-12)


def test_third_outs_end_the_half_inning(model, shares, state_factory):
    state = state_factory(base_code=2, outs=2, **GAME)
    d = decide(model, shares, state)
    end = wp_after(state, 0, 3, 0, new_pa=True)
    for o in d.branch("out").outcomes:  # thrown out at home with two outs: the third out
        assert o.outs_post == 3 and o.base_post == 0
        assert o.value_wp == pytest.approx(end, abs=1e-15) and o.value_runs == 0.0
    batter_out = next(o for o in d.branch("held").outcomes if o.advance == "out")  # the batter thrown out with two outs
    assert batter_out.outs_post == 3 and batter_out.value_wp == pytest.approx(end, abs=1e-15) and batter_out.value_runs == 0.0
    scored_first = next(o for o in d.branch("safe").outcomes if o.advance == "first")
    assert scored_first.value_runs == pytest.approx(1.0 + run_expectancy(state_factory(base_code=1, outs=2, **GAME)), abs=1e-12)
    scored_then_out = next(o for o in d.branch("safe").outcomes if o.advance == "out")  # the run counts, then the half ends
    assert scored_then_out.outs_post == 3 and scored_then_out.value_runs == 1.0
    assert scored_then_out.value_wp == pytest.approx(wp_after(state, 0, 3, 1, new_pa=True), abs=1e-15)


def test_walk_off_safe_send_wins_the_game(model, shares, state_factory):
    d = decide(model, shares, state_factory(base_code=2, outs=1, inning=9, half=1, bat_score=2, fld_score=2))
    assert all(o.value_wp == 1.0 for o in d.branch("safe").outcomes)


@pytest.mark.parametrize("outs", [0, 1, 2])
def test_faster_runner_never_lowers_the_send_value(model, shares, state_factory, outs):
    state = state_factory(base_code=2, outs=outs, **GAME)
    seq = [decide(model, shares, state, runner=RunnerProfile(sprint_speed=v)) for v in np.linspace(24.0, 31.0, 8)]
    assert all(b.send_wp >= a.send_wp - 1e-15 and b.send_runs >= a.send_runs - 1e-15 for a, b in zip(seq, seq[1:]))


@pytest.mark.parametrize("outs", [0, 1, 2])
def test_stronger_arm_never_raises_the_send_value(model, shares, state_factory, outs):
    state = state_factory(base_code=2, outs=outs, **GAME)
    seq = [decide(model, shares, state, fielder=FielderProfile(arm_avg=v)) for v in np.linspace(80.0, 96.0, 9)]
    assert all(b.send_wp <= a.send_wp + 1e-15 and b.send_runs <= a.send_runs + 1e-15 for a, b in zip(seq, seq[1:]))


def test_deeper_fielding_point_never_lowers_the_send_value(model, shares, state_factory):
    # The fielder starts 40 ft short of the ball on the same line, so only the fielding-point distance changes.
    state = state_factory(base_code=2, outs=1, **GAME)
    angle = -20.0
    seq = []
    for dist in np.linspace(200.0, 360.0, 9):
        ball = BattedBall(x_feet=dist * math.sin(math.radians(angle)), y_feet=dist * math.cos(math.radians(angle)), bb_type="line_drive",
                          fielder_pos=7, launch_speed=92.0, launch_angle=12.0)
        seq.append(decide(model, shares, state, ball=ball, start=PositioningStart(start_distance=dist - 40.0, start_angle=angle)))
    assert all(b.send_wp >= a.send_wp - 1e-15 and b.send_runs >= a.send_runs - 1e-15 for a, b in zip(seq, seq[1:]))


@pytest.mark.parametrize("outs", [0, 1, 2])
def test_break_even_reproduces_equality(model, shares, state_factory, outs):
    d = decide(model, shares, state_factory(base_code=2, outs=outs, **GAME))
    assert d.break_even_wp is not None and d.break_even_runs is not None
    assert d.send_wp_at(d.break_even_wp) == pytest.approx(d.hold_wp, abs=1e-12)
    assert d.send_runs_at(d.break_even_runs) == pytest.approx(d.hold_runs, abs=1e-12)
    assert d.send_wp_at(d.p_safe) == pytest.approx(d.send_wp, abs=1e-15)


def test_confidence_and_close_call(model, shares, state_factory):
    state = state_factory(base_code=2, outs=2, **GAME)
    p_star = decide(model, shares, state).break_even_wp
    assert p_star is not None and 0 < p_star < 0.9
    sure = decide(FixedModel(0.95, [0.93, 0.95, 0.97]), shares, state)
    assert sure.verdict == "send" and sure.confidence == 1.0 and not sure.close_call
    mixed = [p_star + 0.05] * 6 + [p_star - 0.05] * 4
    split = decide(FixedModel(p_star + 0.05, mixed), shares, state, close_call_threshold=0.7)
    assert split.verdict == "send" and split.confidence == pytest.approx(0.6) and split.close_call and split.threshold == 0.7
    lenient = decide(FixedModel(p_star + 0.05, mixed), shares, state, close_call_threshold=0.5)
    assert lenient.confidence == pytest.approx(0.6) and not lenient.close_call
    held = decide(FixedModel(p_star - 0.05, mixed), shares, state)
    assert held.verdict == "hold" and held.confidence == pytest.approx(0.4) and held.close_call


def test_certain_outcomes(shares, state_factory):
    state = state_factory(base_code=2, outs=0, **GAME)
    always = decide(FixedModel(1.0), shares, state)
    assert always.send_runs >= always.hold_runs
    never = decide(FixedModel(0.0), shares, state)
    assert never.hold_runs >= never.send_runs and never.hold_wp >= never.send_wp
    assert never.verdict == "hold"


def test_on_deck_batter_enters_the_values_and_the_send_equation(model, shares, state_factory):
    state = state_factory(base_code=2, outs=1, **GAME)
    league = decide(model, shares, state)
    rates = scaled_profile("batter", "HR", 2.0).rates
    slugger = PlayerProfile("slugger01", "batter", rates, "R", 3000.0)
    unproven = PlayerProfile("rookie01", "batter", rates, "R", 100.0)
    strong = decide(model, shares, state, on_deck=slugger)
    assert strong.hold_wp > league.hold_wp and strong.p_send != league.p_send
    weak_history = decide(model, shares, state, on_deck=unproven)
    assert weak_history.p_send == pytest.approx(league.p_send, abs=1e-15)  # below the established threshold: league average in Z
    assert weak_history.hold_wp == pytest.approx(strong.hold_wp, abs=1e-15)  # the profile still values the next plate appearance


def test_unavailable_states_are_not_offered(model, shares, state_factory):
    assert send_available(state_factory(base_code=2, outs=1), 8)
    for base in (0, 1, 3, 6, 7):
        state = state_factory(base_code=base, outs=1)
        assert not send_available(state, 8)
        with pytest.raises(ValueError):
            decide(model, shares, state)
    assert not send_available(state_factory(base_code=2, outs=1), 6)  # infield fielder


def test_close_call_threshold_is_checked(model, shares, state_factory):
    with pytest.raises(ValueError):
        decide(model, shares, state_factory(base_code=2, outs=1, **GAME), close_call_threshold=0.0)
