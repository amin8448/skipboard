import pytest

from skipboard import run_expectancy, runs_after, wp, wp_after
from skipboard.value import Situation, wp_situation

# Spot checks from research/win_probability.py, regular season, 2023-2025 tables.
SPOT_CHECKS = [
    ("top 1st, tied, bases empty, 0 outs", dict(inning=1, half=0), 0, 0.470),
    ("bottom 9th, tied, bases empty, 0 outs", dict(inning=9, half=1), 0, 0.644),
    ("bottom 9th, tied, runner on second, 0 outs", dict(inning=9, half=1), 2, 0.809),
    ("bottom 9th, down 1, runner on first, 0 outs", dict(inning=9, half=1, fld_score=1), 1, 0.342),
    ("top 9th, down 1, runner on first, 1 out", dict(inning=9, half=0, fld_score=1, outs=1), 1, 0.169),
    ("bottom 7th, tied, runner on first, 1 out", dict(inning=7, half=1, outs=1), 1, 0.606),
    # The table value is 0.4945; it rounds to 0.494.
    ("top 10th, tied, runner on second, 0 outs", dict(inning=10, half=0), 2, 0.494),
    ("bottom 10th, down 1, runner on second, 0 outs", dict(inning=10, half=1, fld_score=1), 2, 0.455),
]


@pytest.mark.parametrize("label, overrides, base_code, expected", SPOT_CHECKS, ids=[c[0] for c in SPOT_CHECKS])
def test_spot_checks(state_factory, label, overrides, base_code, expected):
    assert wp(state_factory(base_code=base_code, **overrides)) == pytest.approx(expected, abs=0.0005)


def test_diff_clamped_and_late_innings_use_inning_12(state_factory):
    assert wp(state_factory(bat_score=25)) == wp(state_factory(bat_score=10))
    assert wp(state_factory(inning=15, half=0)) == wp(state_factory(inning=12, half=0))


def test_third_out_top_5th_switches_to_bottom_5th(state_factory):
    s = state_factory(inning=5, half=0, bat_score=2, fld_score=3, outs=2, base_code=1)
    expected = 1 - wp_situation(Situation("regular", 5, 1, 1, 0, 0))
    assert wp_after(s, base_post=0, outs_post=3, runs=0, new_pa=True) == pytest.approx(expected)


def test_walk_off_returns_one(state_factory):
    s = state_factory(inning=9, half=1, bat_score=3, fld_score=3, outs=1, base_code=4)
    assert wp_after(s, base_post=1, outs_post=1, runs=1, new_pa=True) == 1.0


def test_walk_off_in_extra_innings(state_factory):
    s = state_factory(inning=11, half=1, bat_score=4, fld_score=5, outs=2, base_code=6)
    assert wp_after(s, base_post=1, outs_post=2, runs=2, new_pa=True) == 1.0


def test_third_out_top_9th_fielding_team_ahead_returns_zero(state_factory):
    s = state_factory(inning=9, half=0, bat_score=2, fld_score=3, outs=2)
    assert wp_after(s, base_post=0, outs_post=3, runs=0, new_pa=True) == 0.0


def test_third_out_bottom_9th_trailing_returns_zero(state_factory):
    s = state_factory(inning=9, half=1, bat_score=2, fld_score=3, outs=2)
    assert wp_after(s, base_post=0, outs_post=3, runs=0, new_pa=True) == 0.0


def test_third_out_bottom_9th_tied_goes_to_extra_innings(state_factory):
    s = state_factory(inning=9, half=1, bat_score=3, fld_score=3, outs=2)
    expected = 1 - wp_situation(Situation("regular", 10, 0, 0, 2, 0))
    assert wp_after(s, base_post=0, outs_post=3, runs=0, new_pa=True) == pytest.approx(expected)


def test_extra_innings_runner_regular_but_not_postseason(state_factory):
    regular = state_factory(inning=10, half=0, outs=2, base_code=2)
    postseason = state_factory(inning=10, half=0, outs=2, base_code=2, game_type="postseason")
    v_regular = wp_after(regular, base_post=0, outs_post=3, runs=0, new_pa=True)
    v_post = wp_after(postseason, base_post=0, outs_post=3, runs=0, new_pa=True)
    assert v_regular == pytest.approx(1 - wp_situation(Situation("regular", 10, 1, 0, 2, 0)))
    assert v_post == pytest.approx(1 - wp_situation(Situation("postseason", 10, 1, 0, 0, 0)))
    assert v_regular != pytest.approx(v_post, abs=0.01)


@pytest.mark.parametrize("half", [0, 1])
def test_runs_go_to_batting_team(state_factory, half):
    s = state_factory(inning=3, half=half, bat_score=2, fld_score=3, outs=1, base_code=1)
    after = wp_after(s, base_post=0, outs_post=1, runs=2, new_pa=True)
    assert after == pytest.approx(wp_situation(Situation("regular", 3, half, 1, 0, 1)))
    assert after > wp(s)


def test_runs_count_before_half_inning_switch(state_factory):
    s = state_factory(inning=5, half=0, bat_score=1, fld_score=1, outs=2, base_code=4)
    after = wp_after(s, base_post=0, outs_post=3, runs=1, new_pa=True)
    assert after == pytest.approx(1 - wp_situation(Situation("regular", 5, 1, -1, 0, 0)))


def test_count_does_not_change_v1_value(state_factory):
    s = state_factory(balls=3, strikes=1, base_code=1)
    assert wp_after(s, 3, 0, 0, new_pa=False) == wp_after(s, 3, 0, 0, new_pa=True)


def test_invalid_play_rejected(state_factory):
    s = state_factory(outs=1)
    with pytest.raises(ValueError):
        wp_after(s, base_post=0, outs_post=0, runs=0, new_pa=True)
    with pytest.raises(ValueError):
        wp_after(s, base_post=8, outs_post=1, runs=0, new_pa=True)
    with pytest.raises(ValueError):
        wp_after(s, base_post=0, outs_post=1, runs=-1, new_pa=True)


def test_run_expectancy(state_factory):
    assert run_expectancy(state_factory()) == pytest.approx(0.5046, abs=1e-4)
    assert run_expectancy(state_factory(base_code=7, outs=2)) == pytest.approx(0.7939, abs=1e-4)
    s = state_factory(base_code=2)
    assert runs_after(s, base_post=1, outs_post=0, runs=1) == pytest.approx(1 + 0.9016, abs=1e-4)
    assert runs_after(state_factory(outs=2, base_code=4), base_post=0, outs_post=3, runs=0) == 0.0
