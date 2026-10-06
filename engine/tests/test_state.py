import dataclasses

import pytest

from skipboard import InvalidStateError, Outfielders


@pytest.mark.parametrize(
    "overrides",
    [
        {"outs": -1},
        {"outs": 3},
        {"balls": -1},
        {"balls": 4},
        {"strikes": -1},
        {"strikes": 3},
        {"bat_score": -1},
        {"fld_score": -1},
        {"inning": 0},
        {"inning": 9, "half": 1, "bat_score": 3, "fld_score": 2},
        {"inning": 11, "half": 1, "bat_score": 5, "fld_score": 4},
    ],
)
def test_impossible_states_rejected(state_factory, overrides):
    with pytest.raises(InvalidStateError):
        state_factory(**overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"outs": 2, "balls": 3, "strikes": 2},
        {"inning": 9, "half": 1, "bat_score": 2, "fld_score": 2},
        {"inning": 9, "half": 1, "bat_score": 1, "fld_score": 4},
        {"inning": 9, "half": 0, "bat_score": 6, "fld_score": 2},
        {"inning": 8, "half": 1, "bat_score": 6, "fld_score": 2},
        {"inning": 14, "half": 0, "game_type": "postseason"},
        {"outfielders": Outfielders("lf01", "cf01", "rf01"), "disengagements_used": 2, "disengagement_limit": 3},
    ],
)
def test_normal_states_accepted(state_factory, overrides):
    state_factory(**overrides)


@pytest.mark.parametrize("base_code", range(8))
def test_base_code(state_factory, base_code):
    assert state_factory(base_code=base_code).base_code == base_code


def test_base_code_by_runner(state_factory):
    s = state_factory()
    assert dataclasses.replace(s, runner1="a").base_code == 1
    assert dataclasses.replace(s, runner2="a").base_code == 2
    assert dataclasses.replace(s, runner3="a").base_code == 4
    assert dataclasses.replace(s, runner1="a", runner2="b", runner3="c").base_code == 7


def test_diff(state_factory):
    assert state_factory(bat_score=2, fld_score=5).diff == -3


def test_state_is_frozen(state_factory):
    with pytest.raises(dataclasses.FrozenInstanceError):
        state_factory().outs = 1  # type: ignore[misc]
