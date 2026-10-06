from typing import Any

import pytest

from skipboard import GameState

BASES = {1: "runner1", 2: "runner2", 4: "runner3"}


def make_state(base_code: int = 0, **overrides: Any) -> GameState:
    fields: dict[str, Any] = dict(
        season=2025,
        game_type="regular",
        inning=1,
        half=0,
        bat_score=0,
        fld_score=0,
        outs=0,
        balls=0,
        strikes=0,
        batter="batter01",
        batter_hand="R",
        pitcher="pitcher01",
        pitcher_hand="R",
        catcher="catcher01",
    )
    for bit, name in BASES.items():
        if base_code & bit:
            fields[name] = f"runner{bit:02d}"
    fields.update(overrides)
    return GameState(**fields)


@pytest.fixture
def state_factory():
    return make_state
