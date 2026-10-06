import numpy as np
import pytest
from conftest import league_profile, scaled_profile

from skipboard.plate_appearance import (
    CATEGORIES,
    COUNTS,
    MATCHUP_CATEGORIES,
    PlayerProfile,
    load_pa_tables,
    outcome_distribution,
)


HANDS = [("R", "R"), ("L", "R"), ("B", "L"), ("L", "L")]


@pytest.mark.parametrize("batter_hand, pitcher_hand", HANDS)
def test_distributions_are_valid(state_factory, batter_hand, pitcher_hand):
    hitter = scaled_profile("batter", "HR", 2.0)
    for base in range(8):
        for outs in range(3):
            for balls, strikes in COUNTS:
                s = state_factory(base_code=base, outs=outs, balls=balls, strikes=strikes,
                                  batter_hand=batter_hand, pitcher_hand=pitcher_hand)
                for batter in (None, hitter):
                    d = outcome_distribution(s, batter, league_profile("pitcher"))
                    assert list(d) == list(CATEGORIES)
                    assert min(d.values()) >= 0
                    assert sum(d.values()) == pytest.approx(1, abs=1e-12)
                    if base == 0:
                        assert d["FC"] == 0


def test_missing_profiles_mean_league_rates(state_factory):
    s = state_factory(base_code=3, outs=1, balls=1, strikes=2)
    assert outcome_distribution(s) == pytest.approx(outcome_distribution(s, league_profile("batter"), league_profile("pitcher")))


@pytest.mark.parametrize("base, outs", [(0, 0), (1, 1), (5, 0), (7, 2)])
def test_league_players_at_0_0_reproduce_league_row(state_factory, base, outs):
    t = load_pa_tables()
    row = dict(zip(CATEGORIES, t.counts12[(0, 0)]))
    s = state_factory(base_code=base, outs=outs)
    d = outcome_distribution(s, league_profile("batter"), league_profile("pitcher"), apply_platoon=False, apply_state=False)
    for c in CATEGORIES:
        if c not in ("GB_OUT", "FC"):
            assert d[c] == pytest.approx(row[c], abs=1e-12)
    ground = row["GB_OUT"] + row["FC"]
    share = t.fc_share[base, outs]
    assert d["GB_OUT"] == pytest.approx(ground * (1 - share), abs=1e-12)
    assert d["FC"] == pytest.approx(ground * share, abs=1e-12)


@pytest.mark.parametrize("base, outs", [(0, 0), (4, 1), (5, 0), (6, 2)])
def test_state_adjustment_reproduces_state_shares_at_0_0(state_factory, base, outs):
    # League players at 0-0 with the state ratio on: the nine matchup categories follow the state's own shares.
    t = load_pa_tables()
    d = outcome_distribution(state_factory(base_code=base, outs=outs), apply_platoon=False)
    d["GB"] = d.pop("GB_OUT") + d.pop("FC")
    in_model = 1 - d["ROE"] - d["OTHER"]
    expected = t.league9 * t.state_ratio[base, outs]
    expected = expected / expected.sum()
    for i, c in enumerate(MATCHUP_CATEGORIES):
        assert d[c] / in_model == pytest.approx(expected[i], abs=1e-12)


def test_counts_shift_strikeouts_and_walks(state_factory):
    at = lambda b, s: outcome_distribution(state_factory(balls=b, strikes=s))  # noqa: E731
    assert at(0, 2)["K"] > at(0, 0)["K"]
    assert at(3, 0)["BB"] > at(0, 0)["BB"]


def test_high_strikeout_batter_raises_strikeouts(state_factory):
    for balls, strikes in [(0, 0), (1, 2), (3, 0)]:
        s = state_factory(balls=balls, strikes=strikes)
        assert outcome_distribution(s, scaled_profile("batter", "K", 1.5))["K"] > outcome_distribution(s)["K"]


def test_platoon_changes_rates(state_factory):
    same = outcome_distribution(state_factory(batter_hand="L", pitcher_hand="L"))
    opposite = outcome_distribution(state_factory(batter_hand="L", pitcher_hand="R"))
    switch = outcome_distribution(state_factory(batter_hand="B", pitcher_hand="L"))
    assert same["K"] > opposite["K"]
    assert switch == pytest.approx(outcome_distribution(state_factory(batter_hand="R", pitcher_hand="L")))


def test_invalid_profile_rejected():
    with pytest.raises(ValueError):
        PlayerProfile("x", "batter", (0.5, 0.5))
    with pytest.raises(ValueError):
        PlayerProfile("x", "batter", tuple(np.full(9, 0.2).tolist()))
