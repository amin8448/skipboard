import pytest

from skipboard.plate_appearance import CATEGORIES, load_pa_tables
from skipboard.transitions import Transition, transitions, transitions_for


@pytest.mark.parametrize("outs", range(3))
def test_bases_loaded_home_run_scores_four(outs):
    assert transitions_for(7, outs, "HR") == (Transition(0, outs, 4, 1.0),)


def test_every_cell_sums_to_one():
    for base in range(8):
        for outs in range(3):
            for category in CATEGORIES:
                options = transitions_for(base, outs, category)
                if category == "FC" and base == 0:
                    assert options == ()
                    continue
                assert options, (base, outs, category)
                assert sum(t.p for t in options) == pytest.approx(1, abs=1e-12)
                for t in options:
                    assert 0 <= t.base_post <= 7 and outs <= t.outs_post <= 3 and t.runs >= 0 and t.p > 0


def test_single_mixes_infield_and_outfield():
    share = load_pa_tables().single_if_share[2, 0]
    scores = lambda cat: sum(t.p for t in transitions_for(2, 0, cat) if t.runs >= 1)  # noqa: E731
    mixed = scores("1B")
    assert mixed == pytest.approx(share * scores("1B_IF") + (1 - share) * scores("1B_OF"))
    assert scores("1B_IF") < mixed < scores("1B_OF")


def test_state_wrapper(state_factory):
    s = state_factory(base_code=5, outs=1)
    assert transitions(s, "BB") == transitions_for(5, 1, "BB")
