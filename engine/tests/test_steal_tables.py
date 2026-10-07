import shutil
from pathlib import Path

import pytest

from skipboard.steal_tables import COUNTS, INPLAY_CATEGORIES, is_offered, load_steal_tables
from skipboard.tables import DEFAULT_MODELS_DIR, TableError


@pytest.fixture
def tables():
    return load_steal_tables()


def test_pitch_result_shares_sum_to_one_at_every_offered_count(tables):
    for balls, strikes in COUNTS:
        for outs in range(3):
            if not is_offered(balls, strikes, outs):
                continue
            shares = tables.pitch_result_shares(balls, strikes, outs)
            assert sum(shares.values()) == pytest.approx(1, abs=1e-12)
            expected = {"ball" if balls < 3 else "ball_four", "strike" if strikes < 2 else "strike_three", "foul", "hit_by_pitch", "in_play"}
            assert set(shares) == expected, (balls, strikes)


def test_full_count_with_two_outs_is_not_offered(tables):
    assert not is_offered(3, 2, 2)
    assert is_offered(3, 2, 0) and is_offered(3, 2, 1)
    with pytest.raises(ValueError):
        tables.pitch_result_shares(3, 2, 2)
    with pytest.raises(ValueError):
        tables.pitch_result_shares(4, 0, 0)


def test_inplay_shares_sum_to_one_at_every_count(tables):
    for balls, strikes in COUNTS:
        shares = tables.inplay_shares(balls, strikes)
        assert sum(shares.values()) == pytest.approx(1, abs=1e-12)
        assert set(shares) <= set(INPLAY_CATEGORIES)


def test_advancement_going_has_thirty_valid_cells(tables):
    assert len(tables.advancement.cells) == 30
    for outs in range(3):
        for category in INPLAY_CATEGORIES:
            options = tables.going_transitions(outs, category)
            assert sum(t.p for t in options) == pytest.approx(1, abs=1e-12)
            assert all(outs <= t.outs_post <= 3 and 0 <= t.base_post <= 7 and t.runs >= 0 for t in options)
    with pytest.raises(ValueError):
        tables.going_transitions(0, "K")


def test_going_fc_share_by_outs(tables):
    assert sorted(tables.fc_share) == [0, 1, 2]
    assert all(0 <= tables.going_fc_share(o) <= 1 for o in range(3))
    with pytest.raises(ValueError):
        tables.going_fc_share(3)


def copy_tables(tmp_path: Path) -> Path:
    models = tmp_path / "models"
    shutil.copytree(DEFAULT_MODELS_DIR / "steal", models / "steal")
    return models


def test_shares_not_summing_to_one_raise(tmp_path):
    models = copy_tables(tmp_path)
    path = models / "steal" / "pitch_result_going_v1.csv"
    lines = path.read_text().splitlines()
    head, first = lines[0], lines[1].split(",")
    first[head.split(",").index("share")] = "0.9"
    path.write_text("\n".join([head, ",".join(first)] + lines[2:]) + "\n")
    with pytest.raises(TableError, match="sum to"):
        load_steal_tables(models)


def test_missing_count_raises(tmp_path):
    models = copy_tables(tmp_path)
    path = models / "steal" / "inplay_outcomes_by_count_v1.csv"
    lines = path.read_text().splitlines()
    path.write_text("\n".join([lines[0]] + [ln for ln in lines[1:] if not ln.startswith("2,1,")]) + "\n")
    with pytest.raises(TableError, match="no rows for counts"):
        load_steal_tables(models)
