import numpy as np
import pytest

from skipboard import TableError, load_tables
from skipboard.tables import WP_SHAPE


def test_shapes_and_ranges():
    t = load_tables()
    assert t.re24.shape == (8, 3)
    assert np.isfinite(t.re24).all() and (t.re24 >= 0).all()
    for game_type in ("regular", "postseason"):
        arr = t.wp[game_type]
        assert arr.shape == WP_SHAPE
        assert np.isfinite(arr).all()
        assert ((arr >= 0) & (arr <= 1)).all()


def test_loaded_once():
    assert load_tables() is load_tables()


@pytest.mark.parametrize("game_type", ["regular", "postseason"])
def test_wp_never_decreases_with_diff(game_type):
    arr = load_tables().wp[game_type]
    steps = np.diff(arr, axis=2)  # axis 2 is diff
    assert steps.min() >= -1e-9


def test_re_decreases_with_outs():
    re24 = load_tables().re24
    assert (np.diff(re24, axis=1) < 0).all()


def test_missing_directory_raises(tmp_path):
    with pytest.raises(TableError):
        load_tables(tmp_path)


def test_override_path_reads_same_tables():
    default = load_tables()
    from skipboard.tables import DEFAULT_MODELS_DIR

    override = load_tables(DEFAULT_MODELS_DIR)
    assert np.array_equal(default.re24, override.re24)
