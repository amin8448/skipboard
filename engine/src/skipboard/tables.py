from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

# engine/src/skipboard/tables.py -> project root is four levels up
DEFAULT_MODELS_DIR = Path(__file__).resolve().parents[3] / "models"

RE24_FILE = Path("run_expectancy") / "re24_v1.csv"
WP_FILES = {
    "regular": Path("win_probability") / "wp_regular_v1.csv",
    "postseason": Path("win_probability") / "wp_postseason_v1.csv",
}

N_BASES, N_OUTS = 8, 3
WP_INNINGS = 12
DIFF_MAX = 10
WP_SHAPE = (WP_INNINGS, 2, 2 * DIFF_MAX + 1, N_BASES, N_OUTS)  # inning-1, half, diff+10, base_code, outs


class TableError(ValueError):
    pass


@dataclass(frozen=True)
class Tables:
    re24: np.ndarray  # [base_code, outs]
    wp: dict[str, np.ndarray]  # game_type -> WP_SHAPE array of wp_bat


def load_tables(models_dir: str | Path | None = None) -> Tables:
    path = Path(models_dir).resolve() if models_dir is not None else DEFAULT_MODELS_DIR
    return _load(path)


@lru_cache(maxsize=None)
def _load(models_dir: Path) -> Tables:
    re24 = _load_re24(models_dir / RE24_FILE)
    wp = {game_type: _load_wp(models_dir / file) for game_type, file in WP_FILES.items()}
    return Tables(re24=re24, wp=wp)


def _read(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise TableError(f"table not found: {path}")
    df = pd.read_csv(path)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise TableError(f"{path.name} is missing columns {missing}")
    if df[columns].isna().any().any():
        raise TableError(f"{path.name} has missing values")
    return df


def _fill(df: pd.DataFrame, keys: list[str], value: str, shape: tuple[int, ...], path: Path) -> np.ndarray:
    expected = int(np.prod(shape))
    if len(df) != expected:
        raise TableError(f"{path.name} has {len(df)} rows, expected {expected}")
    idx = tuple(df[k].to_numpy() for k in keys)
    for k, size in zip(keys, shape):
        if df[k].min() < 0 or df[k].max() >= size:
            raise TableError(f"{path.name} column {k} is out of range")
    out = np.full(shape, np.nan)
    out[idx] = df[value].to_numpy(dtype=float)
    if np.isnan(out).any():
        raise TableError(f"{path.name} does not cover every cell (duplicate rows?)")
    out.setflags(write=False)
    return out


def _load_re24(path: Path) -> np.ndarray:
    df = _read(path, ["base_code", "outs", "re"])
    return _fill(df, ["base_code", "outs"], "re", (N_BASES, N_OUTS), path)


def _load_wp(path: Path) -> np.ndarray:
    df = _read(path, ["inning", "half", "diff", "base_code", "outs", "wp_bat"])
    df = df.assign(inning=df["inning"] - 1, diff=df["diff"] + DIFF_MAX)
    arr = _fill(df, ["inning", "half", "diff", "base_code", "outs"], "wp_bat", WP_SHAPE, path)
    if (arr < 0).any() or (arr > 1).any():
        raise TableError(f"{path.name} has win probabilities outside 0 to 1")
    return arr
