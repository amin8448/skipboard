from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

from skipboard.tables import DEFAULT_MODELS_DIR, TableError
from skipboard.transitions import AdvancementTables, Transition, load_advancement_file

STEAL_DIR = Path("steal")
PITCH_RESULT_FILE = "pitch_result_going_v1.csv"
ADVANCEMENT_FILE = "advancement_going_v1.csv"
INPLAY_FILE = "inplay_outcomes_by_count_v1.csv"
FC_SHARE_FILE = "fc_share_going_v1.csv"
SHARE_TOLERANCE = 1e-4  # saved shares are rounded to 6 decimals

COUNTS = tuple((b, s) for b in range(4) for s in range(3))
GOING_BASE_CODE = 1  # runner on first only
INPLAY_CATEGORIES = ("1B_IF", "1B_OF", "2B", "3B", "HR", "GB_OUT", "AIR_OUT", "ROE", "FC", "OTHER")


def is_offered(balls: int, strikes: int, outs: int) -> bool:
    # At 3-2 with 2 outs the runner goes automatically, so the steal decision is not offered.
    return (balls, strikes) in COUNTS and 0 <= outs <= 2 and not (balls == 3 and strikes == 2 and outs == 2)


@dataclass(frozen=True)
class StealTables:
    pitch_result: dict[tuple[int, int], dict[str, float]]  # (balls, strikes) -> result -> share, runner going
    advancement: AdvancementTables  # base_code 1, runner going, ball in play
    inplay: dict[tuple[int, int], dict[str, float]]  # (balls, strikes) -> in-play category -> share
    fc_share: dict[int, float]  # outs -> FC share of ground balls, runner going

    def pitch_result_shares(self, balls: int, strikes: int, outs: int) -> dict[str, float]:
        if not is_offered(balls, strikes, outs):
            raise ValueError(f"the steal decision is not offered at {balls}-{strikes} with {outs} outs")
        return dict(self.pitch_result[(balls, strikes)])

    def inplay_shares(self, balls: int, strikes: int) -> dict[str, float]:
        if (balls, strikes) not in self.inplay:
            raise ValueError(f"no in-play outcome distribution for {balls}-{strikes}")
        return dict(self.inplay[(balls, strikes)])

    def going_transitions(self, outs: int, category: str) -> tuple[Transition, ...]:
        cell = self.advancement.cells.get((GOING_BASE_CODE, outs, category))
        if not cell:
            raise ValueError(f"no runner-going advancement for {category} with {outs} outs")
        return cell

    def going_fc_share(self, outs: int) -> float:
        if outs not in self.fc_share:
            raise ValueError(f"no going FC share for {outs} outs")
        return self.fc_share[outs]


def load_steal_tables(models_dir: str | Path | None = None) -> StealTables:
    path = Path(models_dir).resolve() if models_dir is not None else DEFAULT_MODELS_DIR
    return _load(path / STEAL_DIR)


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


def _shares_by_count(df: pd.DataFrame, key: str, name: str) -> dict[tuple[int, int], dict[str, float]]:
    out: dict[tuple[int, int], dict[str, float]] = {}
    for (balls, strikes), g in df.groupby(["balls", "strikes"]):
        total = g["share"].sum()
        if abs(total - 1) > SHARE_TOLERANCE:
            raise TableError(f"{name}: shares at {balls}-{strikes} sum to {total}")
        out[(int(balls), int(strikes))] = {str(r[key]): float(r["share"] / total) for _, r in g.iterrows()}
    absent = [f"{b}-{s}" for b, s in COUNTS if (b, s) not in out]
    if absent:
        raise TableError(f"{name} has no rows for counts {absent}")
    return out


@lru_cache(maxsize=None)
def _load(steal_dir: Path) -> StealTables:
    pr = _read(steal_dir / PITCH_RESULT_FILE, ["balls", "strikes", "result", "share"])
    pitch_result = _shares_by_count(pr, "result", PITCH_RESULT_FILE)

    advancement = load_advancement_file(steal_dir / ADVANCEMENT_FILE)
    present = {(o, c) for (b, o, c) in advancement.cells if b == GOING_BASE_CODE}
    expected = {(o, c) for o in range(3) for c in INPLAY_CATEGORIES}
    if present != expected or len(advancement.cells) != len(expected):
        raise TableError(f"{ADVANCEMENT_FILE} must hold base_code {GOING_BASE_CODE} cells for outs 0-2 and {INPLAY_CATEGORIES}")

    ip = _read(steal_dir / INPLAY_FILE, ["balls", "strikes", "category", "share"])
    inplay = _shares_by_count(ip, "category", INPLAY_FILE)

    fc = _read(steal_dir / FC_SHARE_FILE, ["base_code", "outs", "fc_share"])
    fc = fc[fc["base_code"] == GOING_BASE_CODE]
    fc_share = {int(r.outs): float(r.fc_share) for r in fc.itertuples()}
    if sorted(fc_share) != [0, 1, 2] or not all(0 <= v <= 1 for v in fc_share.values()):
        raise TableError(f"{FC_SHARE_FILE} must hold base_code {GOING_BASE_CODE} shares between 0 and 1 for outs 0-2")
    return StealTables(pitch_result=pitch_result, advancement=advancement, inplay=inplay, fc_share=fc_share)
