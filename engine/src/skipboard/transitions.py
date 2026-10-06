from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import pandas as pd

from skipboard.plate_appearance import PA_DIR, PATables, load_pa_tables
from skipboard.state import GameState
from skipboard.tables import DEFAULT_MODELS_DIR, TableError

ADVANCEMENT_FILE = "advancement_v1.csv"
CELL_SUM_TOLERANCE = 1e-4  # saved probabilities are rounded to 6 decimals


class Transition(NamedTuple):
    base_post: int
    outs_post: int
    runs: int
    p: float


@dataclass(frozen=True)
class AdvancementTables:
    cells: dict[tuple[int, int, str], tuple[Transition, ...]]  # (base_code, outs, category) -> transitions


def load_advancement(models_dir: str | Path | None = None) -> AdvancementTables:
    path = Path(models_dir).resolve() if models_dir is not None else DEFAULT_MODELS_DIR
    return _load(path / PA_DIR / ADVANCEMENT_FILE)


@lru_cache(maxsize=None)
def _load(path: Path) -> AdvancementTables:
    if not path.exists():
        raise TableError(f"table not found: {path}")
    df = pd.read_csv(path)
    cells: dict[tuple[int, int, str], tuple[Transition, ...]] = {}
    for (base, outs, category), g in df.groupby(["base_pre", "outs_pre", "category"]):
        total = g["p"].sum()
        if abs(total - 1) > CELL_SUM_TOLERANCE:
            raise TableError(f"{path.name}: probabilities for {(base, outs, category)} sum to {total}")
        if (g["outs_post"] < outs).any() or (g["outs_post"] > 3).any():
            raise TableError(f"{path.name}: outs go backward or past 3 in {(base, outs, category)}")
        cells[(int(base), int(outs), str(category))] = tuple(
            Transition(int(r.base_post), int(r.outs_post), int(r.runs), float(r.p) / total) for r in g.itertuples()
        )
    return AdvancementTables(cells)


def transitions_for(
    base_code: int,
    outs: int,
    category: str,
    tables: AdvancementTables | None = None,
    pa_tables: PATables | None = None,
) -> tuple[Transition, ...]:
    t = tables or load_advancement()
    if category != "1B":
        return t.cells.get((base_code, outs, category), ())
    # Singles mix the infield and outfield tables by the league infield share for the state.
    share = float((pa_tables or load_pa_tables()).single_if_share[base_code, outs])
    mixed: dict[tuple[int, int, int], float] = defaultdict(float)
    for part, weight in (("1B_IF", share), ("1B_OF", 1 - share)):
        for tr in t.cells.get((base_code, outs, part), ()):
            mixed[(tr.base_post, tr.outs_post, tr.runs)] += weight * tr.p
    return tuple(Transition(b, o, r, p) for (b, o, r), p in sorted(mixed.items()))


def transitions(state: GameState, category: str, tables: AdvancementTables | None = None) -> tuple[Transition, ...]:
    return transitions_for(state.base_code, state.outs, category, tables)
