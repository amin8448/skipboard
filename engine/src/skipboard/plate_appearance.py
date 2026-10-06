from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from skipboard.state import GameState
from skipboard.tables import DEFAULT_MODELS_DIR, TableError

CATEGORIES = ("K", "BB", "HBP", "1B", "2B", "3B", "HR", "GB_OUT", "AIR_OUT", "ROE", "FC", "OTHER")
MATCHUP_CATEGORIES = ("K", "BB", "HBP", "1B", "2B", "3B", "HR", "GB", "AIR_OUT")  # GB = GB_OUT + FC
COUNT_CATEGORIES = MATCHUP_CATEGORIES + ("ROE", "OTHER")  # ROE and OTHER stay at league rates
MATCHUPS = ("L_vs_L", "L_vs_R", "R_vs_L", "R_vs_R")
COUNTS = tuple((b, s) for b in range(4) for s in range(3))
PA_DIR = Path("pa_model")
ROW_SUM_TOLERANCE = 1e-4  # saved probabilities are rounded to 6 decimals

Role = Literal["batter", "pitcher"]


@dataclass(frozen=True)
class PlayerProfile:
    player_id: str
    role: Role
    rates: tuple[float, ...]  # shrunk rates in MATCHUP_CATEGORIES order
    hand: str | None = None
    weighted_pa: float = 0.0

    def __post_init__(self) -> None:
        if len(self.rates) != len(MATCHUP_CATEGORIES):
            raise ValueError(f"rates must have {len(MATCHUP_CATEGORIES)} values, got {len(self.rates)}")
        if min(self.rates) < 0 or abs(sum(self.rates) - 1) > 1e-6:
            raise ValueError("rates must be non-negative and sum to 1")
        if self.role not in ("batter", "pitcher"):
            raise ValueError(f"role must be 'batter' or 'pitcher', got {self.role!r}")


@dataclass(frozen=True)
class PATables:
    league9: np.ndarray  # league rates over MATCHUP_CATEGORIES
    counts11: dict[tuple[int, int], np.ndarray]  # count -> league distribution over COUNT_CATEGORIES
    counts12: dict[tuple[int, int], np.ndarray]  # count -> league distribution over CATEGORIES
    platoon_ratio: dict[str, np.ndarray]  # matchup -> league platoon rate / league rate, MATCHUP_CATEGORIES
    fc_share: np.ndarray  # [base_code, outs]: FC share of ground balls
    single_if_share: np.ndarray  # [base_code, outs]: infield share of singles
    state_ratio: np.ndarray  # [base_code, outs, category]: state share / overall share, MATCHUP_CATEGORIES


def load_pa_tables(models_dir: str | Path | None = None) -> PATables:
    path = Path(models_dir).resolve() if models_dir is not None else DEFAULT_MODELS_DIR
    return _load(path / PA_DIR)


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise TableError(f"table not found: {path}")
    df = pd.read_csv(path)
    if df.isna().any().any():
        raise TableError(f"{path.name} has missing values")
    return df


def _normalize_rows(df: pd.DataFrame, name: str) -> pd.DataFrame:
    sums = df.sum(axis=1)
    if df.isna().any().any() or (sums - 1).abs().max() > ROW_SUM_TOLERANCE:
        raise TableError(f"{name} rows do not sum to 1")
    return df.div(sums, axis=0)


def _merge_gb(df12: pd.DataFrame) -> pd.DataFrame:
    out = df12.rename(columns={"GB_OUT": "GB"}).drop(columns="FC")
    out["GB"] = out["GB"] + df12["FC"]
    return out


def _state_grid(df: pd.DataFrame, column: str, name: str) -> np.ndarray:
    grid = np.full((8, 3), np.nan)
    grid[df["base_code"].to_numpy(), df["outs"].to_numpy()] = df[column].to_numpy()
    if np.isnan(grid).any() or (grid < 0).any() or (grid > 1).any():
        raise TableError(f"{name} must cover all 24 base-out states with shares between 0 and 1")
    grid.setflags(write=False)
    return grid


@lru_cache(maxsize=None)
def _load(pa_dir: Path) -> PATables:
    counts = _read(pa_dir / "count_outcomes_v1.csv")
    counts["count"] = counts["balls"] * 3 + counts["strikes"]  # position in COUNTS
    c12 = counts.pivot(index="count", columns="category", values="p").reindex(index=range(len(COUNTS)), columns=list(CATEGORIES))
    c12 = _normalize_rows(c12, "count_outcomes_v1.csv")
    c11 = _merge_gb(c12)[list(COUNT_CATEGORIES)]
    rates11 = c11.loc[0]
    league9 = rates11[list(MATCHUP_CATEGORIES)] / rates11[list(MATCHUP_CATEGORIES)].sum()

    plat = _read(pa_dir / "platoon_rates_v1.csv")
    plat = plat[plat["group_type"] == "matchup"].pivot(index="group", columns="category", values="p")
    plat = plat.reindex(index=list(MATCHUPS), columns=list(CATEGORIES))
    plat9 = _merge_gb(_normalize_rows(plat, "platoon_rates_v1.csv"))[list(MATCHUP_CATEGORIES)]
    plat9 = plat9.div(plat9.sum(axis=1), axis=0)

    fc = _read(pa_dir / "fc_share_v1.csv")
    split = _read(pa_dir / "single_split_v1.csv")
    sr = _read(pa_dir / "state_ratio_v1.csv")
    state_ratio = np.full((8, 3, len(MATCHUP_CATEGORIES)), np.nan)
    cat_index = sr["category"].map({c: i for i, c in enumerate(MATCHUP_CATEGORIES)})
    if cat_index.isna().any() or len(sr) != state_ratio.size:
        raise TableError("state_ratio_v1.csv must have one row per base-out state and matchup category")
    state_ratio[sr["base_code"].to_numpy(), sr["outs"].to_numpy(), cat_index.to_numpy(dtype=int)] = sr["ratio"].to_numpy()
    if np.isnan(state_ratio).any() or (state_ratio < 0).any():
        raise TableError("state_ratio_v1.csv must cover every cell with non-negative ratios")
    state_ratio.setflags(write=False)
    return PATables(
        league9=league9.to_numpy(),
        counts11={c: c11.iloc[i].to_numpy() for i, c in enumerate(COUNTS)},
        counts12={c: c12.iloc[i].to_numpy() for i, c in enumerate(COUNTS)},
        platoon_ratio={m: (plat9.loc[m] / league9).to_numpy() for m in MATCHUPS},
        fc_share=_state_grid(fc, "fc_share", "fc_share_v1.csv"),
        single_if_share=_state_grid(split, "if_share", "single_split_v1.csv"),
        state_ratio=state_ratio,
    )


def matchup(state: GameState) -> str:
    # Switch hitters bat opposite the pitcher.
    side = state.batter_hand if state.batter_hand in ("L", "R") else ("L" if state.pitcher_hand == "R" else "R")
    return f"{side}_vs_{state.pitcher_hand}"


def matchup_rates(
    state: GameState,
    batter: PlayerProfile | None,
    pitcher: PlayerProfile | None,
    tables: PATables | None = None,
    apply_platoon: bool = True,
    apply_state: bool = True,
) -> np.ndarray:
    t = tables or load_pa_tables()
    b = np.asarray(batter.rates) if batter is not None else t.league9
    q = np.asarray(pitcher.rates) if pitcher is not None else t.league9
    p = b * q / t.league9  # odds-ratio combination
    if apply_platoon:
        p = p * t.platoon_ratio[matchup(state)]
    if apply_state:
        p = p * t.state_ratio[state.base_code, state.outs]
    return p / p.sum()


def outcome_distribution(
    state: GameState,
    batter: PlayerProfile | None = None,
    pitcher: PlayerProfile | None = None,
    tables: PATables | None = None,
    apply_platoon: bool = True,
    apply_state: bool = True,
) -> dict[str, float]:
    t = tables or load_pa_tables()
    p9 = matchup_rates(state, batter, pitcher, t, apply_platoon, apply_state)
    # Count conditioning: league distribution at the count times the matchup's ratio to league.
    # ROE and OTHER have ratio 1, so they enter at their league count-conditional rates.
    ratio = np.concatenate([p9 / t.league9, [1.0, 1.0]])
    p11 = t.counts11[(state.balls, state.strikes)] * ratio
    p11 = p11 / p11.sum()
    probs = dict(zip(COUNT_CATEGORIES, p11.tolist()))
    gb = probs.pop("GB")
    share = float(t.fc_share[state.base_code, state.outs])
    probs["GB_OUT"] = gb * (1 - share)
    probs["FC"] = gb * share
    return {c: probs[c] for c in CATEGORIES}
