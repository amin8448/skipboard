"""Steal success model: the probability the runner is safe when the catcher throws to second.

A season outside the fitted levels uses the latest fitted season's effect and fill values.
Missing attributes are filled with the season mean; pitcher attributes with the mean for the pitcher's hand.
"""

import json
import math
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Literal

import pandas as pd

# engine/src/skipboard/steal_model.py -> project root is four levels up
DEFAULT_STEAL_MODEL_DIR = Path(__file__).resolve().parents[3] / "data" / "derived" / "steal"
ENV_VAR = "SKIPBOARD_STEAL_MODEL_DIR"
COEFFICIENTS_FILE = "steal_success_v1_coefficients.csv"
META_FILE = "steal_success_v1_meta.json"

PitchResult = Literal["ball", "strike", "strike_three"]
PITCH_RESULTS = ("ball", "strike", "strike_three")
EXPECTED_LEVELS = {
    "pitch_result": ["ball", "strike", "strike_three"],
    "pickoff_throws": ["0", "1", "2+"],
    "pitcher_hand": ["R", "L"],
    "bat_side": ["R", "L"],
}
FIXED_TERMS = (
    "intercept",
    "pitch_result=strike",
    "pitch_result=strike_three",
    "count=3-2",
    "pickoff_throws=1",
    "pickoff_throws=2+",
    "pitchout",
    "pitcher_hand=L",
    "bat_side=L",
)
# Continuous term -> (profile, attribute)
CONTINUOUS_TERMS = {
    "sprint_speed": ("runner", "sprint_speed"),
    "runner_aggressiveness": ("runner", "aggressiveness"),
    "pop_time": ("catcher", "pop_time"),
    "pitcher_primary_lead": ("pitcher", "primary_lead"),
    "pitcher_lead_gained": ("pitcher", "lead_gained"),
}


class StealModelError(ValueError):
    pass


@dataclass(frozen=True, kw_only=True)
class RunnerProfile:
    sprint_speed: float | None = None  # ft/s
    aggressiveness: float | None = None  # shrunk share of pitches the runner goes on


@dataclass(frozen=True, kw_only=True)
class CatcherProfile:
    pop_time: float | None = None  # seconds to second base


@dataclass(frozen=True, kw_only=True)
class PitcherProfile:
    hand: Literal["L", "R"]
    primary_lead: float | None = None  # ft allowed
    lead_gained: float | None = None  # ft gained by the runner between first move and release

    def __post_init__(self) -> None:
        if self.hand not in ("L", "R"):
            raise ValueError(f"pitcher hand must be L or R, got {self.hand!r}")


@dataclass(frozen=True)
class StealSuccessModel:
    estimates: dict[str, float]
    centers: dict[str, float]  # continuous terms only
    fitted_seasons: tuple[int, ...]
    fill_values_by_season: dict[int, dict[str, dict[str, float]]] = field(repr=False)  # season -> pitcher hand -> term -> fill

    @classmethod
    def from_directory(cls, directory: str | Path | None = None) -> "StealSuccessModel":
        return _load(_resolve(directory))

    def season_used(self, season: int) -> int:
        return season if season in self.fitted_seasons else self.fitted_seasons[-1]

    def fill_values(self, season: int, pitcher_hand: Literal["L", "R"]) -> dict[str, float]:
        # Runner and catcher fills are the season means; pitcher fills depend on the pitcher's hand.
        if pitcher_hand not in ("L", "R"):
            raise ValueError(f"pitcher hand must be L or R, got {pitcher_hand!r}")
        return dict(self.fill_values_by_season[self.season_used(season)][pitcher_hand])

    def p_safe(
        self,
        pitch_result: PitchResult,
        balls: int,
        strikes: int,
        pickoff_throws: int,
        bat_side: Literal["L", "R", "B"],
        season: int,
        runner: RunnerProfile,
        catcher: CatcherProfile,
        pitcher: PitcherProfile,
    ) -> float:
        _check_pitch(pitch_result, balls, strikes)
        if pickoff_throws < 0:
            raise ValueError(f"pickoff_throws cannot be negative, got {pickoff_throws}")
        if bat_side not in ("L", "R", "B"):
            raise ValueError(f"bat_side must be L, R or B, got {bat_side!r}")
        side = bat_side if bat_side != "B" else ("L" if pitcher.hand == "R" else "R")  # switch hitters bat opposite the pitcher
        used = self.season_used(season)
        e = self.estimates
        eta = e["intercept"]
        eta += e["pitch_result=strike"] * (pitch_result == "strike") + e["pitch_result=strike_three"] * (pitch_result == "strike_three")
        eta += e["count=3-2"] * (balls == 3 and strikes == 2)
        eta += e["pickoff_throws=1"] * (pickoff_throws == 1) + e["pickoff_throws=2+"] * (pickoff_throws >= 2)
        eta += e["pitcher_hand=L"] * (pitcher.hand == "L") + e["bat_side=L"] * (side == "L")
        eta += e.get(f"season={used}", 0.0)  # pitchout is always off
        fills = self.fill_values_by_season[used][pitcher.hand]
        profiles = {"runner": runner, "catcher": catcher, "pitcher": pitcher}
        for term, center in self.centers.items():
            role, attr = CONTINUOUS_TERMS[term]
            value = getattr(profiles[role], attr)
            eta += e[term] * ((fills[term] if value is None else value) - center)
        return 1.0 / (1.0 + math.exp(-eta))


def _check_pitch(pitch_result: str, balls: int, strikes: int) -> None:
    if pitch_result not in PITCH_RESULTS:
        raise ValueError(f"pitch_result must be one of {PITCH_RESULTS}, got {pitch_result!r}")
    if not (0 <= balls <= 3 and 0 <= strikes <= 2):
        raise ValueError(f"count must be within 0-3 balls and 0-2 strikes, got {balls}-{strikes}")
    # A throw follows a ball (not ball four), a strike (not strike three) or strike three.
    if pitch_result == "ball" and balls == 3:
        raise ValueError(f"a ball at {balls}-{strikes} is ball four, which is not a throw situation")
    if pitch_result == "strike" and strikes == 2:
        raise ValueError(f"a strike at {balls}-{strikes} is strike three; use pitch_result='strike_three'")
    if pitch_result == "strike_three" and strikes != 2:
        raise ValueError(f"strike three needs two strikes before the pitch, got {balls}-{strikes}")


def _resolve(directory: str | Path | None) -> Path:
    if directory is not None:
        return Path(directory).resolve()
    if os.environ.get(ENV_VAR):
        return Path(os.environ[ENV_VAR]).resolve()
    return DEFAULT_STEAL_MODEL_DIR


def _strip(level: object) -> str:
    return str(level).replace(" (reference)", "")


@lru_cache(maxsize=None)
def _load(directory: Path) -> StealSuccessModel:
    coef_path, meta_path = directory / COEFFICIENTS_FILE, directory / META_FILE
    for path in (coef_path, meta_path):
        if not path.exists():
            raise StealModelError(f"steal model file not found: {path}")
    coef = pd.read_csv(coef_path)
    missing_cols = {"term", "estimate", "centering_mean"} - set(coef.columns)
    if missing_cols:
        raise StealModelError(f"{coef_path.name} is missing columns {sorted(missing_cols)}")
    meta = json.loads(meta_path.read_text())

    levels = {k: [_strip(v) for v in meta["factor_levels"][k]] for k in EXPECTED_LEVELS}
    for name, expected in EXPECTED_LEVELS.items():
        if levels[name] != expected:
            raise StealModelError(f"{meta_path.name}: factor levels for {name} are {levels[name]}, expected {expected}")
    seasons = tuple(sorted(int(_strip(s)) for s in meta["factor_levels"]["season"]))
    season_terms = {f"season={s}" for s in seasons[1:]}

    estimates: dict[str, float] = {}
    centers: dict[str, float] = {}
    for row in coef.itertuples():
        term = str(row.term)
        if term not in FIXED_TERMS and term not in CONTINUOUS_TERMS and term not in season_terms:
            raise StealModelError(f"{coef_path.name}: unknown term {term!r}")
        estimates[term] = float(row.estimate)
        if term in CONTINUOUS_TERMS:
            if pd.isna(row.centering_mean):
                raise StealModelError(f"{coef_path.name}: continuous term {term!r} has no centering mean")
            centers[term] = float(row.centering_mean)
        constraint = getattr(row, "constraint", "none")
        if (constraint == ">= 0" and row.estimate < 0) or (constraint == "<= 0" and row.estimate > 0):
            raise StealModelError(f"{coef_path.name}: {term!r} = {row.estimate} violates its constraint {constraint}")
    absent = [t for t in (*FIXED_TERMS, *sorted(season_terms)) if t not in estimates]
    if absent:
        raise StealModelError(f"{coef_path.name} is missing terms {absent}")

    fills: dict[int, dict[str, dict[str, float]]] = {}
    for season in seasons:
        values = meta["fill_values_by_season"].get(str(season))
        if values is None or any(term not in values for term in centers):
            raise StealModelError(f"{meta_path.name}: fill values for season {season} must cover {sorted(centers)}")
        fills[season] = {"L": {}, "R": {}}
        for term in centers:
            value = values[term]
            if CONTINUOUS_TERMS[term][0] == "pitcher":
                if not isinstance(value, dict) or set(value) != {"L", "R"}:
                    raise StealModelError(f"{meta_path.name}: {term} fill for season {season} must be given by pitcher hand (L and R)")
                for hand in ("L", "R"):
                    fills[season][hand][term] = float(value[hand])
            else:
                if isinstance(value, dict):
                    raise StealModelError(f"{meta_path.name}: {term} fill for season {season} must be a single number")
                for hand in ("L", "R"):
                    fills[season][hand][term] = float(value)
    return StealSuccessModel(estimates=estimates, centers=centers, fitted_seasons=seasons, fill_values_by_season=fills)


def load_steal_model(directory: str | Path | None = None) -> StealSuccessModel:
    return StealSuccessModel.from_directory(directory)
