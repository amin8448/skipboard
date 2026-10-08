"""Send model: a runner on second, a single to the outfield, and the third-base coach's call to send or hold.

A bivariate probit with sample selection, fitted in research/fit_send_success.py:
- p_send = Phi(Z gamma): the probability a coach sends the runner.
- p_safe = Phi(X beta): the probability the runner is safe at home if sent, for any play (the unconditional
  probability; the selection correlation rho is parsed but not needed for it).

Geometry follows the research convention: the fielding point (x_feet, y_feet) is in feet with home plate at the
origin, x positive toward right field and y toward center field. The spray angle is atan2(x, y) in degrees
(negative toward left field). spray_toward_line is the angle toward the fielding outfielder's foul line: minus the
spray angle for the left fielder, the spray angle for the right fielder, and its absolute value for the center
fielder (toward either gap). Positioning starts are a distance from home plate and an angle in the same convention.

Missing attributes are filled from the meta file: arm strength by season and position, sprint speed by season,
the fielder's start by season, position and batter side, launch speed and angle by season and batted-ball type
(popups count as fly balls), and on-deck quality with 0 (league average). The send model has no season terms;
a season outside the fitted ones uses the latest fitted season's fill values.
"""

import json
import math
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Literal, NamedTuple

import numpy as np
import pandas as pd

from skipboard.steal_model import RunnerProfile

# engine/src/skipboard/send_model.py -> project root is four levels up
DEFAULT_SEND_MODEL_DIR = Path(__file__).resolve().parents[3] / "data" / "derived" / "send"
ENV_VAR = "SKIPBOARD_SEND_MODEL_DIR"
COEFFICIENTS_FILE = "send_model_v1_coefficients.csv"
META_FILE = "send_model_v1_meta.json"
BOOTSTRAP_FILE = "send_model_v1_bootstrap.csv"
BATTER_ADVANCE_FILE = "send_batter_advance_v1.csv"
SHARE_TOLERANCE = 1e-4  # saved shares are rounded to 6 decimals

BBType = Literal["ground_ball", "line_drive", "fly_ball", "popup"]
BB_TYPES = ("ground_ball", "line_drive", "fly_ball", "popup")
POSITION_LABEL = {7: "LF", 8: "CF", 9: "RF"}
SAFE_CONTINUOUS = ("fielding_distance", "spray_toward_line", "fielder_distance", "launch_speed", "launch_angle", "arm_avg", "runner_sprint_speed")
SEND_CONTINUOUS = SAFE_CONTINUOUS + ("on_deck_quality",)
FACTOR_TERMS = ("pos=CF", "pos=RF", "bb=line_drive", "bb=fly_ball", "bat_side=L", "outs=1", "outs=2")
SCORE_TERMS = tuple(f"score={s:+d}" for s in (-4, -3, -2, -1, 1, 2, 3, 4))
INNING_TERMS = ("inning=4-6", "inning=7-8", "inning=9+")
SAFE_TERMS = ("intercept",) + SAFE_CONTINUOUS + FACTOR_TERMS
SEND_TERMS = ("intercept",) + SEND_CONTINUOUS + FACTOR_TERMS + SCORE_TERMS + INNING_TERMS + ("late_close",)
EXPECTED_LEVELS = {
    "fielder_pos": ["LF", "CF", "RF"],
    "bb_type": ["ground_ball", "line_drive", "fly_ball"],
    "bat_side": ["R", "L"],
    "outs": ["0", "1", "2"],
    "inning_group": ["1-3", "4-6", "7-8", "9+"],
}
BATTER_ADVANCE_RESULTS = ("scored", "out at home", "held at third")
BATTER_ADVANCES = ("first", "second", "out")


class SendModelError(ValueError):
    pass


@dataclass(frozen=True, kw_only=True)
class BattedBall:
    x_feet: float  # fielding point, feet from home plate toward right field
    y_feet: float  # feet from home plate toward center field
    bb_type: BBType
    fielder_pos: Literal[7, 8, 9]
    launch_speed: float | None = None  # mph
    launch_angle: float | None = None  # degrees

    def __post_init__(self) -> None:
        if self.bb_type not in BB_TYPES:
            raise ValueError(f"bb_type must be one of {BB_TYPES}, got {self.bb_type!r}")
        if self.fielder_pos not in POSITION_LABEL:
            raise ValueError(f"fielder_pos must be 7, 8 or 9, got {self.fielder_pos!r}")

    @property
    def model_bb_type(self) -> str:
        return "fly_ball" if self.bb_type == "popup" else self.bb_type


@dataclass(frozen=True, kw_only=True)
class FielderProfile:
    arm_avg: float | None = None  # mph, average arm strength at the position


@dataclass(frozen=True, kw_only=True)
class PositioningStart:
    start_distance: float | None = None  # feet from home plate
    start_angle: float | None = None  # degrees, negative toward left field


@dataclass(frozen=True, kw_only=True)
class SendContext:
    outs: int
    score_diff: int  # batting team minus fielding team
    inning: int
    bat_side: Literal["L", "R"]  # the side the batter hits from (a switch hitter's side in this plate appearance)
    on_deck_quality: float | None = None  # runs against a league-average batter; None is league average (0)

    def __post_init__(self) -> None:
        if not 0 <= self.outs <= 2:
            raise ValueError(f"outs must be 0 to 2, got {self.outs}")
        if self.inning < 1:
            raise ValueError(f"inning must be at least 1, got {self.inning}")
        if self.bat_side not in ("L", "R"):
            raise ValueError(f"bat_side must be L or R, got {self.bat_side!r}")

    @property
    def late_close(self) -> bool:
        return self.inning >= 7 and abs(self.score_diff) <= 1

    @property
    def score_level(self) -> int:
        return max(-4, min(4, self.score_diff))

    @property
    def inning_group(self) -> str:
        return "1-3" if self.inning <= 3 else "4-6" if self.inning <= 6 else "7-8" if self.inning <= 8 else "9+"


class DerivedCovariates(NamedTuple):
    fielding_distance: float  # feet from home plate to the fielding point (also the throw distance)
    spray_toward_line: float  # degrees toward the fielding outfielder's foul line
    fielder_distance: float  # feet from the fielder's start to the fielding point


@dataclass(frozen=True)
class SendFills:
    arm_avg: dict[tuple[int, int], float]  # (season, position) -> mph
    sprint_speed: dict[int, float]  # season -> ft/s
    start: dict[tuple[int, int, str], tuple[float, float]]  # (season, position, batter side) -> (distance, angle)
    launch: dict[tuple[int, str], tuple[float, float]]  # (season, batted-ball type) -> (speed, angle)
    on_deck_quality: float


@dataclass(frozen=True)
class SendModel:
    safe: dict[str, float]
    send: dict[str, float]
    rho: float
    centers: dict[str, float]  # continuous terms, shared by both equations
    fitted_seasons: tuple[int, ...]
    fills: SendFills = field(repr=False)
    safe_bootstrap: np.ndarray = field(repr=False)  # replicates x SAFE_TERMS
    rho_bootstrap: np.ndarray = field(repr=False)

    @classmethod
    def from_directory(cls, directory: str | Path | None = None) -> "SendModel":
        return _load(_resolve(directory))

    def season_used(self, season: int) -> int:
        return season if season in self.fitted_seasons else self.fitted_seasons[-1]

    def arm_fill(self, season: int, fielder_pos: int) -> float:
        return self.fills.arm_avg[(self.season_used(season), fielder_pos)]

    def sprint_fill(self, season: int) -> float:
        return self.fills.sprint_speed[self.season_used(season)]

    def start_fill(self, season: int, fielder_pos: int, bat_side: str) -> tuple[float, float]:
        return self.fills.start[(self.season_used(season), fielder_pos, bat_side)]

    def launch_fill(self, season: int, bb_type: str) -> tuple[float, float]:
        return self.fills.launch[(self.season_used(season), "fly_ball" if bb_type == "popup" else bb_type)]

    def derived_covariates(self, ball: BattedBall, start: PositioningStart, bat_side: Literal["L", "R"], season: int) -> DerivedCovariates:
        # season is needed only when the start has missing values (the league start for that season, position and side).
        fill_distance, fill_angle = self.start_fill(season, ball.fielder_pos, bat_side)
        distance = fill_distance if start.start_distance is None else start.start_distance
        angle = fill_angle if start.start_angle is None else start.start_angle
        spray = math.degrees(math.atan2(ball.x_feet, ball.y_feet))
        toward_line = -spray if ball.fielder_pos == 7 else spray if ball.fielder_pos == 9 else abs(spray)
        start_x = distance * math.sin(math.radians(angle))
        start_y = distance * math.cos(math.radians(angle))
        return DerivedCovariates(
            fielding_distance=math.hypot(ball.x_feet, ball.y_feet),
            spray_toward_line=toward_line,
            fielder_distance=math.hypot(ball.x_feet - start_x, ball.y_feet - start_y),
        )

    def p_safe(self, ball: BattedBall, runner: RunnerProfile, fielder: FielderProfile, start: PositioningStart, context: SendContext, season: int) -> float:
        values = self._values(ball, runner, fielder, start, context, season)
        return _phi(sum(self.safe[t] * values[t] for t in SAFE_TERMS))

    def p_safe_bootstrap(
        self, ball: BattedBall, runner: RunnerProfile, fielder: FielderProfile, start: PositioningStart, context: SendContext, season: int
    ) -> np.ndarray:
        values = self._values(ball, runner, fielder, start, context, season)
        eta = self.safe_bootstrap @ np.array([values[t] for t in SAFE_TERMS])
        return np.array([_phi(float(e)) for e in eta])

    def p_send(self, ball: BattedBall, runner: RunnerProfile, fielder: FielderProfile, start: PositioningStart, context: SendContext, season: int) -> float:
        values = self._values(ball, runner, fielder, start, context, season)
        return _phi(sum(self.send[t] * values[t] for t in SEND_TERMS))

    def _values(
        self, ball: BattedBall, runner: RunnerProfile, fielder: FielderProfile, start: PositioningStart, context: SendContext, season: int
    ) -> dict[str, float]:
        geo = self.derived_covariates(ball, start, context.bat_side, season)
        fill_speed, fill_angle = self.launch_fill(season, ball.model_bb_type)
        raw = {
            "fielding_distance": geo.fielding_distance,
            "spray_toward_line": geo.spray_toward_line,
            "fielder_distance": geo.fielder_distance,
            "launch_speed": fill_speed if ball.launch_speed is None else ball.launch_speed,
            "launch_angle": fill_angle if ball.launch_angle is None else ball.launch_angle,
            "arm_avg": self.arm_fill(season, ball.fielder_pos) if fielder.arm_avg is None else fielder.arm_avg,
            "runner_sprint_speed": self.sprint_fill(season) if runner.sprint_speed is None else runner.sprint_speed,
            "on_deck_quality": self.fills.on_deck_quality if context.on_deck_quality is None else context.on_deck_quality,
        }
        values = {term: raw[term] - center for term, center in self.centers.items()}
        values["intercept"] = 1.0
        values["pos=CF"] = float(ball.fielder_pos == 8)
        values["pos=RF"] = float(ball.fielder_pos == 9)
        values["bb=line_drive"] = float(ball.model_bb_type == "line_drive")
        values["bb=fly_ball"] = float(ball.model_bb_type == "fly_ball")
        values["bat_side=L"] = float(context.bat_side == "L")
        values["outs=1"] = float(context.outs == 1)
        values["outs=2"] = float(context.outs == 2)
        for term in SCORE_TERMS:
            values[term] = float(context.score_level == int(term.split("=")[1]))
        for term in INNING_TERMS:
            values[term] = float(context.inning_group == term.split("=")[1])
        values["late_close"] = float(context.late_close)
        return values


def _phi(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _resolve(directory: str | Path | None) -> Path:
    if directory is not None:
        return Path(directory).resolve()
    if os.environ.get(ENV_VAR):
        return Path(os.environ[ENV_VAR]).resolve()
    return DEFAULT_SEND_MODEL_DIR


def _strip(level: object) -> str:
    return re.sub(r" \(.*\)$", "", str(level))


def _key(text: str, parts: int) -> list[str]:
    pieces = text.split(" ")
    if len(pieces) != parts:
        raise SendModelError(f"fill key {text!r} should have {parts} parts")
    return pieces


@lru_cache(maxsize=None)
def _load(directory: Path) -> SendModel:
    coef_path, meta_path, boot_path = directory / COEFFICIENTS_FILE, directory / META_FILE, directory / BOOTSTRAP_FILE
    for path in (coef_path, meta_path, boot_path):
        if not path.exists():
            raise SendModelError(f"send model file not found: {path}")
    coef = pd.read_csv(coef_path)
    missing_cols = {"equation", "term", "estimate", "constraint", "center"} - set(coef.columns)
    if missing_cols:
        raise SendModelError(f"{coef_path.name} is missing columns {sorted(missing_cols)}")
    meta = json.loads(meta_path.read_text())

    levels = meta.get("factor_levels", {})
    for name, expected in EXPECTED_LEVELS.items():
        found = [_strip(v) for v in levels.get(name, [])]
        if found != expected:
            raise SendModelError(f"{meta_path.name}: factor levels for {name} are {found}, expected {expected}")

    safe: dict[str, float] = {}
    send: dict[str, float] = {}
    centers: dict[str, float] = {}
    rho: float | None = None
    for row in coef.itertuples():
        equation, term, estimate = str(row.equation), str(row.term), float(row.estimate)
        if equation == "rho":
            if term != "rho" or not -1 < estimate < 1:
                raise SendModelError(f"{coef_path.name}: bad rho row ({term!r}, {estimate})")
            rho = estimate
            continue
        known = {"safe": SAFE_TERMS, "send": SEND_TERMS}.get(equation)
        if known is None:
            raise SendModelError(f"{coef_path.name}: unknown equation {equation!r}")
        if term not in known:
            raise SendModelError(f"{coef_path.name}: unknown term {term!r} in the {equation} equation")
        (safe if equation == "safe" else send)[term] = estimate
        if term in SEND_CONTINUOUS:
            if pd.isna(row.center):
                raise SendModelError(f"{coef_path.name}: continuous term {term!r} has no center")
            if term in centers and centers[term] != float(row.center):
                raise SendModelError(f"{coef_path.name}: {term!r} has different centers in the two equations")
            centers[term] = float(row.center)
        constraint = row.constraint
        if (constraint == ">= 0" and estimate < 0) or (constraint == "<= 0" and estimate > 0):
            raise SendModelError(f"{coef_path.name}: {term!r} = {estimate} violates its constraint {constraint}")
    for equation, terms, found in (("safe", SAFE_TERMS, safe), ("send", SEND_TERMS, send)):
        absent = [t for t in terms if t not in found]
        if absent:
            raise SendModelError(f"{coef_path.name} is missing {equation} terms {absent}")
    if rho is None:
        raise SendModelError(f"{coef_path.name} has no rho row")

    boot = pd.read_csv(boot_path)
    expected_header = ["replicate"] + [f"{e}:{t}" for e, t in zip(coef["equation"], coef["term"]) if e != "rho"] + ["rho"]
    if list(boot.columns) != expected_header:
        absent = [c for c in expected_header if c not in boot.columns]
        extra = [c for c in boot.columns if c not in expected_header]
        raise SendModelError(f"{boot_path.name}: header does not match the coefficient terms (missing {absent}, unexpected {extra})")
    if boot.isna().any().any() or len(boot) == 0:
        raise SendModelError(f"{boot_path.name} is empty or has missing values")
    safe_boot = boot[[f"safe:{t}" for t in SAFE_TERMS]].to_numpy(float)

    seasons = tuple(sorted(int(s) for s in meta["seasons"]["final_fit"]))
    fv = meta["fill_values"]
    arm: dict[tuple[int, int], float] = {}
    sprint: dict[int, float] = {}
    start: dict[tuple[int, int, str], tuple[float, float]] = {}
    launch: dict[tuple[int, str], tuple[float, float]] = {}
    try:
        for k, v in fv["arm_strength_by_season_position"].items():
            s, p = _key(k, 2)
            arm[(int(s), _position(p))] = float(v["arm_avg"])
        for k, v in fv["sprint_speed_by_season"].items():
            sprint[int(k)] = float(v)
        for k, v in fv["positioning_start_by_season_position_side"].items():
            s, p, side = _key(k, 3)
            start[(int(s), _position(p), side)] = (float(v["start_distance"]), float(v["start_angle"]))
        for k, v in meta["launch_fill_values"].items():
            s, b = _key(k, 2)
            launch[(int(s), b)] = (float(v["launch_speed"]), float(v["launch_angle"]))
        on_deck = float(fv["on_deck_quality"])
    except KeyError as e:
        raise SendModelError(f"{meta_path.name}: fill values missing {e}") from e
    for season in seasons:
        needed = ([(season, p) for p in POSITION_LABEL], [season], [(season, p, side) for p in POSITION_LABEL for side in ("L", "R")],
                  [(season, b) for b in EXPECTED_LEVELS["bb_type"]])
        for table, keys in zip((arm, sprint, start, launch), needed):
            absent_keys = [k for k in keys if k not in table]
            if absent_keys:
                raise SendModelError(f"{meta_path.name}: fill values missing for {absent_keys}")
    fills = SendFills(arm_avg=arm, sprint_speed=sprint, start=start, launch=launch, on_deck_quality=on_deck)
    return SendModel(safe=safe, send=send, rho=rho, centers=centers, fitted_seasons=seasons, fills=fills,
                     safe_bootstrap=safe_boot, rho_bootstrap=boot["rho"].to_numpy(float))


def _position(label: str) -> int:
    for pos, name in POSITION_LABEL.items():
        if name == label:
            return pos
    raise SendModelError(f"unknown position label {label!r} in the fill values")


def load_send_model(directory: str | Path | None = None) -> SendModel:
    return SendModel.from_directory(directory)


def load_batter_advance(path: str | Path | None = None) -> dict[tuple[str, int], dict[str, float]]:
    # The batter's advance by runner result and outs. Read from the research export until the table is promoted to models/send/.
    file = Path(path).resolve() if path is not None else DEFAULT_SEND_MODEL_DIR / BATTER_ADVANCE_FILE
    if not file.exists():
        raise SendModelError(f"batter advance table not found: {file}")
    df = pd.read_csv(file)
    columns = ["runner_result", "outs", *BATTER_ADVANCES]
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise SendModelError(f"{file.name} is missing columns {missing}")
    if df[columns].isna().any().any():
        raise SendModelError(f"{file.name} has missing values")
    shares: dict[tuple[str, int], dict[str, float]] = {}
    for row in df.itertuples():
        key = (str(row.runner_result), int(row.outs))
        if key[0] not in BATTER_ADVANCE_RESULTS or not 0 <= key[1] <= 2:
            raise SendModelError(f"{file.name}: unexpected row {key}")
        if key in shares:
            raise SendModelError(f"{file.name}: duplicate row {key}")
        values = {a: float(getattr(row, a)) for a in BATTER_ADVANCES}
        total = sum(values.values())
        if min(values.values()) < 0 or abs(total - 1) > SHARE_TOLERANCE:
            raise SendModelError(f"{file.name}: shares for {key} must be non-negative and sum to 1, got {total}")
        shares[key] = {a: v / total for a, v in values.items()}
    absent = [(r, o) for r in BATTER_ADVANCE_RESULTS for o in range(3) if (r, o) not in shares]
    if absent:
        raise SendModelError(f"{file.name} is missing rows {absent}")
    return shares
