from dataclasses import dataclass
from typing import Literal, NamedTuple

GameType = Literal["regular", "postseason"]
BatterHand = Literal["L", "R", "B"]
PitcherHand = Literal["L", "R"]


class InvalidStateError(ValueError):
    pass


class Outfielders(NamedTuple):
    left: str | None
    center: str | None
    right: str | None


@dataclass(frozen=True, kw_only=True)
class GameState:
    season: int
    game_type: GameType
    inning: int
    half: int  # 0 top, 1 bottom
    bat_score: int
    fld_score: int
    outs: int
    balls: int
    strikes: int
    runner1: str | None = None
    runner2: str | None = None
    runner3: str | None = None
    batter: str
    batter_hand: BatterHand
    pitcher: str
    pitcher_hand: PitcherHand
    catcher: str
    outfielders: Outfielders | None = None
    disengagements_used: int = 0
    disengagement_limit: int | None = None

    def __post_init__(self) -> None:
        for problem in _problems(self):
            raise InvalidStateError(problem)

    @property
    def base_code(self) -> int:
        return (self.runner1 is not None) * 1 + (self.runner2 is not None) * 2 + (self.runner3 is not None) * 4

    @property
    def diff(self) -> int:
        return self.bat_score - self.fld_score


def _problems(s: GameState) -> list[str]:
    problems = []
    if s.game_type not in ("regular", "postseason"):
        problems.append(f"game_type must be 'regular' or 'postseason', got {s.game_type!r}")
    if s.inning < 1:
        problems.append(f"inning must be at least 1, got {s.inning}")
    if s.half not in (0, 1):
        problems.append(f"half must be 0 (top) or 1 (bottom), got {s.half}")
    if not 0 <= s.outs <= 2:
        problems.append(f"outs must be 0 to 2, got {s.outs}")
    if not 0 <= s.balls <= 3:
        problems.append(f"balls must be 0 to 3, got {s.balls}")
    if not 0 <= s.strikes <= 2:
        problems.append(f"strikes must be 0 to 2, got {s.strikes}")
    if s.bat_score < 0 or s.fld_score < 0:
        problems.append(f"scores cannot be negative, got {s.bat_score}-{s.fld_score}")
    if s.half == 1 and s.inning >= 9 and s.bat_score > s.fld_score:
        problems.append("batting team leads in the bottom of the 9th or later: the game is over")
    if s.batter_hand not in ("L", "R", "B"):
        problems.append(f"batter_hand must be L, R or B, got {s.batter_hand!r}")
    if s.pitcher_hand not in ("L", "R"):
        problems.append(f"pitcher_hand must be L or R, got {s.pitcher_hand!r}")
    runners = [r for r in (s.runner1, s.runner2, s.runner3) if r is not None]
    if len(set(runners)) < len(runners) or s.batter in runners:
        problems.append("the same player appears on two bases or as both batter and runner")
    if s.disengagements_used < 0:
        problems.append(f"disengagements_used cannot be negative, got {s.disengagements_used}")
    if s.disengagement_limit is not None and s.disengagements_used > s.disengagement_limit:
        problems.append(f"disengagements_used ({s.disengagements_used}) exceeds the limit ({s.disengagement_limit})")
    return problems
