import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from skipboard import send_model
from skipboard.send_model import (
    DEFAULT_SEND_MODEL_DIR,
    ENV_VAR,
    BattedBall,
    FielderProfile,
    PositioningStart,
    SendContext,
    SendModelError,
    load_batter_advance,
    load_send_model,
)
from skipboard.steal_model import RunnerProfile

FIXTURE = Path(__file__).parent / "fixtures" / "send_model"
REAL_FILES = [DEFAULT_SEND_MODEL_DIR / send_model.COEFFICIENTS_FILE, DEFAULT_SEND_MODEL_DIR / send_model.META_FILE,
              DEFAULT_SEND_MODEL_DIR / send_model.BOOTSTRAP_FILE, DEFAULT_SEND_MODEL_DIR / "send_fit_table_2023_2025.parquet"]


@pytest.fixture
def model():
    return load_send_model(FIXTURE)


def phi(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def geometry(x: float, y: float, pos: int, start_distance: float, start_angle: float) -> tuple[float, float, float]:
    spray = math.degrees(math.atan2(x, y))
    toward = {7: -spray, 8: abs(spray), 9: spray}[pos]
    sx, sy = start_distance * math.sin(math.radians(start_angle)), start_distance * math.cos(math.radians(start_angle))
    return math.hypot(x, y), toward, math.hypot(x - sx, y - sy)


def probs(model, ball, runner=None, fielder=None, start=None, context=None, season=2025):
    args = (ball, runner or RunnerProfile(sprint_speed=27.5), fielder or FielderProfile(arm_avg=88.0),
            start or PositioningStart(start_distance=290.0, start_angle=-28.0),
            context or SendContext(outs=1, score_diff=0, inning=5, bat_side="R"), season)
    return model.p_safe(*args), model.p_send(*args)


def test_first_case_matches_hand_computation(model):
    ball = BattedBall(x_feet=-150.0, y_feet=200.0, bb_type="line_drive", fielder_pos=7, launch_speed=95.0, launch_angle=12.0)
    start = PositioningStart(start_distance=290.0, start_angle=-30.0)
    context = SendContext(outs=1, score_diff=1, inning=8, bat_side="R", on_deck_quality=0.03)
    p_safe, p_send = probs(model, ball, RunnerProfile(sprint_speed=28.5), FielderProfile(arm_avg=90.0), start, context)
    dist, toward, fdist = geometry(-150.0, 200.0, 7, 290.0, -30.0)
    assert dist == pytest.approx(250.0)
    assert toward == pytest.approx(36.869897645844, abs=1e-9)
    derived = model.derived_covariates(ball, start, "R", 2025)
    assert derived == pytest.approx((dist, toward, fdist), abs=1e-12)
    eta_safe = (0.8 + 0.03 * (dist - 260) - 0.01 * (toward - 20) + 0.03 * (fdist - 65) - 0.02 * (95 - 92) - 0.03 * (12 - 11)
                - 0.03 * (90 - 88) + 0.2 * (28.5 - 27.5) - 0.1 + 0.3)
    eta_send = (0.2 + 0.02 * (dist - 260) + 0.01 * (toward - 20) + 0.03 * (fdist - 65) - 0.01 * (95 - 92) - 0.03 * (12 - 11)
                - 0.03 * (90 - 88) + 0.2 * (28.5 - 27.5) + 0.5 * (0.03 - 0.01) - 0.05 + 0.5 - 0.05 - 0.2 + 0.35)
    assert p_safe == pytest.approx(phi(eta_safe), abs=1e-12)
    assert p_send == pytest.approx(phi(eta_send), abs=1e-12)


def test_second_case_with_fills_matches_hand_computation(model):
    # A popup (a fly ball for the model) to right, every attribute missing, an unfitted season: the 2025 fills apply.
    # 2025 fills in the fixture: launch 73 mph and 33 degrees for fly balls, RF arm 90, sprint 27.5, RF start vs L 291 ft at 26 degrees.
    ball = BattedBall(x_feet=180.0, y_feet=240.0, bb_type="popup", fielder_pos=9)
    context = SendContext(outs=2, score_diff=-6, inning=10, bat_side="L")
    args = (ball, RunnerProfile(), FielderProfile(), PositioningStart(), context, 2031)
    dist, toward, fdist = geometry(180.0, 240.0, 9, 291.0, 26.0)
    assert dist == pytest.approx(300.0)
    eta_safe = (0.8 + 0.03 * (dist - 260) - 0.01 * (toward - 20) + 0.03 * (fdist - 65) - 0.02 * (73 - 92) - 0.03 * (33 - 11)
                - 0.03 * (90 - 88) + 0.2 * 0.0 + 0.3 + 0.1 - 0.05 + 1.0)
    eta_send = (0.2 + 0.02 * (dist - 260) + 0.01 * (toward - 20) + 0.03 * (fdist - 65) - 0.01 * (73 - 92) - 0.03 * (33 - 11)
                - 0.03 * (90 - 88) + 0.2 * 0.0 + 0.5 * (0.0 - 0.01) + 0.4 + 0.1 + 0.1 + 2.0 - 0.3 - 0.2)
    assert not context.late_close and context.score_level == -4 and context.inning_group == "9+"
    assert model.p_safe(*args) == pytest.approx(phi(eta_safe), abs=1e-12)
    assert model.p_send(*args) == pytest.approx(phi(eta_send), abs=1e-12)


@pytest.mark.parametrize("pos, expected_sign", [(7, 1), (8, 1), (9, -1)])
def test_spray_toward_line_convention(model, pos, expected_sign):
    # A ball toward the left-field line: spray angle about -53 degrees.
    ball = BattedBall(x_feet=-200.0, y_feet=150.0, bb_type="line_drive", fielder_pos=pos)
    toward = model.derived_covariates(ball, PositioningStart(start_distance=300.0, start_angle=0.0), "R", 2025).spray_toward_line
    spray = math.degrees(math.atan2(-200.0, 150.0))
    assert spray < 0
    assert toward == pytest.approx(expected_sign * abs(spray), abs=1e-12)


def test_late_close_and_context_checks():
    assert SendContext(outs=0, score_diff=-1, inning=7, bat_side="R").late_close
    assert not SendContext(outs=0, score_diff=2, inning=9, bat_side="R").late_close
    assert not SendContext(outs=0, score_diff=0, inning=6, bat_side="R").late_close
    with pytest.raises(ValueError):
        SendContext(outs=3, score_diff=0, inning=1, bat_side="R")
    with pytest.raises(ValueError):
        SendContext(outs=0, score_diff=0, inning=1, bat_side="B")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        BattedBall(x_feet=0.0, y_feet=250.0, bb_type="bunt", fielder_pos=8)  # type: ignore[arg-type]


BALL = BattedBall(x_feet=-150.0, y_feet=200.0, bb_type="ground_ball", fielder_pos=7, launch_speed=95.0, launch_angle=2.0)


def test_faster_runner_never_lowers_p_safe(model):
    values = [probs(model, BALL, runner=RunnerProfile(sprint_speed=v))[0] for v in np.linspace(24.0, 31.0, 15)]
    assert all(b >= a for a, b in zip(values, values[1:]))


def test_stronger_arm_never_raises_p_safe(model):
    values = [probs(model, BALL, fielder=FielderProfile(arm_avg=v))[0] for v in np.linspace(78.0, 98.0, 21)]
    assert all(b <= a for a, b in zip(values, values[1:]))


def test_deeper_fielding_point_never_lowers_p_safe(model):
    # The fielder starts 40 ft short of the ball on the same line, so only the fielding-point distance changes.
    angle = -20.0
    values, fielder_distances = [], []
    for d in np.linspace(200.0, 360.0, 17):
        ball = BattedBall(x_feet=d * math.sin(math.radians(angle)), y_feet=d * math.cos(math.radians(angle)), bb_type="line_drive",
                          fielder_pos=7, launch_speed=92.0, launch_angle=12.0)
        start = PositioningStart(start_distance=d - 40.0, start_angle=angle)
        fielder_distances.append(model.derived_covariates(ball, start, "R", 2025).fielder_distance)
        values.append(probs(model, ball, start=start)[0])
    assert fielder_distances == pytest.approx([40.0] * len(fielder_distances), abs=1e-9)
    assert all(b >= a for a, b in zip(values, values[1:]))


def test_greater_fielder_distance_never_lowers_p_safe(model):
    # The ball stays put; the fielder starts farther from it along the same line from home plate.
    angle = math.degrees(math.atan2(BALL.x_feet, BALL.y_feet))
    values = [probs(model, BALL, start=PositioningStart(start_distance=250.0 + k * 10.0, start_angle=angle))[0] for k in range(11)]
    assert all(b >= a for a, b in zip(values, values[1:]))


def test_missing_values_equal_the_explicit_fills(model):
    for season in (2024, 2025, 2031):
        for pos, side in ((7, "R"), (8, "L"), (9, "R")):
            ball = BattedBall(x_feet=40.0, y_feet=280.0, bb_type="fly_ball", fielder_pos=pos)
            context = SendContext(outs=0, score_diff=2, inning=3, bat_side=side)
            speed, angle = model.launch_fill(season, "fly_ball")
            dist, ang = model.start_fill(season, pos, side)
            explicit = (BattedBall(x_feet=40.0, y_feet=280.0, bb_type="fly_ball", fielder_pos=pos, launch_speed=speed, launch_angle=angle),
                        RunnerProfile(sprint_speed=model.sprint_fill(season)), FielderProfile(arm_avg=model.arm_fill(season, pos)),
                        PositioningStart(start_distance=dist, start_angle=ang), SendContext(outs=0, score_diff=2, inning=3, bat_side=side,
                                                                                           on_deck_quality=0.0), season)
            filled = (ball, RunnerProfile(), FielderProfile(), PositioningStart(), context, season)
            assert model.p_safe(*filled) == pytest.approx(model.p_safe(*explicit), abs=1e-15)
            assert model.p_send(*filled) == pytest.approx(model.p_send(*explicit), abs=1e-15)
    assert model.arm_fill(2031, 9) == model.arm_fill(2025, 9) == 90.0
    assert model.launch_fill(2025, "popup") == model.launch_fill(2025, "fly_ball")


def test_bootstrap_replicates(model):
    args = (BALL, RunnerProfile(sprint_speed=28.0), FielderProfile(arm_avg=88.0), PositioningStart(start_distance=290.0, start_angle=-28.0),
            SendContext(outs=1, score_diff=0, inning=5, bat_side="R"), 2025)
    reps = model.p_safe_bootstrap(*args)
    assert reps.shape == (5,)
    assert reps.min() <= reps.mean() <= reps.max()
    assert reps[2] == pytest.approx(model.p_safe(*args), abs=1e-12)  # the third fixture replicate equals the estimates
    assert model.rho == 0.8 and model.rho_bootstrap.tolist() == pytest.approx([0.7, 0.75, 0.8, 0.85, 0.9])


def copy_fixture(tmp_path: Path) -> Path:
    target = tmp_path / "model"
    shutil.copytree(FIXTURE, target)
    return target


def test_bootstrap_header_missing_a_term_raises(tmp_path):
    target = copy_fixture(tmp_path)
    path = target / send_model.BOOTSTRAP_FILE
    pd.read_csv(path).drop(columns="safe:arm_avg").to_csv(path, index=False)
    with pytest.raises(SendModelError, match="header"):
        load_send_model(target)


def test_missing_bootstrap_file_raises(tmp_path):
    target = copy_fixture(tmp_path)
    (target / send_model.BOOTSTRAP_FILE).unlink()
    with pytest.raises(SendModelError, match="not found"):
        load_send_model(target)


def test_unknown_or_missing_terms_raise(tmp_path):
    target = copy_fixture(tmp_path)
    path = target / send_model.COEFFICIENTS_FILE
    path.write_text(path.read_text().replace("safe,spray_toward_line,", "safe,spray_mystery,"))
    with pytest.raises(SendModelError, match="unknown term"):
        load_send_model(target)
    target2 = tmp_path / "model2"
    shutil.copytree(FIXTURE, target2)
    coef = pd.read_csv(target2 / send_model.COEFFICIENTS_FILE)
    coef[coef["term"] != "late_close"].to_csv(target2 / send_model.COEFFICIENTS_FILE, index=False)
    with pytest.raises(SendModelError, match="missing send terms"):
        load_send_model(target2)


def test_factor_levels_that_differ_raise(tmp_path):
    target = copy_fixture(tmp_path)
    path = target / send_model.META_FILE
    path.write_text(path.read_text().replace('"fly_ball (includes popup)"', '"popup"'))
    with pytest.raises(SendModelError, match="factor levels"):
        load_send_model(target)


def test_constraint_violation_raises(tmp_path):
    target = copy_fixture(tmp_path)
    path = target / send_model.COEFFICIENTS_FILE
    path.write_text(path.read_text().replace("safe,arm_avg,-0.03,", "safe,arm_avg,0.03,"))
    with pytest.raises(SendModelError, match="constraint"):
        load_send_model(target)


def test_directory_from_argument_environment_or_default(monkeypatch, tmp_path):
    monkeypatch.delenv(ENV_VAR, raising=False)
    assert send_model._resolve(None) == DEFAULT_SEND_MODEL_DIR
    monkeypatch.setenv(ENV_VAR, str(FIXTURE))
    assert send_model._resolve(None) == FIXTURE.resolve()
    assert load_send_model().safe == load_send_model(FIXTURE).safe
    assert send_model._resolve(tmp_path) == tmp_path.resolve()


def test_batter_advance_loader(tmp_path):
    shares = load_batter_advance(FIXTURE / send_model.BATTER_ADVANCE_FILE)
    assert len(shares) == 9
    assert all(sum(v.values()) == pytest.approx(1.0, abs=1e-12) for v in shares.values())
    assert shares[("out at home", 1)] == pytest.approx({"first": 0.5, "second": 0.45, "out": 0.05})
    bad = pd.read_csv(FIXTURE / send_model.BATTER_ADVANCE_FILE)
    bad.loc[0, "first"] = 0.8  # the row now sums to 0.9
    bad.to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(SendModelError, match="sum to 1"):
        load_batter_advance(tmp_path / "bad.csv")
    short = pd.read_csv(FIXTURE / send_model.BATTER_ADVANCE_FILE).iloc[:-1]
    short.to_csv(tmp_path / "short.csv", index=False)
    with pytest.raises(SendModelError, match="missing rows"):
        load_batter_advance(tmp_path / "short.csv")
    with pytest.raises(SendModelError, match="not found"):
        load_batter_advance(tmp_path / "absent.csv")


@pytest.mark.skipif(not all(p.exists() for p in REAL_FILES),
                    reason="research exports are not in data/derived/send (they are built locally from data that is not tracked)")
def test_parity_with_research_exports():
    import json

    model = load_send_model(DEFAULT_SEND_MODEL_DIR)
    coef = pd.read_csv(REAL_FILES[0])
    meta = json.loads(REAL_FILES[1].read_text())
    fit = pd.read_parquet(REAL_FILES[3])
    bb = fit["bb_type"].replace({"popup": "fly_ball"})
    launch = meta["launch_fill_values"]
    speed = fit["launch_speed"].fillna(pd.Series([launch[f"{s} {b}"]["launch_speed"] for s, b in zip(fit["season"], bb)], index=fit.index))
    angle = fit["launch_angle"].fillna(pd.Series([launch[f"{s} {b}"]["launch_angle"] for s, b in zip(fit["season"], bb)], index=fit.index))
    toward = np.select([fit["fielder_pos"] == 7, fit["fielder_pos"] == 9], [-fit["spray_angle"], fit["spray_angle"]], fit["spray_angle"].abs())
    raw = {"fielding_distance": fit["landing_distance"], "spray_toward_line": toward, "fielder_distance": fit["fielder_distance"],
           "launch_speed": speed, "launch_angle": angle, "arm_avg": fit["arm_avg"], "runner_sprint_speed": fit["runner_sprint_speed"],
           "on_deck_quality": fit["on_deck_quality"]}
    score = fit["score_diff"].clip(-4, 4)
    group = pd.cut(fit["inning"], [0, 3, 6, 8, 99], labels=["1-3", "4-6", "7-8", "9+"]).astype(str)
    factors = {"pos=CF": fit["fielder_pos"] == 8, "pos=RF": fit["fielder_pos"] == 9, "bb=line_drive": bb == "line_drive", "bb=fly_ball": bb == "fly_ball",
               "bat_side=L": fit["stand"] == "L", "outs=1": fit["outs"] == 1, "outs=2": fit["outs"] == 2,
               "late_close": (fit["inning"] >= 7) & (fit["score_diff"].abs() <= 1)}
    factors.update({f"score={s:+d}": score == s for s in (-4, -3, -2, -1, 1, 2, 3, 4)})
    factors.update({f"inning={g}": group == g for g in ("4-6", "7-8", "9+")})
    eta = {"safe": np.zeros(len(fit)), "send": np.zeros(len(fit))}
    for r in coef[coef["equation"] != "rho"].itertuples():
        if r.term == "intercept":
            x = np.ones(len(fit))
        elif r.term in raw:
            x = np.asarray(raw[r.term], float) - r.center
        else:
            x = np.asarray(factors[r.term], float)
        eta[r.equation] += r.estimate * x
    expected_safe = np.array([phi(e) for e in eta["safe"]])
    expected_send = np.array([phi(e) for e in eta["send"]])
    engine_safe, engine_send = [], []
    for r in fit.itertuples():
        args = (BattedBall(x_feet=r.x_feet, y_feet=r.y_feet, bb_type=r.bb_type, fielder_pos=int(r.fielder_pos),
                           launch_speed=None if pd.isna(r.launch_speed) else r.launch_speed,
                           launch_angle=None if pd.isna(r.launch_angle) else r.launch_angle),
                RunnerProfile(sprint_speed=r.runner_sprint_speed), FielderProfile(arm_avg=r.arm_avg),
                PositioningStart(start_distance=r.start_distance, start_angle=r.start_angle),
                SendContext(outs=int(r.outs), score_diff=int(r.score_diff), inning=int(r.inning), bat_side=r.stand, on_deck_quality=r.on_deck_quality),
                int(r.season))
        engine_safe.append(model.p_safe(*args))
        engine_send.append(model.p_send(*args))
    assert len(fit) > 4000
    assert np.max(np.abs(np.array(engine_safe) - expected_safe)) < 1e-10
    assert np.max(np.abs(np.array(engine_send) - expected_send)) < 1e-10
