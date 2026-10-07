import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from skipboard import steal_model
from skipboard.steal_model import (
    DEFAULT_STEAL_MODEL_DIR,
    ENV_VAR,
    CatcherProfile,
    PitcherProfile,
    RunnerProfile,
    StealModelError,
    load_steal_model,
)

FIXTURE = Path(__file__).parent / "fixtures" / "steal_model"
REAL_FILES = [DEFAULT_STEAL_MODEL_DIR / steal_model.COEFFICIENTS_FILE, DEFAULT_STEAL_MODEL_DIR / steal_model.META_FILE,
              DEFAULT_STEAL_MODEL_DIR / "steal_fit_table_2023_2025.parquet"]


@pytest.fixture
def model():
    return load_steal_model(FIXTURE)


def expit(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def p_with(model, pitch_result="ball", balls=0, strikes=0, pickoff_throws=0, bat_side="R", season=2025,
           runner=None, catcher=None, pitcher=None):
    return model.p_safe(pitch_result, balls, strikes, pickoff_throws, bat_side, season,
                        runner or RunnerProfile(), catcher or CatcherProfile(), pitcher or PitcherProfile(hand="R"))


def test_first_case_matches_hand_computation(model):
    p = model.p_safe("ball", 1, 0, 0, "R", 2024, RunnerProfile(sprint_speed=29.0, aggressiveness=0.10),
                     CatcherProfile(pop_time=2.05), PitcherProfile(hand="R", primary_lead=11.0, lead_gained=5.0))
    eta = 1.5 + 4.0 * (0.10 - 0.08) + 0.05 * (29.0 - 28.0) + 6.0 * (2.05 - 2.0) + 0.2 * (11.0 - 10.0) + 0.3 * (5.0 - 4.0) - 0.1
    assert eta == pytest.approx(2.33)
    assert p == pytest.approx(expit(eta), abs=1e-12)


def test_second_case_matches_hand_computation(model):
    # 3-2 strike three, three pickoff throws (2+ level), switch hitter against a left-hander (bats right),
    # 2025, sprint speed and pop time missing (2025 fill values 27.2 and 1.97).
    p = model.p_safe("strike_three", 3, 2, 3, "B", 2025, RunnerProfile(aggressiveness=0.06), CatcherProfile(),
                     PitcherProfile(hand="L", primary_lead=9.5, lead_gained=2.0))
    eta = (1.5 + 0.1 - 1.0 - 0.2 + 0.9 - 0.15 + 4.0 * (0.06 - 0.08) + 0.05 * (27.2 - 28.0) + 6.0 * (1.97 - 2.0)
           + 0.2 * (9.5 - 10.0) + 0.3 * (2.0 - 4.0))
    assert eta == pytest.approx(0.15)
    assert p == pytest.approx(expit(eta), abs=1e-12)


CONTEXTS = [
    dict(pitch_result="ball", balls=0, strikes=0, pickoff_throws=0, bat_side="R", season=2023, hand="R"),
    dict(pitch_result="strike", balls=1, strikes=1, pickoff_throws=1, bat_side="L", season=2024, hand="L"),
    dict(pitch_result="strike_three", balls=3, strikes=2, pickoff_throws=2, bat_side="B", season=2025, hand="R"),
]
SWEEPS = {
    "sprint_speed": np.linspace(23.0, 31.0, 9),
    "pop_time": np.linspace(1.75, 2.20, 10),
    "primary_lead": np.linspace(7.5, 14.0, 8),
    "lead_gained": np.linspace(-0.5, 12.0, 11),
}


@pytest.mark.parametrize("attribute", list(SWEEPS))
def test_monotone_directions(model, attribute):
    for ctx in CONTEXTS:
        ctx = dict(ctx)
        hand = ctx.pop("hand")
        probs = []
        for value in SWEEPS[attribute]:
            runner = RunnerProfile(sprint_speed=value) if attribute == "sprint_speed" else RunnerProfile()
            catcher = CatcherProfile(pop_time=value) if attribute == "pop_time" else CatcherProfile()
            pitcher = PitcherProfile(hand=hand, **({attribute: value} if attribute in ("primary_lead", "lead_gained") else {}))
            probs.append(p_with(model, runner=runner, catcher=catcher, pitcher=pitcher, **ctx))
        assert all(b >= a for a, b in zip(probs, probs[1:])), (attribute, ctx)


def test_count_3_2_term_only_at_full_count_strike_three(model):
    full = p_with(model, "strike_three", 3, 2)
    two_two = p_with(model, "strike_three", 2, 2)
    assert logit(full) - logit(two_two) == pytest.approx(-1.0)
    assert p_with(model, "strike_three", 1, 2) == pytest.approx(two_two)
    assert p_with(model, "strike", 2, 1) == pytest.approx(p_with(model, "strike", 0, 1))


@pytest.mark.parametrize("pitch_result", ["ball", "strike"])
def test_full_count_with_ball_or_strike_raises(model, pitch_result):
    with pytest.raises(ValueError):
        p_with(model, pitch_result, 3, 2)


@pytest.mark.parametrize("pitch_result, balls, strikes", [("ball", 3, 1), ("strike", 0, 2), ("strike_three", 1, 1), ("foul", 0, 0), ("ball", 4, 0)])
def test_impossible_pitches_raise(model, pitch_result, balls, strikes):
    with pytest.raises(ValueError):
        p_with(model, pitch_result, balls, strikes)


def test_two_or_more_pickoff_throws_share_one_level(model):
    two, three, seven = (p_with(model, pickoff_throws=n) for n in (2, 3, 7))
    assert two == three == seven
    assert p_with(model, pickoff_throws=1) != two
    with pytest.raises(ValueError):
        p_with(model, pickoff_throws=-1)


@pytest.mark.parametrize("season", [2024, 2025, 2031])
def test_missing_attribute_uses_season_fill(model, season):
    fills = model.fill_values(season)
    explicit = p_with(model, season=season,
                      runner=RunnerProfile(sprint_speed=fills["sprint_speed"], aggressiveness=fills["runner_aggressiveness"]),
                      catcher=CatcherProfile(pop_time=fills["pop_time"]),
                      pitcher=PitcherProfile(hand="R", primary_lead=fills["pitcher_primary_lead"], lead_gained=fills["pitcher_lead_gained"]))
    assert p_with(model, season=season) == pytest.approx(explicit, abs=1e-15)


def test_switch_hitter_bats_opposite_the_pitcher(model):
    lefty, righty = PitcherProfile(hand="L"), PitcherProfile(hand="R")
    assert p_with(model, bat_side="B", pitcher=lefty) == p_with(model, bat_side="R", pitcher=lefty)
    assert p_with(model, bat_side="B", pitcher=righty) == p_with(model, bat_side="L", pitcher=righty)
    assert p_with(model, bat_side="L", pitcher=righty) != p_with(model, bat_side="R", pitcher=righty)


@pytest.mark.parametrize("season", [2019, 2026, 2031])
def test_unfitted_season_uses_latest_fitted_season(model, season):
    assert model.fitted_seasons == (2023, 2024, 2025)
    assert model.season_used(season) == 2025
    assert model.fill_values(season) == model.fill_values(2025)
    for pitch_result, balls, strikes in [("ball", 0, 0), ("strike_three", 3, 2)]:
        assert p_with(model, pitch_result, balls, strikes, season=season) == p_with(model, pitch_result, balls, strikes, season=2025)


def test_pitcher_hand_is_required_and_checked():
    with pytest.raises(TypeError):
        PitcherProfile()  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        PitcherProfile(hand="S")  # type: ignore[arg-type]


def test_directory_from_argument_environment_or_default(monkeypatch, tmp_path):
    monkeypatch.delenv(ENV_VAR, raising=False)
    assert steal_model._resolve(None) == DEFAULT_STEAL_MODEL_DIR
    monkeypatch.setenv(ENV_VAR, str(FIXTURE))
    assert steal_model._resolve(None) == FIXTURE.resolve()
    assert load_steal_model().estimates == load_steal_model(FIXTURE).estimates
    assert steal_model._resolve(tmp_path) == tmp_path.resolve()


def copy_fixture(tmp_path: Path) -> Path:
    target = tmp_path / "model"
    shutil.copytree(FIXTURE, target)
    return target


def test_unknown_term_raises(tmp_path):
    target = copy_fixture(tmp_path)
    with open(target / steal_model.COEFFICIENTS_FILE, "a") as f:
        f.write("mystery_term,1.0,0.1,none,\n")
    with pytest.raises(StealModelError, match="unknown term"):
        load_steal_model(target)


def test_constraint_violation_raises(tmp_path):
    target = copy_fixture(tmp_path)
    path = target / steal_model.COEFFICIENTS_FILE
    path.write_text(path.read_text().replace("sprint_speed,0.05,", "sprint_speed,-0.05,"))
    with pytest.raises(StealModelError, match="constraint"):
        load_steal_model(target)


def test_missing_files_raise(tmp_path):
    with pytest.raises(StealModelError, match="not found"):
        load_steal_model(tmp_path)


@pytest.mark.skipif(not all(p.exists() for p in REAL_FILES),
                    reason="research exports are not in data/derived/steal (they are built locally from data that is not tracked)")
def test_parity_with_research_exports():
    model = load_steal_model(DEFAULT_STEAL_MODEL_DIR)
    coef = pd.read_csv(REAL_FILES[0]).set_index("term")
    est, center = coef["estimate"], coef["centering_mean"]
    fit = pd.read_parquet(REAL_FILES[2])
    fit = fit[fit["pitchout"].astype(int) == 0].reset_index(drop=True)
    eta = np.full(len(fit), est["intercept"])
    for term in [t for t in steal_model.CONTINUOUS_TERMS if t in est.index]:
        eta += est[term] * (fit[term].to_numpy(float) - center[term])
    eta += est["pitch_result=strike"] * (fit["pitch_result"] == "strike") + est["pitch_result=strike_three"] * (fit["pitch_result"] == "strike_three")
    eta += est["count=3-2"] * ((fit["balls"] == 3) & (fit["strikes"] == 2))
    eta += est["pickoff_throws=1"] * (fit["pickoff_throws"] == "1") + est["pickoff_throws=2+"] * (fit["pickoff_throws"] == "2+")
    eta += est["pitcher_hand=L"] * (fit["pitcher_hand"] == "L") + est["bat_side=L"] * (fit["bat_side"] == "L")
    for term in [t for t in est.index if t.startswith("season=")]:
        eta += est[term] * (fit["season"] == int(term.split("=")[1]))
    expected = 1.0 / (1.0 + np.exp(-np.asarray(eta, float)))
    engine = np.array([
        model.p_safe(r.pitch_result, int(r.balls), int(r.strikes), {"0": 0, "1": 1, "2+": 2}[r.pickoff_throws], r.bat_side, int(r.season),
                     RunnerProfile(sprint_speed=float(r.sprint_speed), aggressiveness=float(r.runner_aggressiveness)),
                     CatcherProfile(pop_time=float(r.pop_time)),
                     PitcherProfile(hand=r.pitcher_hand, primary_lead=float(r.pitcher_primary_lead), lead_gained=float(r.pitcher_lead_gained)))
        for r in fit.itertuples()
    ])
    assert len(engine) > 8000
    assert np.max(np.abs(engine - expected)) < 1e-10
