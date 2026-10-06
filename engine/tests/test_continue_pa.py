from skipboard.continue_pa import continue_pa_runs, continue_pa_wp
from skipboard.value import run_expectancy, wp, wp_after

BASE_LABEL = {0: "___", 1: "1__", 2: "_2_", 4: "__3", 3: "12_", 5: "1_3", 6: "_23", 7: "123"}

# Tolerances = largest observed gap (state ratio v1, shrunk) plus a margin: 0.0509 + 0.005 runs,
# 0.0122 + 0.002 win probability, 3.03 + 0.5 percentage points for break-even rates.
RUNS_TOLERANCE = 0.056
WP_TOLERANCE = 0.0143
BREAK_EVEN_TOLERANCE_PP = 3.54

# Diagnostics of the raw plate-appearance model (continue_pa), not of decision values: decisions use
# the anchored value_wp and value_runs (test_valuation.py), which match the tables exactly.
# League-average players at 0-0: no profiles and no platoon adjustment (base-out state ratio on).


def test_raw_continue_pa_runs_vs_run_expectancy_diagnostic(state_factory):
    # Sources of the gap: docs/models.md, "Plate appearance (pa v1)" > "Consistency checks and tolerances".
    print("\nraw continue_pa_runs vs run expectancy, diagnostic (league-average players, 0-0)")
    print(f"{'bases':<6}{'outs':>5}{'re24':>9}{'engine':>9}{'diff':>9}")
    worst = 0.0
    for base in BASE_LABEL:
        for outs in range(3):
            s = state_factory(base_code=base, outs=outs)
            table, engine = run_expectancy(s), continue_pa_runs(s, apply_platoon=False)
            worst = max(worst, abs(engine - table))
            print(f"{BASE_LABEL[base]:<6}{outs:>5}{table:>9.4f}{engine:>9.4f}{engine - table:>+9.4f}")
    print(f"largest absolute difference: {worst:.4f} (tolerance {RUNS_TOLERANCE})")
    assert worst <= RUNS_TOLERANCE


def test_raw_continue_pa_wp_vs_wp_diagnostic(state_factory):
    # Sources of the gap: docs/models.md, "Plate appearance (pa v1)" > "Consistency checks and tolerances".
    print("\nraw continue_pa_wp vs wp, diagnostic (league-average players, 0-0, tied)")
    worst = 0.0
    for inning in (1, 5, 9):
        for half in (0, 1):
            diffs = []
            for base in BASE_LABEL:
                for outs in range(3):
                    s = state_factory(inning=inning, half=half, base_code=base, outs=outs, bat_score=2, fld_score=2)
                    diffs.append(continue_pa_wp(s, apply_platoon=False) - wp(s))
            row_worst = max(abs(d) for d in diffs)
            worst = max(worst, row_worst)
            print(f"{'top' if half == 0 else 'bot'} {inning}: " + " ".join(f"{d:+.4f}" for d in diffs) + f"  | max {row_worst:.4f}")
    print("(each row: bases ___ 1__ _2_ __3 12_ 1_3 _23 123, outs 0 1 2 within each)")
    print(f"largest absolute difference: {worst:.4f} (tolerance {WP_TOLERANCE})")
    assert worst <= WP_TOLERANCE


def _break_even(v_runner_first: float, v_runner_second: float, v_fail: float) -> float:
    return (v_runner_first - v_fail) / (v_runner_second - v_fail)


def test_raw_continue_pa_steal_break_even_vs_tables_diagnostic(state_factory):
    # Sources of the gap: docs/models.md, "Plate appearance (pa v1)" > "Consistency checks and tolerances".
    print("\nraw continue_pa steal break-even vs tables, diagnostic (league-average players, 0-0)")
    print(f"{'context':<18}{'outs':>5}{'tables':>9}{'engine':>9}{'diff (pp)':>11}")
    worst = 0.0

    def report(label: str, outs: int, table: float, engine: float) -> None:
        nonlocal worst
        worst = max(worst, abs(engine - table) * 100)
        print(f"{label:<18}{outs:>5}{table:>9.3f}{engine:>9.3f}{(engine - table) * 100:>+11.2f}")

    for outs in range(3):
        first, second = state_factory(base_code=1, outs=outs), state_factory(base_code=2, outs=outs)
        if outs < 2:
            fail = state_factory(outs=outs + 1)
            fail_table, fail_engine = run_expectancy(fail), continue_pa_runs(fail, apply_platoon=False)
        else:
            fail_table = fail_engine = 0.0  # the half-inning ends
        table = _break_even(run_expectancy(first), run_expectancy(second), fail_table)
        engine = _break_even(continue_pa_runs(first, apply_platoon=False), continue_pa_runs(second, apply_platoon=False), fail_engine)
        report("runs", outs, table, engine)

    for inning, half, label in [(9, 0, "top 9th, tied"), (9, 1, "bottom 9th, tied"), (7, 1, "bottom 7th, tied")]:
        game = dict(inning=inning, half=half, bat_score=2, fld_score=2)
        for outs in range(3):
            first, second = state_factory(base_code=1, outs=outs, **game), state_factory(base_code=2, outs=outs, **game)
            if outs < 2:
                fail = state_factory(outs=outs + 1, **game)
                fail_table, fail_engine = wp(fail), continue_pa_wp(fail, apply_platoon=False)
            else:
                fail_table = fail_engine = wp_after(first, base_post=0, outs_post=3, runs=0, new_pa=False)
            table = _break_even(wp(first), wp(second), fail_table)
            engine = _break_even(continue_pa_wp(first, apply_platoon=False), continue_pa_wp(second, apply_platoon=False), fail_engine)
            report(label, outs, table, engine)

    print(f"largest absolute difference: {worst:.2f} pp (tolerance {BREAK_EVEN_TOLERANCE_PP} pp)")
    assert worst <= BREAK_EVEN_TOLERANCE_PP


def test_continue_pa_wp_in_range(state_factory):
    for half in (0, 1):
        s = state_factory(inning=9, half=half, base_code=7, outs=2, bat_score=3, fld_score=4, balls=3, strikes=2)
        assert 0 <= continue_pa_wp(s) <= 1
