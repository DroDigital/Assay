import random

import pytest

from assay.stats import Verdict, decide, samples_to_decide, wilson_interval


def test_wilson_matches_published_values():
    low, high = wilson_interval(5, 10)
    assert low == pytest.approx(0.2366, abs=1e-4)
    assert high == pytest.approx(0.7634, abs=1e-4)
    assert wilson_interval(0, 100)[1] == pytest.approx(0.0370, abs=1e-4)


def test_wilson_without_trials_is_uninformative():
    assert wilson_interval(0, 0) == (0.0, 1.0)


@pytest.mark.parametrize("n", [1, 2, 7, 50, 1000, 10**6])
def test_boundaries_are_exact(n):
    # A 1e-17 residue at k=0 would turn "no violations" into a failure at zero tolerance.
    assert wilson_interval(0, n)[0] == 0.0
    assert wilson_interval(n, n)[1] == 1.0


@pytest.mark.parametrize(
    ("k", "n", "confidence"), [(-1, 5, 0.95), (6, 5, 0.95), (1, 5, 1.0), (1, 5, 0.0)]
)
def test_invalid_arguments_are_rejected(k, n, confidence):
    with pytest.raises(ValueError, match="need 0 <= k <= n|confidence must be"):
        wilson_interval(k, n, confidence)


def test_interval_properties_hold_for_random_inputs():
    rng = random.Random(1234)
    for _ in range(500):
        n = rng.randint(1, 500)
        k = rng.randint(0, n)
        confidence = rng.choice([0.8, 0.9, 0.95, 0.99])
        low, high = wilson_interval(k, n, confidence)
        assert 0.0 <= low <= k / n <= high <= 1.0
        wider = wilson_interval(k, n, min(0.999, confidence + 0.005))
        assert wider[0] <= low + 1e-12 and wider[1] >= high - 1e-12


def test_more_data_narrows_the_interval():
    widths = [h - lo for lo, h in (wilson_interval(n // 10, n) for n in (10, 100, 1000, 10000))]
    assert widths == sorted(widths, reverse=True)


def test_empirical_coverage_is_close_to_nominal():
    """Over many simulated experiments the 95% interval should contain the truth ~95% of the time."""
    rng = random.Random(99)
    truth, n, runs = 0.05, 200, 3000
    covered = 0
    for _ in range(runs):
        k = sum(rng.random() < truth for _ in range(n))
        low, high = wilson_interval(k, n)
        covered += low <= truth <= high
    assert 0.93 <= covered / runs <= 0.97


@pytest.mark.parametrize(
    ("k", "n", "tolerance", "expected"),
    [
        (0, 100, 0.0, Verdict.PASS),
        (1, 100, 0.0, Verdict.FAIL),
        (0, 1, 0.0, Verdict.PASS),
        (0, 100, 0.05, Verdict.PASS),
        (0, 50, 0.05, Verdict.INCONCLUSIVE),
        (30, 100, 0.05, Verdict.FAIL),
        (2, 20, 0.05, Verdict.INCONCLUSIVE),
        (3, 20, 0.05, Verdict.FAIL),
        (0, 0, 0.1, Verdict.INCONCLUSIVE),
        (100, 100, 1.0, Verdict.PASS),
    ],
)
def test_verdict_table(k, n, tolerance, expected):
    assert decide(k, n, tolerance) is expected


def test_min_trials_forces_inconclusive():
    assert decide(0, 10, 0.0, min_trials=30) is Verdict.INCONCLUSIVE
    assert decide(0, 30, 0.0, min_trials=30) is Verdict.PASS


def test_tolerance_must_be_a_probability():
    with pytest.raises(ValueError, match="tolerance must be in"):
        decide(0, 10, 1.5)


def test_verdict_severity_orders_pass_below_fail():
    assert Verdict.PASS.severity < Verdict.INCONCLUSIVE.severity < Verdict.FAIL.severity


@pytest.mark.parametrize(("tolerance", "expected"), [(0.05, 73), (0.01, 381), (0.001, 3838)])
def test_trials_needed_to_certify_zero_violations(tolerance, expected):
    needed = samples_to_decide(0, 1, tolerance)
    assert needed == expected
    assert decide(0, needed, tolerance) is Verdict.PASS
    assert decide(0, needed - 1, tolerance) is Verdict.INCONCLUSIVE


def test_samples_to_decide_edge_cases():
    assert samples_to_decide(0, 200, 0.05) == 200  # already decisive
    assert samples_to_decide(5, 100, 0.05) is None  # observed rate sits on the tolerance
    assert samples_to_decide(0, 0, 0.05) is None
    assert samples_to_decide(1, 10, 0.0001) == 10  # a clear FAIL needs no more data
    assert (
        samples_to_decide(1, 19, 0.05, cap=1000) is None
    )  # rate barely above tolerance: needs > cap
    needed = samples_to_decide(3, 60, 0.1)
    assert needed is not None and needed > 60
    assert decide(round(3 / 60 * needed), needed, 0.1) is not Verdict.INCONCLUSIVE
