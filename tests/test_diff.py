import json

import pytest

from assay import Property, Suite
from assay.diff import (
    ADDED,
    IMPROVED,
    REGRESSED,
    REMOVED,
    UNCHANGED,
    diff_results,
    load_result,
    render_diff,
)
from assay.errors import SpecError


def entry(name, verdict, violations, trials, low, high):
    return {"name": name, "verdict": verdict, "violations": violations, "trials": trials,
            "rate": violations / trials if trials else None, "ci_low": low, "ci_high": high}  # fmt: skip


def doc(*contracts):
    return {"schema": "assay.result/v1", "contracts": list(contracts)}


def status_of(old, new):
    return diff_results(doc(old), doc(new))[0].status


def test_verdict_changes_are_regressions_or_improvements():
    ok = entry("c", "pass", 0, 100, 0, 0.03)
    inconclusive = entry("c", "inconclusive", 3, 100, 0.01, 0.08)
    bad = entry("c", "fail", 20, 100, 0.13, 0.29)
    assert status_of(ok, bad) == REGRESSED and status_of(ok, inconclusive) == REGRESSED
    assert status_of(bad, ok) == IMPROVED and status_of(bad, inconclusive) == IMPROVED


def test_same_verdict_is_a_regression_only_when_the_rate_moves_beyond_chance():
    base = entry("c", "fail", 20, 100, 0.13, 0.29)
    worse = entry("c", "fail", 60, 100, 0.50, 0.69)
    similar = entry("c", "fail", 25, 100, 0.18, 0.34)
    better = entry("c", "fail", 2, 100, 0.005, 0.07)
    assert status_of(base, worse) == REGRESSED
    assert status_of(base, similar) == UNCHANGED
    assert status_of(base, better) == IMPROVED


def test_added_and_removed_contracts():
    diffs = {d.name: d for d in diff_results(
        doc(entry("old", "pass", 0, 10, 0, 0.3)),
        doc(entry("new-bad", "fail", 9, 10, 0.6, 0.97)),
    )}  # fmt: skip
    assert diffs["old"].status == REMOVED and not diffs["old"].is_regression
    assert diffs["new-bad"].status == ADDED and diffs["new-bad"].is_regression
    added_ok = diff_results(doc(), doc(entry("n", "pass", 0, 10, 0, 0.3)))[0]
    assert added_ok.status == ADDED and not added_ok.is_regression


def test_contracts_without_trials_are_not_compared_by_rate():
    empty = entry("c", "inconclusive", 0, 0, 0.0, 1.0)
    assert status_of(empty, empty) == UNCHANGED


def test_render_orders_regressions_first_and_counts_them():
    diffs = diff_results(
        doc(
            entry("a", "pass", 0, 100, 0, 0.03),
            entry("b", "fail", 50, 100, 0.4, 0.6),
            entry("c", "pass", 0, 100, 0, 0.03),
        ),
        doc(
            entry("a", "fail", 50, 100, 0.4, 0.6),
            entry("b", "pass", 0, 100, 0, 0.03),
            entry("c", "pass", 0, 100, 0, 0.03),
        ),
    )
    text = render_diff(diffs)
    lines = text.splitlines()
    assert lines[0].strip().startswith("REGRESSED") and "a" in lines[0]
    assert lines[1].strip().startswith("IMPROVED")
    assert "1 regression(s) · 1 improvement(s) · 3 contract(s) compared" in text
    assert "\x1b[" not in text and "\x1b[31m" in render_diff(diffs, color=True)


def test_load_result_validates_the_file(tmp_path):
    with pytest.raises(SpecError, match="cannot read"):
        load_result(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text("[]")
    with pytest.raises(SpecError, match="not an assay result"):
        load_result(bad)
    bad.write_text("{nope")
    with pytest.raises(SpecError, match="cannot read"):
        load_result(bad)
    good = tmp_path / "good.json"
    result = Suite("t", lambda x: x, [1]).add(Property("p", check=lambda y: True)).run()
    good.write_text(json.dumps(result.to_dict()))
    assert load_result(good)["suite"]["name"] == "t"


def test_end_to_end_with_real_results():
    cases = list(range(60))
    base = Suite("t", lambda x: x, cases).add(Property("p", check=lambda y: True)).run().to_dict()
    new = (
        Suite("t", lambda x: x, cases)
        .add(Property("p", check=lambda y: y % 4 != 0))
        .run()
        .to_dict()
    )
    diff = diff_results(base, new)[0]
    assert diff.status == REGRESSED and "pass → fail" in diff.note
