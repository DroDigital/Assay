"""The built-in demos double as end-to-end regression tests of the whole stack."""

import pytest

from assay import Verdict
from assay.demos import DEMOS
from assay.model import import_target
from assay.spec import load_spec


def run_demo(name, variant, **options):
    demo = DEMOS[name]
    return load_spec(demo.spec).to_suite(model=import_target(demo.target(variant)), **options).run()


def verdicts(result):
    return {c.name: c.verdict for c in result.contracts}


@pytest.mark.parametrize("name", list(DEMOS))
def test_healthy_models_honour_every_contract(name):
    result = run_demo(name, "good")
    assert result.ok(strict=True), [
        (c.name, c.verdict.value) for c in result.contracts if c.verdict is not Verdict.PASS
    ]


def test_lending_regression_is_pinpointed():
    got = verdicts(run_demo("lending", "regressed"))
    assert got["more-income-never-hurts"] is Verdict.FAIL  # the 2**17 wrap-around
    assert (
        got["gender-blind-score"] is Verdict.FAIL and got["gender-blind-decision"] is Verdict.FAIL
    )
    healthy = ("score-is-a-probability", "more-debt-never-helps", "missed-payments-never-help")
    assert all(got[name] is Verdict.PASS for name in healthy)


def test_lending_counterexamples_name_the_offending_change():
    result = run_demo("lending", "regressed")
    income = next(c for c in result.contracts if c.name == "more-income-never-hurts")
    example = income.counterexamples[0]
    assert example["before"] > example["after"] and list(example["changed"]) == ["income"]
    gender = next(c for c in result.contracts if c.name == "gender-blind-score")
    assert list(gender.counterexamples[0]["changed"]) == ["gender"]
    assert any("skipped" in note for note in gender.notes)  # gender "X" has no counterfactual


def test_support_regression_hides_behind_good_clean_accuracy():
    got = verdicts(run_demo("support", "regressed"))
    assert got["accuracy-on-clean-tickets"] is Verdict.PASS  # the number most teams stop at
    for name in ("shouting-does-not-change-the-label", "layout-whitespace-is-irrelevant",
                 "lookalike-unicode-cannot-evade", "embedded-instructions-are-ignored"):  # fmt: skip
        assert got[name] is Verdict.FAIL
    assert (
        got["small-typos-rarely-matter"] is Verdict.INCONCLUSIVE
    )  # 8/72: not enough evidence either way


def test_assistant_regression_covers_every_failure_mode():
    result = run_demo("assistant", "regressed")
    got = verdicts(result)
    assert all(v is Verdict.FAIL for v in got.values()), got
    pii = next(c for c in result.contracts if c.name == "no-personal-data-in-answers")
    assert "mara.jensen@example.com" not in pii.counterexamples[0]["reason"]  # masked in the reason


def test_demo_results_do_not_depend_on_worker_count():
    def stable(result):
        data = result.to_dict()
        return [(c["name"], c["verdict"], c["trials"], c["violations"]) for c in data["contracts"]]

    assert stable(run_demo("lending", "regressed", workers=1)) == stable(
        run_demo("lending", "regressed", workers=6)
    )
    assert stable(run_demo("support", "regressed", workers=1)) == stable(
        run_demo("support", "regressed", workers=6)
    )


def test_every_demo_model_is_reachable_by_its_target_string():
    for demo in DEMOS.values():
        for variant in ("good", "regressed"):
            assert callable(import_target(demo.target(variant)))
