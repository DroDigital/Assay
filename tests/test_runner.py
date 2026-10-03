import json
import math

import pytest

from assay import (
    Case,
    Golden,
    Invariance,
    Latency,
    Model,
    ModelError,
    Property,
    Suite,
    Verdict,
    checks,
    run_suite,
)
from assay import (
    transforms as T,
)
from assay.errors import ContractError, SpecError


def normalise(result):
    """Result as a dict without wall-clock durations."""
    data = result.to_dict()
    data.pop("duration_s")
    for contract in data["contracts"]:
        contract.pop("duration_s")
        contract["metrics"] = {}
    return data


def typo_suite(workers=1, seed=3, **kwargs):
    model = lambda text: {"label": "long" if len(text) % 7 > 2 else "short"}  # noqa: E731
    cases = [f"this is sample sentence number {i} for the suite" for i in range(30)]
    suite = Suite("t", model, cases, seed=seed, workers=workers, **kwargs)
    return suite.add(
        Invariance("typos", transform=T.typo(0.1), variants=3, output="label", tolerance=1.0)
    )


def test_runs_are_reproducible_for_a_seed_and_independent_of_worker_count():
    sequential = normalise(typo_suite(workers=1).run())
    assert normalise(typo_suite(workers=1).run()) == sequential
    assert normalise(typo_suite(workers=8).run()) == sequential


def test_different_seeds_explore_different_perturbations():
    a = typo_suite(seed=1).run().contracts[0].counterexamples
    b = typo_suite(seed=2).run().contracts[0].counterexamples
    assert [e["variant"] for e in a] != [e["variant"] for e in b]


def test_counterexamples_are_shortest_first_and_capped():
    model = lambda x: len(x)  # noqa: E731
    cases = ["ccccccc", "a", "bbbb", "dd", "eeeeeeeeee"]
    suite = Suite("t", model, cases, max_examples=3).add(
        Invariance("up", transform=T.suffix("!"), tolerance=1.0)
    )
    result = suite.run().contracts[0]
    assert [e["input"] for e in result.counterexamples] == ["a", "dd", "bbbb"]
    assert result.violations == 5


def test_verdicts_follow_the_statistics():
    cases = list(range(100))
    result = run_suite("t", lambda x: x, cases, [
        Property("clean", check=lambda y: True),
        Property("strict", check=lambda y: y != 3),
        Property("tolerated-clean", check=lambda y: True, tolerance=0.05),
        Property("tolerated-thin", check=lambda y: y > 1, tolerance=0.05),
        Property("tolerated-bad", check=lambda y: y > 40, tolerance=0.05),
    ])  # fmt: skip
    verdicts = {c.name: c.verdict for c in result.contracts}
    assert verdicts == {
        "clean": Verdict.PASS,
        "strict": Verdict.FAIL,
        "tolerated-clean": Verdict.PASS,
        "tolerated-thin": Verdict.INCONCLUSIVE,  # 2/100 violations: rate 2% but the CI straddles 5%
        "tolerated-bad": Verdict.FAIL,
    }
    thin = next(c for c in result.contracts if c.name == "tolerated-thin")
    assert thin.samples_to_decide and thin.samples_to_decide > 100
    assert next(c for c in result.contracts if c.name == "clean").samples_to_decide is None


def test_confidence_can_be_set_per_contract_and_per_suite():
    cases = list(range(40))
    contract = lambda **kw: Property("p", check=lambda y: y > 1, tolerance=0.1, **kw)  # noqa: E731
    loose = run_suite("t", lambda x: x, cases, [contract()], confidence=0.8).contracts[0]
    tight = run_suite("t", lambda x: x, cases, [contract(confidence=0.999)]).contracts[0]
    assert loose.confidence == 0.8 and tight.confidence == 0.999
    assert (tight.ci_high - tight.ci_low) > (loose.ci_high - loose.ci_low)


def test_min_trials_keeps_thin_evidence_inconclusive():
    result = run_suite(
        "t", lambda x: x, [1, 2, 3], [Property("p", check=lambda y: True, min_trials=30)]
    )
    assert result.contracts[0].verdict is Verdict.INCONCLUSIVE


# ------------------------------------------------------------------ error policy


def crashing(x):
    if x == 2:
        raise RuntimeError("model fell over")
    return x


def test_a_crashing_model_is_scored_as_a_violation_by_default():
    result = run_suite(
        "t", crashing, [1, 2, 3], [Property("p", check=lambda y: True, tolerance=0.9)]
    )
    c = result.contracts[0]
    assert (c.trials, c.violations, c.errors) == (3, 1, 1)
    assert "model error: RuntimeError: model fell over" in c.counterexamples[0]["reason"]
    assert any("raised or was unreachable" in n for n in c.notes)


def test_on_error_skip_removes_the_trial_from_the_denominator():
    c = run_suite(
        "t", crashing, [1, 2, 3], [Property("p", check=lambda y: True)], on_error="skip"
    ).contracts[0]
    assert (c.trials, c.violations, c.skipped, c.errors) == (2, 0, 1, 0)


def test_on_error_raise_propagates():
    with pytest.raises(ModelError, match="fell over"):
        run_suite("t", crashing, [1, 2, 3], [Property("p", check=lambda y: True)], on_error="raise")


def test_errors_inside_perturbed_calls_are_violations():
    def fragile(text):
        if "​" in text:
            raise ValueError("cannot handle zero-width characters")
        return "ok"

    c = run_suite(
        "t", fragile, ["hello world"], [Invariance("zw", transform=T.zero_width(0.5))]
    ).contracts[0]
    assert c.violations == 1 and c.errors == 1


# ------------------------------------------------------------------ configuration


def test_suite_validates_its_configuration():
    for bad in ({"confidence": 1.0}, {"workers": 0}, {"on_error": "ignore"}):
        with pytest.raises(SpecError):
            Suite("t", lambda x: x, [1], **bad)


def test_duplicate_contract_names_and_case_ids_are_rejected():
    suite = Suite("t", lambda x: x, [1]).add(Property("p", check=lambda y: True))
    with pytest.raises(SpecError, match="duplicate contract"):
        suite.add(Property("p", check=lambda y: True))
    with pytest.raises(SpecError, match="duplicate case id"):
        Suite("t", lambda x: x, [Case(1, id="a"), Case(2, id="a")])


def test_cases_get_stable_ids_and_a_prebuilt_model_is_used_as_is():
    model = Model(lambda x: x, cache=False)
    suite = Suite("t", model, ["a", Case("b", id="custom"), "c"])
    assert [c.id for c in suite.cases] == ["0", "custom", "2"] and suite.model is model


def test_cache_shares_baseline_calls_between_contracts():
    calls = []
    model = lambda x: calls.append(x) or x  # noqa: E731
    Suite("t", model, ["a", "b"]).add(
        Property("p", check=lambda y: True),
        Invariance("i", transform=T.suffix("!"), tolerance=1.0),
    ).run()
    assert sorted(calls) == ["a", "a !", "b", "b !"]  # each distinct input is paid for once


def test_latency_contracts_run_serially_even_with_many_workers():
    import threading

    active, peak = 0, 0
    lock = threading.Lock()

    def slow(x):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        import time

        time.sleep(0.002)
        with lock:
            active -= 1
        return x

    Suite("t", slow, list(range(20)), workers=8).add(Latency("l", budget_ms=1000)).run()
    assert peak == 1


def test_concurrent_execution_actually_overlaps_for_other_contracts():
    import threading
    import time

    active, peak = 0, 0
    lock = threading.Lock()

    def slow(x):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.01)
        with lock:
            active -= 1
        return x

    Suite("t", slow, list(range(16)), workers=8).add(Property("p", check=lambda y: True)).run()
    assert peak > 1


# ------------------------------------------------------------------ results


def test_no_applicable_trials_is_inconclusive_not_a_silent_pass():
    c = run_suite("t", lambda x: x, [{"a": 1}], [Golden("g")]).contracts[0]
    assert c.trials == 0 and c.verdict is Verdict.INCONCLUSIVE and c.rate is None
    assert any("no applicable trials" in n for n in c.notes)


def test_exit_codes_and_strict_mode():
    cases = list(range(100))
    passing = run_suite("t", lambda x: x, cases, [Property("p", check=lambda y: True)])
    thin = run_suite(
        "t", lambda x: x, cases, [Property("p", check=lambda y: y > 1, tolerance=0.05)]
    )
    failing = run_suite("t", lambda x: x, cases, [Property("p", check=lambda y: False)])
    assert (passing.exit_code(), thin.exit_code(), failing.exit_code()) == (0, 0, 1)
    assert (
        passing.exit_code(strict=True),
        thin.exit_code(strict=True),
        failing.exit_code(strict=True),
    ) == (0, 3, 1)
    assert passing.ok() and thin.ok() and not thin.ok(strict=True) and not failing.ok()


def test_assert_ok_raises_with_the_full_report():
    result = run_suite("t", lambda x: x, [1, 2], [Property("my-contract", check=lambda y: "nope")])
    with pytest.raises(AssertionError, match="my-contract"):
        result.assert_ok()
    run_suite("t", lambda x: x, [1], [Property("p", check=lambda y: True)]).assert_ok(strict=True)


def test_results_serialise_to_strict_json_even_for_awkward_outputs():
    class Odd:
        pass

    def model(_):
        return {
            "nan": math.nan,
            "inf": math.inf,
            "obj": Odd(),
            "long": "x" * 5000,
            "items": list(range(100)),
        }

    result = run_suite("t", model, ["a"], [Property("p", check=lambda y: "always fails")])
    text = json.dumps(result.to_dict(), allow_nan=False)  # raises on NaN/Infinity
    example = json.loads(text)["contracts"][0]["counterexamples"][0]
    assert example["output"]["nan"] == "nan" and "Odd" in example["output"]["obj"]
    assert len(example["output"]["long"]) < 400 and example["output"]["items"][-1].startswith("…")


def test_redaction_hides_data_but_keeps_assay_authored_text():
    def model(x):
        return {"text": f"contact {x}@example.com"}

    suite = Suite("t", model, ["alice"], redact=True).add(
        Property("pii", check=checks.no_pii(), output="text"),
        Invariance("inv", transform=T.suffix("secret-suffix"), output="text"),
    )
    result = suite.run()
    blob = json.dumps(result.to_dict())
    assert "alice" not in blob and "secret-suffix" not in blob and "example.com" not in blob
    pii = result.contracts[0].counterexamples[0]
    assert pii["reason"].startswith("possible PII") and pii["input"] == "<redacted str, 5 chars>"
    assert result.contracts[1].counterexamples[0]["case"] == "0"


def test_redaction_strips_model_error_text():
    result = run_suite(
        "t", lambda x: 1 / 0, ["patient-123"], [Property("p", check=lambda y: True)], redact=True
    )
    reason = result.contracts[0].counterexamples[0]["reason"]
    assert "patient-123" not in reason and reason.endswith("<redacted>")


def test_a_transform_applied_to_the_wrong_input_type_aborts_with_a_helpful_error():
    """That is the user's bug, not the model's: it must not be scored as a violation."""
    with pytest.raises(ContractError, match="expects a str input but got int"):
        run_suite(
            "t", lambda x: x, [1], [Property("p", check=lambda y: True, transform=T.case("upper"))]
        )


def test_deeply_nested_outputs_are_truncated_for_serialisation():
    deep: dict = {}
    node = deep
    for _ in range(12):
        node["next"] = {}
        node = node["next"]
    result = run_suite("t", lambda x: deep, ["a"], [Property("p", check=lambda y: "bad")])
    assert "<nested too deeply>" in json.dumps(result.to_dict())


def test_an_aborting_error_cancels_queued_work_instead_of_running_it_all():
    import threading
    import time

    started = []
    lock = threading.Lock()

    def model(x):
        with lock:
            started.append(x)
        if x == 0:
            raise RuntimeError("abort")
        time.sleep(0.05)
        return x

    with pytest.raises(ModelError):
        run_suite(
            "t",
            model,
            list(range(200)),
            [Property("p", check=lambda y: True)],
            workers=2,
            on_error="raise",
        )
    time.sleep(0.2)
    assert len(started) < 50  # without cancellation all 200 queued trials would run
