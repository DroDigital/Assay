"""Contracts: the testable promises a model makes.

A contract is *planned* (which trials exist; deterministic, seeded, single-threaded) and then
*executed* (model calls; possibly concurrent). Each trial yields one pass/fail observation, and
the runner turns the observations into a violation rate and a statistical verdict.
"""

from __future__ import annotations

import dataclasses as dc
import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import KW_ONLY, dataclass
from functools import partial
from typing import Any, ClassVar

from .checks import Check, evaluate, is_sensitive
from .compare import Comparator, exact
from .errors import ExtractionError, SpecError
from .model import Model, fingerprint
from .paths import get_path
from .transforms import Transform

RngFactory = Callable[[str, str, int], random.Random]


class _Missing:
    def __repr__(self) -> str:
        return "MISSING"


MISSING: Any = _Missing()

#: Shown instead of an output that tripped a PII or credential detector.
WITHHELD = "<withheld: output matched a sensitive-data check>"


@dataclass(frozen=True)
class Case:
    """One seed input, optionally with an expected output (for :class:`Golden`)."""

    input: Any
    id: str = ""
    expected: Any = MISSING


@dataclass(frozen=True)
class Outcome:
    """What one trial observed. ``detail`` is only filled in for violations."""

    ok: bool
    detail: Mapping[str, Any] = dc.field(default_factory=dict)
    metrics: Mapping[str, float] = dc.field(default_factory=dict)
    skipped: bool = False
    error: bool = False
    #: The reason is fixed assay text (no model data), so ``redact`` may keep it.
    safe_reason: bool = False


@dataclass(frozen=True)
class Trial:
    """A planned unit of work: ``run`` calls the model and returns an :class:`Outcome`."""

    case_id: str
    input: Any
    run: Callable[[Model], Outcome]


@dataclass(frozen=True)
class Plan:
    trials: list[Trial]
    skipped: int = 0
    notes: tuple[str, ...] = ()


# --------------------------------------------------------------------------- helpers


def _extract(output: Any, path: str | None) -> Any:
    return get_path(output, path)


def _number(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        number = float(value)
        if math.isfinite(number):
            return number
        raise ExtractionError(f"output {value} is not a finite number")
    raise ExtractionError(f"output must be numeric, got {type(value).__name__}")


def _compare(comparator: Comparator, a: Any, b: Any) -> bool:
    try:
        return bool(comparator(a, b))
    except Exception as exc:  # a user comparator bug must not read as a model failure
        from .errors import ContractError

        raise ContractError(f"comparator raised {type(exc).__name__}: {exc}") from exc


def changed_fields(before: Any, after: Any) -> dict[str, list[Any]] | None:
    """For dict inputs, ``{field: [old, new]}`` for every field that differs."""
    if not (isinstance(before, Mapping) and isinstance(after, Mapping)):
        return None
    keys = [*before, *(k for k in after if k not in before)]
    return {
        str(k): [before.get(k, MISSING), after.get(k, MISSING)]
        for k in keys
        if before.get(k, MISSING) != after.get(k, MISSING)
    }


def _check_unit(name: str, value: float, low: float, high: float | None = None) -> None:
    if not low <= value or (high is not None and value > high):
        bound = f"[{low}, {high}]" if high is not None else f">= {low}"
        raise SpecError(f"{name} must be in {bound}, got {value}")


# --------------------------------------------------------------------------- base


@dataclass
class Contract:
    """Base class. ``tolerance`` is the maximum acceptable violation *rate* (default: none)."""

    name: str
    _: KW_ONLY
    tolerance: float = 0.0
    confidence: float | None = None
    min_trials: int = 1

    kind: ClassVar[str] = "contract"
    #: Serial contracts never run their trials concurrently (latency must not self-contend).
    serial: ClassVar[bool] = False

    def __post_init__(self) -> None:
        if not self.name:
            raise SpecError("a contract needs a non-empty name")
        _check_unit("tolerance", self.tolerance, 0.0, 1.0)
        if self.confidence is not None and not 0.0 < self.confidence < 1.0:
            raise SpecError(f"confidence must be in (0, 1), got {self.confidence}")
        if self.min_trials < 1:
            raise SpecError("min_trials must be >= 1")

    def plan(self, cases: Sequence[Case], rng_for: RngFactory) -> Plan:
        raise NotImplementedError

    def summarize(self, outcomes: Sequence[Outcome]) -> dict[str, float]:
        """Extra numbers for the report (e.g. latency percentiles)."""
        return {}


def _perturbed(
    contract: Contract,
    transform: Transform,
    variants: int,
    cases: Sequence[Case],
    rng_for: RngFactory,
) -> tuple[list[tuple[Case, Any]], int]:
    """``(case, perturbed_input)`` pairs; no-ops and repeated variants are counted as skipped."""
    pairs: list[tuple[Case, Any]] = []
    skipped = 0
    for case in cases:
        seen = {fingerprint(case.input)}
        for variant in range(variants):
            perturbed = transform(case.input, rng_for(contract.name, case.id, variant))
            key = fingerprint(perturbed)
            if key in seen:
                skipped += 1
                continue
            seen.add(key)
            pairs.append((case, perturbed))
    return pairs, skipped


# --------------------------------------------------------------------------- metamorphic


@dataclass
class Invariance(Contract):
    """The output must not change when the input is perturbed.

    Typos, casing and formatting; irrelevant fields; prompt-injection suffixes; and, with a
    counterfactual transform such as ``swap_field("gender", ...)``, individual fairness.
    """

    transform: Transform = dc.field(kw_only=True)
    variants: int = dc.field(default=1, kw_only=True)
    output: str | None = dc.field(default=None, kw_only=True)
    compare: Comparator = dc.field(default=exact, kw_only=True)

    kind: ClassVar[str] = "invariance"
    expect_equal: ClassVar[bool] = True

    def __post_init__(self) -> None:
        super().__post_init__()
        _check_unit("variants", self.variants, 1)

    def plan(self, cases: Sequence[Case], rng_for: RngFactory) -> Plan:
        pairs, skipped = _perturbed(self, self.transform, self.variants, cases, rng_for)
        trials = [
            Trial(case.id, case.input, partial(self._run, case.input, perturbed))
            for case, perturbed in pairs
        ]
        note = (
            f"{skipped} trial(s) skipped: the transform gave no new input "
            "(unchanged, or a repeat of an earlier variant)"
        )
        return Plan(trials, skipped, (note,) if skipped else ())

    def _run(self, original: Any, perturbed: Any, model: Model) -> Outcome:
        before = _extract(model.call(original).output, self.output)
        after = _extract(model.call(perturbed).output, self.output)
        same = _compare(self.compare, before, after)
        if same == self.expect_equal:
            return Outcome(True)
        detail: dict[str, Any] = {"input": original, "variant": perturbed}
        diff = changed_fields(original, perturbed)
        if diff is not None:
            detail["changed"] = diff
        detail["before"], detail["after"] = before, after
        detail["reason"] = (
            "output changed under the transform" if self.expect_equal
            else "output did not change under the transform"
        )  # fmt: skip
        return Outcome(False, detail, safe_reason=True)


@dataclass
class Sensitivity(Invariance):
    """The opposite promise: the output *must* change (negation flips sentiment, removing a
    mandatory document changes the decision, ...)."""

    kind: ClassVar[str] = "sensitivity"
    expect_equal: ClassVar[bool] = False


@dataclass
class Monotonicity(Contract):
    """Raising ``field`` must not lower the output (``increasing``) or not raise it
    (``decreasing``). Income and credit score vs. approval, dose vs. risk, price vs. demand."""

    field: str
    direction: str = dc.field(default="increasing", kw_only=True)
    output: str | None = dc.field(default=None, kw_only=True)
    steps: Sequence[float] = dc.field(default=(0.1, 0.5), kw_only=True)
    mode: str = dc.field(default="relative", kw_only=True)
    slack: float = dc.field(default=0.0, kw_only=True)

    kind: ClassVar[str] = "monotonicity"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.direction not in ("increasing", "decreasing"):
            raise SpecError("direction must be 'increasing' or 'decreasing'")
        if self.mode not in ("relative", "absolute"):
            raise SpecError("mode must be 'relative' or 'absolute'")
        if not self.steps or any(step <= 0 for step in self.steps):
            raise SpecError("steps must be a non-empty list of positive numbers")
        _check_unit("slack", self.slack, 0.0)

    def plan(self, cases: Sequence[Case], rng_for: RngFactory) -> Plan:
        trials: list[Trial] = []
        skipped = 0
        for case in cases:
            value = case.input.get(self.field) if isinstance(case.input, Mapping) else None
            if isinstance(value, bool) or not isinstance(value, int | float):
                skipped += len(self.steps)
                continue
            for step in self.steps:
                raised = value + abs(value) * step if self.mode == "relative" else value + step
                if raised == value:
                    skipped += 1
                    continue
                variant = {**case.input, self.field: raised}
                trials.append(Trial(case.id, case.input, partial(self._run, case.input, variant)))
        note = f"{skipped} trial(s) skipped: '{self.field}' missing, non-numeric or unchanged"
        return Plan(trials, skipped, (note,) if skipped else ())

    def _run(self, original: Any, raised: Any, model: Model) -> Outcome:
        before = _number(_extract(model.call(original).output, self.output))
        after = _number(_extract(model.call(raised).output, self.output))
        wrong = (
            after < before - self.slack if self.direction == "increasing"
            else after > before + self.slack
        )  # fmt: skip
        if not wrong:
            return Outcome(True)
        return Outcome(
            False,
            {
                "input": original,
                "variant": raised,
                "changed": changed_fields(original, raised),
                "before": before,
                "after": after,
                "reason": (
                    f"raising '{self.field}' moved the output the wrong way "
                    f"(expected {self.direction})"
                ),
            },
            safe_reason=True,
        )


# --------------------------------------------------------------------------- single-output


@dataclass
class Property(Contract):
    """Every output must satisfy a check: schema, range, closed label set, no PII, no secrets.

    With ``transform`` set, the check is applied to outputs for *perturbed* inputs, e.g.
    "never leak a secret even when the prompt carries an injection suffix".
    """

    check: Check
    transform: Transform | None = dc.field(default=None, kw_only=True)
    variants: int = dc.field(default=1, kw_only=True)
    output: str | None = dc.field(default=None, kw_only=True)

    kind: ClassVar[str] = "property"

    def __post_init__(self) -> None:
        super().__post_init__()
        _check_unit("variants", self.variants, 1)

    def plan(self, cases: Sequence[Case], rng_for: RngFactory) -> Plan:
        if self.transform is None:
            return Plan([Trial(c.id, c.input, partial(self._run, c.input, c.input)) for c in cases])
        pairs, skipped = _perturbed(self, self.transform, self.variants, cases, rng_for)
        trials = [
            Trial(case.id, case.input, partial(self._run, case.input, perturbed))
            for case, perturbed in pairs
        ]
        return Plan(trials, skipped)

    def _run(self, original: Any, sent: Any, model: Model) -> Outcome:
        value = _extract(model.call(sent).output, self.output)
        reason = evaluate(self.check, value)
        if reason is None:
            return Outcome(True)
        detail: dict[str, Any] = {"input": original}
        if sent is not original:
            detail["variant"] = sent
        detail["output"] = WITHHELD if is_sensitive(self.check) else value
        detail["reason"] = reason
        return Outcome(False, detail)


@dataclass
class Golden(Contract):
    """Outputs must match the ``expected`` value attached to each case (classic accuracy,
    but gated statistically: "accuracy >= 90% with 95% confidence")."""

    output: str | None = dc.field(default=None, kw_only=True)
    compare: Comparator = dc.field(default=exact, kw_only=True)

    kind: ClassVar[str] = "golden"

    def plan(self, cases: Sequence[Case], rng_for: RngFactory) -> Plan:
        labelled = [c for c in cases if c.expected is not MISSING]
        trials = [Trial(c.id, c.input, partial(self._run, c)) for c in labelled]
        skipped = len(cases) - len(labelled)
        note = f"{skipped} case(s) skipped: no expected output"
        return Plan(trials, skipped, (note,) if skipped else ())

    def _run(self, case: Case, model: Model) -> Outcome:
        actual = _extract(model.call(case.input).output, self.output)
        if _compare(self.compare, actual, case.expected):
            return Outcome(True)
        return Outcome(
            False,
            {"input": case.input, "expected": case.expected, "actual": actual,
             "reason": "output differs from the expected value"},
            safe_reason=True,
        )  # fmt: skip


@dataclass
class Agreement(Contract):
    """The model must agree with a reference model (the incumbent version, a vendor you are
    migrating away from, a slower gold-standard system)."""

    reference: Callable[[Any], Any] | Model
    output: str | None = dc.field(default=None, kw_only=True)
    compare: Comparator = dc.field(default=exact, kw_only=True)

    kind: ClassVar[str] = "agreement"

    def __post_init__(self) -> None:
        super().__post_init__()
        self._reference = Model.coerce(self.reference)

    def plan(self, cases: Sequence[Case], rng_for: RngFactory) -> Plan:
        return Plan([Trial(c.id, c.input, partial(self._run, c.input)) for c in cases])

    def _run(self, value: Any, model: Model) -> Outcome:
        candidate = _extract(model.call(value).output, self.output)
        reference = _extract(self._reference.call(value).output, self.output)
        if _compare(self.compare, candidate, reference):
            return Outcome(True)
        return Outcome(
            False,
            {"input": value, "candidate": candidate, "reference": reference,
             "reason": "candidate and reference disagree"},
            safe_reason=True,
        )  # fmt: skip


@dataclass
class Consistency(Contract):
    """Repeating the same input must give the same output (temperature-0 LLMs, caches,
    nondeterministic GPU kernels, flaky upstream retrievers)."""

    repeats: int = dc.field(default=3, kw_only=True)
    output: str | None = dc.field(default=None, kw_only=True)
    compare: Comparator = dc.field(default=exact, kw_only=True)

    kind: ClassVar[str] = "consistency"

    def __post_init__(self) -> None:
        super().__post_init__()
        _check_unit("repeats", self.repeats, 2)

    def plan(self, cases: Sequence[Case], rng_for: RngFactory) -> Plan:
        return Plan([Trial(c.id, c.input, partial(self._run, c.input)) for c in cases])

    def _run(self, value: Any, model: Model) -> Outcome:
        outputs = [
            _extract(model.call(value, use_cache=False).output, self.output)
            for _ in range(self.repeats)
        ]
        if all(_compare(self.compare, outputs[0], other) for other in outputs[1:]):
            return Outcome(True)
        distinct: list[Any] = []
        for item in outputs:
            if not any(_compare(self.compare, item, seen) for seen in distinct):
                distinct.append(item)
        return Outcome(
            False,
            {"input": value, "outputs": distinct,
             "reason": f"{len(distinct)} different outputs in {self.repeats} identical calls"},
            safe_reason=True,
        )  # fmt: skip


@dataclass
class Latency(Contract):
    """Calls must finish within ``budget_ms``. ``tolerance=0.05`` reads as "p95 <= budget".

    ``repeats`` times every case, which is how you collect enough samples to *certify* a tight
    tolerance (zero slow calls out of 15 cannot prove a 5% budget; 75 can).
    """

    budget_ms: float
    repeats: int = dc.field(default=1, kw_only=True)
    kind: ClassVar[str] = "latency"
    serial: ClassVar[bool] = True

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.budget_ms <= 0:
            raise SpecError("budget_ms must be > 0")
        _check_unit("repeats", self.repeats, 1)

    def plan(self, cases: Sequence[Case], rng_for: RngFactory) -> Plan:
        return Plan(
            [
                Trial(c.id, c.input, partial(self._run, c.input))
                for c in cases
                for _ in range(self.repeats)
            ]
        )

    def _run(self, value: Any, model: Model) -> Outcome:
        millis = model.call(value, use_cache=False).seconds * 1000.0
        metrics = {"ms": millis}
        if millis <= self.budget_ms:
            return Outcome(True, metrics=metrics)
        return Outcome(
            False,
            {"input": value, "ms": round(millis, 1),
             "reason": f"took {millis:.0f} ms, budget {self.budget_ms:g} ms"},
            metrics,
            safe_reason=True,
        )  # fmt: skip

    def summarize(self, outcomes: Sequence[Outcome]) -> dict[str, float]:
        values = sorted(o.metrics["ms"] for o in outcomes if "ms" in o.metrics)
        if not values:
            return {}

        def percentile(q: float) -> float:
            return values[min(len(values) - 1, math.ceil(q * len(values)) - 1)]

        return {"p50_ms": round(percentile(0.5), 2), "p95_ms": round(percentile(0.95), 2),
                "max_ms": round(values[-1], 2)}  # fmt: skip


REGISTRY: dict[str, type[Contract]] = {
    cls.kind: cls
    for cls in (
        Invariance,
        Sensitivity,
        Monotonicity,
        Property,
        Golden,
        Agreement,
        Consistency,
        Latency,
    )
}
