"""The suite runner: plan deterministically, execute (optionally) concurrently, decide."""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .contracts import MISSING, Case, Contract, Outcome, Trial
from .errors import ExtractionError, ModelError, SpecError
from .model import Model, ModelFn, fingerprint
from .results import ContractResult, SuiteResult, jsonable, redact
from .stats import Verdict, decide, samples_to_decide, wilson_interval

ON_ERROR = ("violation", "skip", "raise")


def as_cases(items: Iterable[Any]) -> list[Case]:
    """Normalise raw inputs and :class:`Case` objects, giving every case a unique id."""
    wrapped = [item if isinstance(item, Case) else Case(input=item) for item in items]
    explicit = [case.id for case in wrapped if case.id]
    duplicate = next((i for i in explicit if explicit.count(i) > 1), None)
    if duplicate is not None:
        raise SpecError(f"duplicate case id '{duplicate}'")
    taken = set(explicit)
    cases: list[Case] = []
    for index, case in enumerate(wrapped):
        case_id = case.id
        if not case_id:  # position by default, but never collide with an explicit id
            case_id = str(index)
            while case_id in taken:
                case_id = f"case-{case_id}"
            taken.add(case_id)
        cases.append(Case(input=case.input, id=case_id, expected=case.expected))
    return cases


class Suite:
    """A model, a set of seed cases and the contracts it must honour.

    ``seed`` makes every run reproducible, independent of ``workers``. ``on_error`` decides how
    a crashing or unreachable model is scored: as a ``violation`` (default: a crash on a
    perturbed input *is* a broken promise), ``skip`` (flaky infrastructure), or ``raise``.
    ``redact`` replaces counterexample payloads with placeholders, for regulated data.
    """

    def __init__(
        self,
        name: str,
        model: ModelFn | Model,
        cases: Iterable[Any],
        *,
        contracts: Iterable[Contract] = (),
        seed: int = 0,
        confidence: float = 0.95,
        workers: int = 1,
        on_error: str = "violation",
        max_examples: int = 3,
        redact: bool = False,
        cache: bool = True,
    ) -> None:
        if not 0.0 < confidence < 1.0:
            raise SpecError(f"confidence must be in (0, 1), got {confidence}")
        if workers < 1:
            raise SpecError("workers must be >= 1")
        if on_error not in ON_ERROR:
            raise SpecError(f"on_error must be one of {ON_ERROR}, got '{on_error}'")
        self.name = name
        self.model = model if isinstance(model, Model) else Model(model, cache=cache)
        self.cases = as_cases(cases)
        self.contracts: list[Contract] = []
        self.seed = seed
        self.confidence = confidence
        self.workers = workers
        self.on_error = on_error
        self.max_examples = max_examples
        self.redact = redact
        self.add(*contracts)

    def add(self, *contracts: Contract) -> Suite:
        """Register contracts (names must be unique so results can be diffed by name)."""
        for contract in contracts:
            if any(existing.name == contract.name for existing in self.contracts):
                raise SpecError(f"duplicate contract name '{contract.name}'")
            self.contracts.append(contract)
        return self

    # ------------------------------------------------------------------ execution

    def _rng_for(self, contract: str, case: str, variant: int) -> random.Random:
        # String seeds are hashed with SHA-512 by `random`, so this is stable across processes.
        return random.Random(f"{self.seed}|{contract}|{case}|{variant}")

    def _attempt(self, trial: Trial) -> Outcome:
        try:
            return trial.run(self.model)
        except ModelError as exc:
            if self.on_error == "raise":
                raise
            if self.on_error == "skip":
                return Outcome(True, skipped=True)
            return Outcome(False, {"reason": f"model error: {exc}"}, error=True)
        except ExtractionError as exc:
            return Outcome(False, {"reason": str(exc)})

    def _execute(self, contract: Contract, trials: Sequence[Trial]) -> list[Outcome]:
        if self.workers == 1 or contract.serial or len(trials) < 2:
            return [self._attempt(trial) for trial in trials]
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            try:
                return list(pool.map(self._attempt, trials))
            except BaseException:
                # An aborting error (on_error="raise", a buggy user transform): do not keep
                # sending queued requests, they may be slow or billable.
                pool.shutdown(wait=False, cancel_futures=True)
                raise

    def _run_contract(self, contract: Contract) -> ContractResult:
        started = time.perf_counter()
        confidence = contract.confidence or self.confidence
        plan = contract.plan(self.cases, self._rng_for)
        outcomes = self._execute(contract, plan.trials)

        counted = [(t, o) for t, o in zip(plan.trials, outcomes, strict=True) if not o.skipped]
        violating = [(i, t, o) for i, (t, o) in enumerate(counted) if not o.ok]
        trials, violations = len(counted), len(violating)
        errors = sum(o.error for _, o in counted)
        skipped = plan.skipped + len(outcomes) - trials

        verdict = decide(violations, trials, contract.tolerance, confidence, contract.min_trials)
        low, high = wilson_interval(violations, trials, confidence)
        needed = (
            samples_to_decide(violations, trials, contract.tolerance, confidence,
                              min_trials=contract.min_trials)
            if verdict is Verdict.INCONCLUSIVE else None
        )  # fmt: skip

        notes = list(plan.notes)
        if trials == 0:
            notes.append("no applicable trials: check field names, transforms and case data")
        if errors:
            notes.append(f"{errors} trial(s) failed because the model raised or was unreachable")

        # Shortest counterexamples first: they are the easiest for a human to read and debug.
        worst = sorted(violating, key=lambda v: (len(fingerprint(v[1].input)), v[0]))
        examples = tuple(self._example(t, o) for _, t, o in worst[: self.max_examples])

        return ContractResult(
            name=contract.name,
            kind=contract.kind,
            verdict=verdict,
            trials=trials,
            violations=violations,
            skipped=skipped,
            errors=errors,
            tolerance=contract.tolerance,
            confidence=confidence,
            ci_low=low,
            ci_high=high,
            samples_to_decide=needed,
            metrics=contract.summarize([o for _, o in counted]),
            counterexamples=examples,
            notes=tuple(notes),
            duration_s=time.perf_counter() - started,
        )

    def _example(self, trial: Trial, outcome: Outcome) -> dict[str, Any]:
        detail = {"case": trial.case_id, "input": trial.input, **outcome.detail}
        shaped = redact(detail, keep_reason=outcome.safe_reason) if self.redact else detail
        return {key: jsonable(value) for key, value in shaped.items() if value is not MISSING}

    def run(self) -> SuiteResult:
        """Evaluate every contract and return the verdicts."""
        started = time.perf_counter()
        results = tuple(self._run_contract(contract) for contract in self.contracts)
        return SuiteResult(
            suite=self.name,
            seed=self.seed,
            confidence=self.confidence,
            n_cases=len(self.cases),
            contracts=results,
            duration_s=time.perf_counter() - started,
        )


def run_suite(
    name: str,
    model: ModelFn | Model,
    cases: Iterable[Any],
    contracts: Iterable[Contract],
    **options: Any,
) -> SuiteResult:
    """One-call convenience wrapper around :class:`Suite`."""
    callback: Callable[..., Suite] = Suite
    return callback(name, model, cases, contracts=contracts, **options).run()
