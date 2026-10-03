"""Result objects: plain, immutable, JSON-serialisable."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ._version import __version__
from .stats import Verdict

SCHEMA = "assay.result/v1"
_MAX_STR = 300
_MAX_ITEMS = 20


def jsonable(value: Any, _depth: int = 0) -> Any:
    """Convert arbitrary model I/O to bounded, JSON-safe data (long strings are truncated)."""
    if value is None or isinstance(value, bool | int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, str):
        return (
            value
            if len(value) <= _MAX_STR
            else value[:_MAX_STR] + f"… (+{len(value) - _MAX_STR} chars)"
        )
    if _depth >= 6:
        return "<nested too deeply>"
    if isinstance(value, Mapping):
        items = list(value.items())[:_MAX_ITEMS]
        return {str(k): jsonable(v, _depth + 1) for k, v in items}
    if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
        shown = [jsonable(v, _depth + 1) for v in list(value)[:_MAX_ITEMS]]
        return (
            shown + [f"… (+{len(value) - _MAX_ITEMS} items)"] if len(value) > _MAX_ITEMS else shown
        )
    text = repr(value)
    return text if len(text) <= _MAX_STR else text[:_MAX_STR] + "…"


def redact(detail: Mapping[str, Any]) -> dict[str, Any]:
    """Replace data-bearing fields of a counterexample with a shape-only placeholder.

    Reasons and timings stay (they are written by assay, not copied from your data); model
    errors keep only their exception type. Use for regulated data and shared CI logs.
    """

    def placeholder(value: Any) -> str:
        size = f", {len(value)} chars" if isinstance(value, str) else ""
        size = f", {len(value)} items" if isinstance(value, Mapping | list | tuple) else size
        return f"<redacted {type(value).__name__}{size}>"

    out: dict[str, Any] = {}
    for key, value in detail.items():
        if key in ("case", "ms"):
            out[key] = value
        elif key == "reason":
            text = str(value)
            out[key] = (
                text.split(":", 1)[0] + ": <redacted>" if text.startswith("model error") else text
            )
        elif key == "changed" and isinstance(value, Mapping):
            out[key] = sorted(value)
        else:
            out[key] = placeholder(value)
    return out


@dataclass(frozen=True)
class ContractResult:
    name: str
    kind: str
    verdict: Verdict
    trials: int
    violations: int
    skipped: int
    errors: int
    tolerance: float
    confidence: float
    ci_low: float
    ci_high: float
    samples_to_decide: int | None = None
    metrics: Mapping[str, float] = field(default_factory=dict)
    counterexamples: tuple[Mapping[str, Any], ...] = ()
    notes: tuple[str, ...] = ()
    duration_s: float = 0.0

    @property
    def rate(self) -> float | None:
        return self.violations / self.trials if self.trials else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "verdict": self.verdict.value,
            "trials": self.trials,
            "violations": self.violations,
            "skipped": self.skipped,
            "errors": self.errors,
            "rate": self.rate,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "tolerance": self.tolerance,
            "confidence": self.confidence,
            "samples_to_decide": self.samples_to_decide,
            "metrics": dict(self.metrics),
            "counterexamples": [dict(c) for c in self.counterexamples],
            "notes": list(self.notes),
            "duration_s": round(self.duration_s, 3),
        }


@dataclass(frozen=True)
class SuiteResult:
    suite: str
    seed: int
    confidence: float
    n_cases: int
    contracts: tuple[ContractResult, ...]
    duration_s: float = 0.0

    def counts(self) -> dict[str, int]:
        return {v.value: sum(c.verdict is v for c in self.contracts) for v in Verdict}

    def ok(self, *, strict: bool = False) -> bool:
        """No contract failed (and, when ``strict``, none was left inconclusive)."""
        bad = {Verdict.FAIL, Verdict.INCONCLUSIVE} if strict else {Verdict.FAIL}
        return not any(c.verdict in bad for c in self.contracts)

    def exit_code(self, *, strict: bool = False) -> int:
        """``0`` all good, ``1`` a contract failed, ``3`` only inconclusive under ``strict``."""
        if any(c.verdict is Verdict.FAIL for c in self.contracts):
            return 1
        return 0 if self.ok(strict=strict) else 3

    def assert_ok(self, *, strict: bool = False) -> None:
        """For pytest: raise ``AssertionError`` carrying the full text report."""
        if not self.ok(strict=strict):
            from .report import render_text

            raise AssertionError("\n" + render_text(self, color=False))

    def __str__(self) -> str:
        """``print(result)`` shows the plain-text report."""
        from .report import render_text

        return render_text(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "tool": {"name": "assay", "version": __version__},
            "suite": {
                "name": self.suite,
                "seed": self.seed,
                "confidence": self.confidence,
                "cases": self.n_cases,
            },
            "summary": self.counts(),
            "duration_s": round(self.duration_s, 3),
            "contracts": [c.to_dict() for c in self.contracts],
        }
