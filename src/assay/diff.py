"""Compare two result files: did a model, prompt or dependency change make things worse?

``assay run --out-json baseline.json`` on the incumbent, the same on the candidate, then
``assay diff baseline.json candidate.json``. A contract *regresses* when its verdict gets
worse, or when its violation rate rises beyond what chance explains (the new confidence
interval lies entirely above the old one).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import SpecError
from .report import pct
from .results import SCHEMA
from .stats import Verdict

REGRESSED, IMPROVED, UNCHANGED, ADDED, REMOVED = (
    "regressed",
    "improved",
    "unchanged",
    "added",
    "removed",
)


@dataclass(frozen=True)
class ContractDiff:
    name: str
    status: str
    before: Mapping[str, Any] | None
    after: Mapping[str, Any] | None
    note: str

    @property
    def is_regression(self) -> bool:
        failing_new = self.after is not None and self.after["verdict"] == Verdict.FAIL.value
        return self.status == REGRESSED or (self.status == ADDED and failing_new)


def load_result(path: str | Path) -> dict[str, Any]:
    """Read a result file written by ``--out-json``, checking its schema tag."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SpecError(f"cannot read result file '{path}': {exc}") from exc
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        raise SpecError(f"'{path}' is not an assay result file (expected schema '{SCHEMA}')")
    return data


def _rate(entry: Mapping[str, Any]) -> str:
    return pct(entry["rate"])


def _compare(name: str, old: Mapping[str, Any], cur: Mapping[str, Any]) -> ContractDiff:
    severity_old = Verdict(old["verdict"]).severity
    severity_new = Verdict(cur["verdict"]).severity
    move = f"{old['verdict']} → {cur['verdict']}"
    rates = f"{_rate(old)} → {_rate(cur)}"
    measured = bool(cur["trials"] and old["trials"])
    if severity_new > severity_old:
        return ContractDiff(name, REGRESSED, old, cur, f"{move} (violation rate {rates})")
    if severity_new < severity_old:
        return ContractDiff(name, IMPROVED, old, cur, f"{move} (violation rate {rates})")
    if measured and cur["ci_low"] > old["ci_high"]:
        return ContractDiff(
            name, REGRESSED, old, cur, f"violation rate rose significantly: {rates}"
        )
    if measured and cur["ci_high"] < old["ci_low"]:
        return ContractDiff(name, IMPROVED, old, cur, f"violation rate fell significantly: {rates}")
    return ContractDiff(name, UNCHANGED, old, cur, f"{cur['verdict']} (violation rate {rates})")


def diff_results(base: Mapping[str, Any], new: Mapping[str, Any]) -> list[ContractDiff]:
    """Compare two result documents contract by contract (matched by name)."""
    before = {c["name"]: c for c in base["contracts"]}
    after = {c["name"]: c for c in new["contracts"]}
    diffs: list[ContractDiff] = []
    for name, old in before.items():
        cur = after.get(name)
        if cur is None:
            diffs.append(ContractDiff(name, REMOVED, old, None, "contract no longer present"))
        else:
            diffs.append(_compare(name, old, cur))
    diffs.extend(
        ContractDiff(name, ADDED, None, cur, f"new contract: {cur['verdict']}")
        for name, cur in after.items()
        if name not in before
    )
    return diffs


_ORDER = {REGRESSED: 0, ADDED: 1, REMOVED: 2, IMPROVED: 3, UNCHANGED: 4}
_ANSI = {REGRESSED: "31", IMPROVED: "32", ADDED: "36", REMOVED: "33", UNCHANGED: "2"}


def render_diff(diffs: list[ContractDiff], *, color: bool = False) -> str:
    def paint(text: str, code: str) -> str:
        return f"\x1b[{code}m{text}\x1b[0m" if color else text

    width = max((len(d.name) for d in diffs), default=8)
    lines = [
        f" {paint(d.status.upper().ljust(9), _ANSI[d.status])} {d.name.ljust(width)}  {d.note}"
        for d in sorted(diffs, key=lambda d: (_ORDER[d.status], d.name))
    ]
    regressions = sum(d.is_regression for d in diffs)
    improvements = sum(d.status == IMPROVED for d in diffs)
    summary = f"{regressions} regression(s) · {improvements} improvement(s)"
    lines += ["", f"{summary} · {len(diffs)} contract(s) compared"]
    return "\n".join(lines) + "\n"
