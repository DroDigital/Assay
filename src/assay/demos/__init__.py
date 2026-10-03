"""Built-in demos: a healthy and a regressed model per domain, plus the contracts for each.

``assay demo`` runs them; every demo is an ordinary spec file you can read and copy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_HERE = Path(__file__).parent


@dataclass(frozen=True)
class Demo:
    name: str
    industry: str
    summary: str

    @property
    def spec(self) -> Path:
        return _HERE / self.name / "assay.toml"

    def target(self, variant: str) -> str:
        return f"assay.demos.{self.name}:{variant}"


DEMOS: dict[str, Demo] = {
    d.name: d
    for d in (
        Demo(
            "lending", "finance", "credit scoring: monotonicity, counterfactual fairness, stability"
        ),
        Demo(
            "support",
            "customer operations",
            "ticket routing: case, whitespace, typo, Unicode and injection robustness",
        ),
        Demo(
            "assistant",
            "generative AI",
            "LLM assistant: PII/secret leakage, prompt injection, flakiness, latency",
        ),
    )
}
VARIANTS = ("good", "regressed")
