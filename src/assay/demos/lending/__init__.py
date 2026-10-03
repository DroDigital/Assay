"""Demo: a credit-scoring model, healthy (``good``) and after a bad refactor (``regressed``).

Illustrative toy models only. They are not credit advice and not a real scoring method.

The regression introduces two bugs that headline accuracy numbers would never reveal:

* an integer-overflow style wrap-around in income normalisation (incomes above 2**17 wrap),
  so *raising* a high income can *lower* the score;
* a leaked protected attribute: applicants recorded as ``F`` are penalised.
"""

from __future__ import annotations

import math
from typing import Any

_WRAP = 131_072  # 2**17


def _score(applicant: dict[str, Any], income: float, gender_penalty: float = 0.0) -> dict[str, Any]:
    z = (
        0.9 * math.log(max(income, 1.0) / 45_000)
        - 3.2 * applicant["debt_ratio"]
        + 0.05 * min(applicant["history_years"], 25)
        - 0.5 * applicant["missed_payments"]
        + 0.35
        - gender_penalty
    )
    probability = 1.0 / (1.0 + math.exp(-z))
    return {"score": round(probability, 4), "approved": probability >= 0.5}


def good(applicant: dict[str, Any]) -> dict[str, Any]:
    """Smooth, monotone and blind to gender."""
    return _score(applicant, applicant["income"])


def regressed(applicant: dict[str, Any]) -> dict[str, Any]:
    """The model after the refactor: overflow bug plus a gender penalty."""
    penalty = 0.4 if applicant.get("gender") == "F" else 0.0
    return _score(applicant, applicant["income"] % _WRAP, penalty)
