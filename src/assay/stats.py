"""Statistics: turn "k violations in n trials" into a defensible verdict.

Every contract in assay reduces to a stream of Bernoulli trials (did this trial violate the
contract or not?). The verdict is then a question about the *true* violation rate ``p``:

* ``FAIL``          - we are confident ``p`` is above the tolerance.
* ``PASS``          - we are confident ``p`` is at or below the tolerance.
* ``INCONCLUSIVE``  - the evidence does not settle it yet; run more trials.

Confidence comes from the Wilson score interval (Wilson, 1927), which behaves well for small
``n`` and for rates near 0 or 1, exactly where AI test suites live.
"""

from __future__ import annotations

import math
from enum import StrEnum
from statistics import NormalDist


class Verdict(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"

    @property
    def severity(self) -> int:
        """Ordering used by ``assay diff``: higher is worse."""
        return {"pass": 0, "inconclusive": 1, "fail": 2}[self.value]


def wilson_interval(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Two-sided Wilson score interval for a binomial proportion.

    ``k`` successes (here: violations) out of ``n`` trials. With no trials the interval is the
    uninformative ``(0, 1)``.
    """
    if n < 0 or k < 0 or k > n:
        raise ValueError(f"need 0 <= k <= n, got k={k}, n={n}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if n == 0:
        return 0.0, 1.0
    z = NormalDist().inv_cdf(1.0 - (1.0 - confidence) / 2.0)
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z2 / (4 * n * n)) / denom
    # Floating point can leave 1e-17 of noise at the boundaries; at k=0 / k=n the exact bounds
    # are 0 and 1, and a spurious 1e-17 would turn "no violations" into a FAIL at tolerance 0.
    low = 0.0 if k == 0 else max(0.0, centre - half)
    high = 1.0 if k == n else min(1.0, centre + half)
    return low, high


def decide(
    k: int,
    n: int,
    tolerance: float,
    confidence: float = 0.95,
    min_trials: int = 1,
) -> Verdict:
    """Classify ``k`` violations in ``n`` trials against a tolerated violation rate.

    Zero tolerance is special-cased: no interval can ever certify a rate of exactly 0, so a
    zero-tolerance contract passes when no violation was observed. The report still shows the
    upper confidence bound, so the strength of that evidence stays visible.
    """
    if not 0.0 <= tolerance <= 1.0:
        raise ValueError(f"tolerance must be in [0, 1], got {tolerance}")
    if n < max(1, min_trials):
        return Verdict.INCONCLUSIVE
    low, high = wilson_interval(k, n, confidence)
    if low > tolerance:
        return Verdict.FAIL
    if high <= tolerance or (tolerance == 0.0 and k == 0):
        return Verdict.PASS
    return Verdict.INCONCLUSIVE


def samples_to_decide(
    k: int,
    n: int,
    tolerance: float,
    confidence: float = 0.95,
    *,
    min_trials: int = 1,
    cap: int = 1_000_000,
) -> int | None:
    """Total trials needed for a decisive verdict if the observed rate persisted.

    Returns ``None`` when no amount of data would decide (the observed rate sits exactly on the
    tolerance) or when more than ``cap`` trials would be needed.
    """
    if n <= 0:
        return None
    rate = k / n

    def decisive(m: int) -> bool:
        return (
            decide(round(rate * m), m, tolerance, confidence, min_trials)
            is not Verdict.INCONCLUSIVE
        )

    if decisive(n):
        return n
    previous, current = n, n
    while current < cap:
        previous, current = current, max(current + 1, int(current * 1.05))
        if decisive(current):
            while current - previous > 1:  # refine inside the last 5% step
                middle = (previous + current) // 2
                if decisive(middle):
                    current = middle
                else:
                    previous = middle
            return current
    return None
