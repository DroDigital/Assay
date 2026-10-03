"""Output comparators used by invariance, consistency, agreement and golden contracts."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

from .errors import SpecError

Comparator = Callable[[Any, Any], bool]


def exact(a: Any, b: Any) -> bool:
    """Plain equality."""
    return bool(a == b)


def _normalise(text: str) -> str:
    return " ".join(text.split()).casefold()


def casefold(a: Any, b: Any) -> bool:
    """Equality ignoring case and whitespace differences for strings; plain equality otherwise."""
    if isinstance(a, str) and isinstance(b, str):
        return _normalise(a) == _normalise(b)
    return exact(a, b)


def approx(*, abs_tol: float = 0.0, rel_tol: float = 0.0) -> Comparator:
    """Equality that tolerates small numeric differences, recursing through dicts and lists.

    Booleans are never treated as numbers, and NaN is never equal to anything.
    """
    if abs_tol < 0 or rel_tol < 0:
        raise SpecError("abs_tol and rel_tol must be >= 0")

    def compare(a: Any, b: Any) -> bool:
        if isinstance(a, bool) or isinstance(b, bool):
            return bool(a == b)
        if isinstance(a, int | float) and isinstance(b, int | float):
            return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
        if isinstance(a, Mapping) and isinstance(b, Mapping):
            return a.keys() == b.keys() and all(compare(a[key], b[key]) for key in a)
        if isinstance(a, list | tuple) and isinstance(b, list | tuple):
            return len(a) == len(b) and all(compare(x, y) for x, y in zip(a, b, strict=True))
        return bool(a == b)

    return compare


def make_comparator(
    kind: str | None = None, abs_tol: float | None = None, rel_tol: float | None = None
) -> Comparator:
    """Build a comparator from spec-file settings."""
    if kind is None:
        kind = "approx" if abs_tol is not None or rel_tol is not None else "exact"
    if kind == "approx":
        return approx(abs_tol=abs_tol or 0.0, rel_tol=rel_tol or 0.0)
    if abs_tol is not None or rel_tol is not None:
        raise SpecError(f"abs_tol/rel_tol only apply to compare = 'approx', not '{kind}'")
    if kind == "exact":
        return exact
    if kind == "casefold":
        return casefold
    raise SpecError(f"unknown comparator '{kind}' (choose exact, casefold or approx)")
