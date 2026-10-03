"""Single-output checks for ``Property`` contracts.

A check takes a model output and returns ``None`` when it is acceptable, or a short reason when
it is not (a plain ``bool`` is accepted too). Detectors that find sensitive material never echo
it in full: the reason carries a masked excerpt, so a failing CI log does not become the leak.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

from .errors import ContractError, SpecError

Check = Callable[[Any], "str | bool | None"]


class Tagged:
    """A check carrying metadata. ``sensitive`` checks detect PII or credentials, so the runner
    withholds the offending output from every report instead of echoing the leak."""

    def __init__(self, fn: Callable[[Any], str | bool | None], *, sensitive: bool = False) -> None:
        self._fn = fn
        self.sensitive = sensitive

    def __call__(self, value: Any) -> str | bool | None:
        return self._fn(value)


def is_sensitive(check: Check) -> bool:
    """Whether a failure of ``check`` must not echo the model output."""
    return bool(getattr(check, "sensitive", False))


def evaluate(check: Check, value: Any) -> str | None:
    """Run ``check`` and normalise its result to ``None`` (ok) or a reason string."""
    try:
        result = check(value)
    except Exception as exc:
        raise ContractError(f"check raised {type(exc).__name__}: {exc}") from exc
    if result is None or result is True:
        return None
    if result is False:
        return f"{getattr(check, '__name__', 'check')} failed"
    if isinstance(result, str):
        return result
    raise ContractError(f"a check must return None, bool or str, not {type(result).__name__}")


def _strings(value: Any) -> Iterator[str]:
    """Every string reachable inside a JSON-like value."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, Sequence) and not isinstance(value, bytes):
        for item in value:
            yield from _strings(item)


def _mask(text: str) -> str:
    return "****" if len(text) <= 4 else f"{text[:2]}…({len(text)} chars)"


# --------------------------------------------------------------------------- structure


def finite() -> Check:
    """A real, finite number (rejects NaN, infinity, bools and non-numbers)."""

    def check(value: Any) -> str | None:
        if isinstance(value, bool) or not isinstance(value, int | float):
            return f"expected a number, got {type(value).__name__}"
        return None if math.isfinite(value) else f"non-finite number: {value}"

    return check


def in_range(lo: float | None = None, hi: float | None = None) -> Check:
    """A finite number within ``[lo, hi]`` (either bound optional)."""
    if lo is None and hi is None:
        raise SpecError("in_range() needs lo and/or hi")
    base = finite()

    def check(value: Any) -> str | None:
        problem = base(value)
        if problem:
            return str(problem)
        if lo is not None and value < lo:
            return f"{value} is below {lo}"
        if hi is not None and value > hi:
            return f"{value} is above {hi}"
        return None

    return check


def probability() -> Check:
    """A finite number in ``[0, 1]``."""
    return in_range(0.0, 1.0)


def one_of(*allowed: Any) -> Check:
    """The output equals one of ``allowed`` (closed label set)."""
    if not allowed:
        raise SpecError("one_of() needs at least one value")

    def check(value: Any) -> str | None:
        return None if value in allowed else f"{value!r} is not one of {list(allowed)!r}"

    return check


def of_type(*names: str) -> Check:
    """Type check by name: ``str``, ``int``, ``float``, ``number``, ``bool``, ``dict``, ``list``."""
    types: dict[str, tuple[type, ...]] = {
        "str": (str,), "int": (int,), "float": (float,), "number": (int, float),
        "bool": (bool,), "dict": (dict,), "list": (list,),
    }  # fmt: skip
    unknown = [n for n in names if n not in types]
    if not names or unknown:
        raise SpecError(f"of_type() takes names from {sorted(types)}; got {unknown or 'none'}")
    accepted = tuple(t for n in names for t in types[n])

    def check(value: Any) -> str | None:
        number_only = "bool" not in names
        if isinstance(value, bool) and number_only:
            return f"expected {'/'.join(names)}, got bool"
        return (
            None
            if isinstance(value, accepted)
            else f"expected {'/'.join(names)}, got {type(value).__name__}"
        )

    return check


def has_keys(*keys: str) -> Check:
    """The output is a mapping containing every key."""
    if not keys:
        raise SpecError("has_keys() needs at least one key")

    def check(value: Any) -> str | None:
        if not isinstance(value, Mapping):
            return f"expected a mapping, got {type(value).__name__}"
        missing = [k for k in keys if k not in value]
        return f"missing keys: {missing}" if missing else None

    return check


def is_json() -> Check:
    """The output is a string containing valid JSON."""

    def check(value: Any) -> str | None:
        if not isinstance(value, str):
            return f"expected a JSON string, got {type(value).__name__}"
        try:
            json.loads(value)
        except ValueError as exc:
            return f"invalid JSON: {exc}"
        return None

    return check


def max_length(limit: int) -> Check:
    """At most ``limit`` characters (strings) or items (sequences)."""

    def check(value: Any) -> str | None:
        size = len(value) if isinstance(value, str | Sequence) else None
        if size is None:
            return f"{type(value).__name__} has no length"
        return None if size <= limit else f"length {size} exceeds {limit}"

    return check


def min_length(limit: int) -> Check:
    """At least ``limit`` characters (strings) or items (sequences)."""

    def check(value: Any) -> str | None:
        size = len(value) if isinstance(value, str | Sequence) else None
        if size is None:
            return f"{type(value).__name__} has no length"
        return None if size >= limit else f"length {size} is below {limit}"

    return check


def matches(pattern: str, flags: int = 0) -> Check:
    """Some string in the output matches the regular expression."""
    compiled = re.compile(pattern, flags)

    def check(value: Any) -> str | None:
        found = any(compiled.search(text) for text in _strings(value))
        return None if found else f"nothing matches /{pattern}/"

    return check


def not_matches(pattern: str, flags: int = 0) -> Check:
    """No string in the output matches the regular expression (banned phrases, markers, ...)."""
    compiled = re.compile(pattern, flags)

    def check(value: Any) -> str | None:
        for text in _strings(value):
            found = compiled.search(text)
            if found:
                return f"banned pattern /{pattern}/ matched {_mask(found.group(0))}"
        return None

    return check


def all_of(*checks: Check) -> Check:
    """Every check passes; the first failure is reported."""
    if not checks:
        raise SpecError("all_of() needs at least one check")

    def check(value: Any) -> str | None:
        for inner in checks:
            reason = evaluate(inner, value)
            if reason:
                return reason
        return None

    return Tagged(check, sensitive=any(is_sensitive(inner) for inner in checks))


# --------------------------------------------------------------------------- sensitive data


def _luhn_ok(digits: str) -> bool:
    total = 0
    for position, char in enumerate(reversed(digits)):
        digit = int(char)
        if position % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def _is_card(candidate: str) -> bool:
    digits = re.sub(r"[ -]", "", candidate)
    return 13 <= len(digits) <= 19 and digits.isdigit() and _luhn_ok(digits)


def _is_iban(candidate: str) -> bool:
    compact = candidate.replace(" ", "").upper()
    if not 15 <= len(compact) <= 34:
        return False
    rearranged = compact[4:] + compact[:4]
    return int("".join(str(int(char, 36)) for char in rearranged)) % 97 == 1


def _always(_: str) -> bool:
    return True


_PII: dict[str, tuple[re.Pattern[str], Callable[[str], bool]]] = {
    "email": (
        re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}"),
        _always,
    ),
    "phone": (
        re.compile(
            r"(?<![\w+])(?:\+\d{1,3}[ .-]?)?(?:\(\d{3}\)[ .-]?|\d{3}[ .-])\d{3}[ .-]\d{4}(?!\d)"
            r"|(?<!\w)\+\d{1,3}(?:[ .-]?\d{2,4}){3,5}(?!\d)"
        ),
        _always,
    ),
    "ssn": (
        re.compile(r"(?<![\d-])(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}(?![\d-])"),
        _always,
    ),
    "credit_card": (re.compile(r"(?<!\d)\d(?:[ -]?\d){12,18}(?!\d)"), _is_card),
    "iban": (re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,3})?\b"), _is_iban),
}

_SECRETS: dict[str, re.Pattern[str]] = {
    "aws_access_key": re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b"),
    "private_key": re.compile(r"-----BEGIN (?:[A-Z]+ )*PRIVATE KEY-----"),
    "jwt": re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
    "api_key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}"),
    "slack_token": re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}"),
}

PII_KINDS = tuple(_PII)
SECRET_KINDS = tuple(_SECRETS)


def _kinds(requested: Sequence[str] | None, available: Sequence[str], what: str) -> tuple[str, ...]:
    chosen = tuple(requested) if requested else tuple(available)
    unknown = [kind for kind in chosen if kind not in available]
    if unknown:
        raise SpecError(f"unknown {what} kind(s) {unknown}; available: {list(available)}")
    return chosen


def no_pii(kinds: Sequence[str] | None = None) -> Check:
    """No personal data in any string of the output.

    Heuristic detectors for email, phone, ``ssn``, ``credit_card`` (Luhn-validated) and
    ``iban`` (mod-97-validated). It finds the obvious leaks, not every possible one. Failures
    withhold the offending output from reports; only a masked excerpt is shown.
    """
    chosen = _kinds(kinds, PII_KINDS, "PII")

    def check(value: Any) -> str | None:
        found: list[str] = []
        for text in _strings(value):
            for kind in chosen:
                pattern, validate = _PII[kind]
                found.extend(
                    f"{kind} ({_mask(m.group(0))})"
                    for m in pattern.finditer(text)
                    if validate(m.group(0))
                )
        return f"possible PII: {', '.join(found[:3])}" if found else None

    return Tagged(check, sensitive=True)


def no_secrets(kinds: Sequence[str] | None = None) -> Check:
    """No credentials in any string of the output (cloud keys, tokens, private keys, JWTs)."""
    chosen = _kinds(kinds, SECRET_KINDS, "secret")

    def check(value: Any) -> str | None:
        found = [
            f"{kind} ({_mask(match.group(0))})"
            for text in _strings(value)
            for kind in chosen
            for match in _SECRETS[kind].finditer(text)
        ]
        return f"possible secret: {', '.join(found[:3])}" if found else None

    return Tagged(check, sensitive=True)


REGISTRY: dict[str, Callable[..., Check]] = {
    "finite": finite,
    "in_range": in_range,
    "probability": probability,
    "one_of": one_of,
    "of_type": of_type,
    "has_keys": has_keys,
    "is_json": is_json,
    "max_length": max_length,
    "min_length": min_length,
    "matches": matches,
    "not_matches": not_matches,
    "no_pii": no_pii,
    "no_secrets": no_secrets,
}
