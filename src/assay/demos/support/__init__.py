"""Demo: a support-ticket router, healthy (``good``) and brittle (``regressed``).

Both models are tiny keyword systems, enough to show the failure modes that matter for real
text models: they score identically on clean data and diverge under case, whitespace, typos,
Unicode obfuscation and prompt injection.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from typing import Any

LABELS = ("billing", "outage", "account", "shipping", "other")

_KEYWORDS: dict[str, tuple[str, ...]] = {
    "billing": (
        "refund",
        "charged",
        "invoice",
        "payment",
        "credit card",
        "subscription",
        "overcharged",
    ),
    "outage": ("outage", "not working", "down", "unavailable", "crashes", "timeout", "error 500"),
    "account": ("password", "log in", "locked", "login", "verification code", "username", "2fa"),
    "shipping": ("delivery", "package", "tracking", "parcel", "courier", "arrived", "shipped"),
}

# Cyrillic look-alikes folded back to Latin, plus invisible characters stripped.
_FOLD = str.maketrans("аесіјорѕхуАЕСІЈОРЅХУ", "aecijopsxyAECIJOPSXY")
_INVISIBLE = dict.fromkeys(map(ord, "​‌‍⁠﻿"))
_WORDS = sorted({w for words in _KEYWORDS.values() for w in words if " " not in w})
_INJECTED = re.compile(r"classify (?:this )?(?:ticket |request )?as (\w+)", re.IGNORECASE)
_INSTRUCTION = re.compile(
    r"\b(?:ignore (?:all )?(?:the )?(?:previous|prior|above)"
    r"|classify (?:this|the)|system:|note to the ai)",
    re.IGNORECASE,
)


def _normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(_INVISIBLE).translate(_FOLD)
    sentences = re.split(r"(?<=[.!?])\s+", text)  # drop sentences that read like instructions
    text = " ".join(s for s in sentences if not _INSTRUCTION.search(s))
    return " ".join(text.lower().split())


def _answer(label: str, hits: int) -> dict[str, Any]:
    return {"label": label, "confidence": round(hits / (hits + 1), 3)}


def good(ticket: str) -> dict[str, Any]:
    """Normalises Unicode, case and whitespace, tolerates typos, ignores embedded instructions."""
    text = _normalise(ticket)
    tokens = re.findall(r"[a-z0-9']+", text)
    fixed = [(difflib.get_close_matches(t, _WORDS, n=1, cutoff=0.8) or [t])[0] for t in tokens]
    haystack = " ".join(fixed)
    scores = {
        label: sum(1 for k in keys if re.search(rf"\b{re.escape(k)}\b", haystack))
        for label, keys in _KEYWORDS.items()
    }
    label, hits = max(scores.items(), key=lambda item: item[1])
    return _answer(label, hits) if hits else _answer("other", 0)


def regressed(ticket: str) -> dict[str, Any]:
    """Exact, case-sensitive substring matching, and it obeys instructions found in the text."""
    injected = _INJECTED.search(ticket)
    if injected and injected.group(1).lower() in LABELS:
        return _answer(injected.group(1).lower(), 3)
    # Whitespace-naive tokeniser: splits on single spaces only, so "login\n" or "refund\tnow"
    # never match, and multi-word keywords break as soon as the spacing differs.
    tokens = {word.strip(".,!?") for word in ticket.split(" ")}
    scores = {
        label: sum(1 for k in keys if (k in tokens if " " not in k else k in ticket))
        for label, keys in _KEYWORDS.items()
    }
    label, hits = max(scores.items(), key=lambda item: item[1])
    return _answer(label, hits) if hits else _answer("other", 0)
