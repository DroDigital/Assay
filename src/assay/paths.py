"""Dotted-path access into JSON-like model outputs (``choices.0.message.content``)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .errors import ExtractionError


def get_path(obj: Any, path: str | None) -> Any:
    """Return ``obj[part0][part1]...`` for a dotted ``path``; ``None``/``""`` returns ``obj``.

    Mapping keys are looked up as strings, sequence parts must be (possibly negative) integers.
    A missing step raises :class:`ExtractionError`, which the runner scores as a violation:
    a model whose output schema silently changed has broken its contract.
    """
    if not path:
        return obj
    current = obj
    walked: list[str] = []
    for part in path.split("."):
        if isinstance(current, Mapping):
            if part not in current:
                raise ExtractionError(_missing(path, walked, part, current))
            current = current[part]
        elif isinstance(current, Sequence) and not isinstance(current, str | bytes):
            try:
                current = current[int(part)]
            except (ValueError, IndexError):
                raise ExtractionError(_missing(path, walked, part, current)) from None
        else:
            raise ExtractionError(
                f"cannot read '{part}' of {type(current).__name__} while resolving '{path}'"
            )
        walked.append(part)
    return current


def _missing(path: str, walked: list[str], part: str, container: Any) -> str:
    where = ".".join(walked) or "<root>"
    if isinstance(container, Mapping):
        have = ", ".join(sorted(map(str, container))[:8]) or "<empty>"
        return f"path '{path}': no key '{part}' at {where} (keys: {have})"
    return f"path '{path}': no index '{part}' at {where} (length {len(container)})"
