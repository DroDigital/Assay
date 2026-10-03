"""Exception hierarchy.

The split matters because each class has a different *policy* in the runner:

* :class:`ModelError` - the system under test failed. Counted as a violation by default.
* :class:`ExtractionError` - the output did not have the expected shape. Always a violation.
* :class:`ContractError` - a user-supplied callable (transform, check, comparator) is buggy.
  This is *our* user's bug, not the model's, so it aborts the run instead of being scored.
* :class:`SpecError` - a spec file or constructor argument is invalid.
"""

from __future__ import annotations


class AssayError(Exception):
    """Base class for every error raised deliberately by assay."""


class SpecError(AssayError):
    """A spec file or contract definition is invalid."""


class ModelError(AssayError):
    """The model under test raised, timed out, or could not be reached."""


class ExtractionError(AssayError):
    """The model output does not contain the requested field."""


class ContractError(AssayError):
    """A user-supplied transform, check or comparator raised or misbehaved."""
