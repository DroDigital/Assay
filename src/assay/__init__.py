"""assay: statistically rigorous behavioural contracts for AI and ML systems.

Quick start::

    from assay import Suite, Invariance, transforms as T

    suite = Suite("triage", model=classify, cases=tickets)
    suite.add(Invariance("typo-robust", transform=T.typo(0.05), output="label", tolerance=0.05))
    suite.run().assert_ok()
"""

from . import checks, transforms
from ._version import __version__
from .compare import approx, casefold, exact
from .contracts import (
    Agreement,
    Case,
    Consistency,
    Contract,
    Golden,
    Invariance,
    Latency,
    Monotonicity,
    Property,
    Sensitivity,
)
from .errors import AssayError, ContractError, ModelError, SpecError
from .model import Model, command_model, http_model
from .results import ContractResult, SuiteResult
from .runner import Suite, run_suite
from .stats import Verdict, decide, samples_to_decide, wilson_interval

__all__ = [
    "Agreement",
    "AssayError",
    "Case",
    "Consistency",
    "Contract",
    "ContractError",
    "ContractResult",
    "Golden",
    "Invariance",
    "Latency",
    "Model",
    "ModelError",
    "Monotonicity",
    "Property",
    "Sensitivity",
    "SpecError",
    "Suite",
    "SuiteResult",
    "Verdict",
    "__version__",
    "approx",
    "casefold",
    "checks",
    "command_model",
    "decide",
    "exact",
    "http_model",
    "run_suite",
    "samples_to_decide",
    "transforms",
    "wilson_interval",
]
