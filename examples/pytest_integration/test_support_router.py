"""Using assay inside an ordinary pytest suite: contracts become regular test functions.

    pytest examples/pytest_integration

``assert_ok()`` raises ``AssertionError`` carrying the full report, so a failing contract shows
its counterexamples right in the pytest output.
"""

import pytest

from assay import Case, Golden, Invariance, Property, Suite, checks
from assay import transforms as T
from assay.demos.support import good, regressed

TICKETS = [
    ("billing", "I was charged twice for my subscription and need a refund."),
    ("outage", "The whole service is down for our team since this morning."),
    ("account", "I forgot my password and the reset email never arrives."),
    ("shipping", "My package has not arrived and the tracking page has not moved."),
    ("other", "Great product, just wanted to say thank you to the team."),
]
LABELS = ("billing", "outage", "account", "shipping", "other")


def build_suite(model):
    cases = [
        Case(input=text, id=f"ticket-{i}", expected=label)
        for i, (label, text) in enumerate(TICKETS)
    ]
    return Suite("support-router", model, cases, seed=3).add(
        Golden("accuracy", output="label", tolerance=0.3),
        Property("label-in-closed-set", check=checks.one_of(*LABELS), output="label"),
        Invariance("shouting", transform=T.case("upper"), output="label"),
        Invariance("typos", transform=T.typo(0.06), variants=5, output="label", tolerance=0.25),
    )


def test_healthy_router_honours_its_contracts():
    build_suite(good).run().assert_ok()


def test_regressed_router_is_caught():
    with pytest.raises(AssertionError, match="shouting"):
        build_suite(regressed).run().assert_ok()
