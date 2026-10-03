from __future__ import annotations

import pytest

from assay.demos import assistant


@pytest.fixture(autouse=True)
def _fresh_assistant_state() -> None:
    """The regressed assistant is deliberately stateful; start every test from zero."""
    assistant._calls.clear()


class FakeClock:
    """A controllable clock so latency behaviour is tested without sleeping."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()
