import math

import pytest

from assay.compare import approx, casefold, exact, make_comparator
from assay.errors import ExtractionError, SpecError
from assay.paths import get_path


def test_get_path_walks_mappings_and_lists():
    data = {"choices": [{"message": {"content": "hi"}}], "n": 1}
    assert get_path(data, "choices.0.message.content") == "hi"
    assert get_path(data, "choices.-1.message") == {"content": "hi"}
    assert get_path(data, None) is data
    assert get_path(data, "") is data


@pytest.mark.parametrize(
    ("path", "fragment"),
    [
        ("missing", "no key 'missing'"),
        ("choices.5", "no index '5'"),
        ("choices.x", "no index 'x'"),
        ("n.deeper", "cannot read 'deeper' of int"),
    ],
)
def test_get_path_errors_say_where_it_went_wrong(path, fragment):
    with pytest.raises(ExtractionError, match=fragment):
        get_path({"choices": [1], "n": 1}, path)


def test_missing_key_message_lists_available_keys():
    with pytest.raises(ExtractionError, match=r"keys: a, b"):
        get_path({"a": 1, "b": 2}, "c")


def test_exact_and_casefold():
    assert exact(1, 1.0) and not exact("a", "b")
    assert casefold("  Hello   WORLD ", "hello world")
    assert not casefold("a", "b")
    assert casefold(3, 3)  # non-strings fall back to equality


def test_approx_numbers_and_structures():
    close = approx(abs_tol=0.01)
    assert close(1.0, 1.005) and not close(1.0, 1.02)
    assert close({"s": 0.5, "l": "x"}, {"s": 0.504, "l": "x"})
    assert not close({"s": 0.5}, {"s": 0.5, "extra": 1})
    assert close([1.0, 2.0], (1.001, 2.001))
    assert not close([1.0], [1.0, 2.0])
    assert approx(rel_tol=0.1)(100, 105)


def test_approx_never_treats_bools_as_numbers_and_rejects_nan():
    close = approx(abs_tol=2)
    assert not close(True, 2)
    assert close(True, True)
    assert not close(math.nan, math.nan)


def test_approx_rejects_negative_tolerances():
    with pytest.raises(SpecError):
        approx(abs_tol=-1)


def test_make_comparator():
    assert make_comparator() is exact
    assert make_comparator("casefold") is casefold
    assert make_comparator(abs_tol=0.1)(1.0, 1.05)
    assert make_comparator("approx", rel_tol=0.5)(10, 12)
    with pytest.raises(SpecError, match="only apply"):
        make_comparator("exact", abs_tol=0.1)
    with pytest.raises(SpecError, match="unknown comparator"):
        make_comparator("fuzzy")
