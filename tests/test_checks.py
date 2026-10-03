import math

import pytest

from assay import checks as C
from assay.errors import ContractError, SpecError

# Credentials are assembled at runtime so no secret-shaped literal sits in the repository.
AWS_KEY = "AK" + "IA" + "B" * 16
GITHUB_TOKEN = "gh" + "p_" + "a1" * 18
OPENAI_STYLE_KEY = "sk-" + "Z9" * 15
PEM_HEADER = "-----BEGIN " + "RSA PRIVATE KEY-----"
JWT = ".".join(["ey" + "J" + "a" * 12, "b" * 14, "c" * 14])


def test_evaluate_normalises_results():
    assert C.evaluate(lambda v: None, 1) is None
    assert C.evaluate(lambda v: True, 1) is None
    assert C.evaluate(lambda v: "bad", 1) == "bad"
    assert "failed" in C.evaluate(lambda v: False, 1)
    with pytest.raises(ContractError, match="must return"):
        C.evaluate(lambda v: 42, 1)
    with pytest.raises(ContractError, match="raised ZeroDivisionError"):
        C.evaluate(lambda v: 1 / 0, 1)


@pytest.mark.parametrize(
    ("value", "ok"),
    [
        (0.5, True),
        (0, True),
        (1, True),
        (1.1, False),
        (-0.01, False),
        (math.nan, False),
        (math.inf, False),
        (True, False),
        ("0.5", False),
    ],
)
def test_probability(value, ok):
    assert (C.evaluate(C.probability(), value) is None) is ok


def test_in_range_bounds_are_optional_but_one_is_required():
    assert C.evaluate(C.in_range(lo=0), 5) is None
    assert "below" in C.evaluate(C.in_range(lo=0), -1)
    assert "above" in C.evaluate(C.in_range(hi=10), 11)
    with pytest.raises(SpecError):
        C.in_range()


def test_one_of_of_type_has_keys():
    assert C.evaluate(C.one_of("a", "b"), "a") is None
    assert "not one of" in C.evaluate(C.one_of("a", "b"), "c")
    assert C.evaluate(C.of_type("number"), 3) is None
    assert C.evaluate(C.of_type("number"), True) is not None  # bool is not a number here
    assert C.evaluate(C.of_type("bool"), True) is None
    assert C.evaluate(C.of_type("str", "list"), []) is None
    assert C.evaluate(C.has_keys("a", "b"), {"a": 1, "b": 2}) is None
    assert "missing keys: ['b']" in C.evaluate(C.has_keys("a", "b"), {"a": 1})
    assert "mapping" in C.evaluate(C.has_keys("a"), [1])
    for bad in (
        lambda: C.one_of(),
        lambda: C.of_type("nope"),
        lambda: C.of_type(),
        lambda: C.has_keys(),
    ):
        with pytest.raises(SpecError):
            bad()


def test_is_json_and_lengths():
    assert C.evaluate(C.is_json(), '{"a": 1}') is None
    assert "invalid JSON" in C.evaluate(C.is_json(), "{nope")
    assert "expected a JSON string" in C.evaluate(C.is_json(), {"a": 1})
    assert C.evaluate(C.max_length(3), "abc") is None
    assert "exceeds" in C.evaluate(C.max_length(3), "abcd")
    assert C.evaluate(C.min_length(2), [1, 2]) is None
    assert "below" in C.evaluate(C.min_length(2), "a")
    assert "no length" in C.evaluate(C.max_length(3), 12345)


def test_matches_and_not_matches_scan_nested_strings():
    assert C.evaluate(C.matches(r"^ok"), {"a": ["ok then"]}) is None
    assert "nothing matches" in C.evaluate(C.matches(r"^ok"), "nope")
    assert C.evaluate(C.not_matches(r"forbidden"), {"a": ["fine"], "b": 3}) is None
    reason = C.evaluate(C.not_matches(r"forbidden"), {"a": ["so forbidden"]})
    assert "banned pattern" in reason and "forbidden" not in reason.split("matched")[1]


def test_all_of_reports_first_failure():
    check = C.all_of(C.of_type("number"), C.in_range(0, 1))
    assert C.evaluate(check, 0.5) is None
    assert "above" in C.evaluate(check, 2)
    assert "expected number" in C.evaluate(check, "x")


@pytest.mark.parametrize(
    "leak",
    [
        "write to jane.doe@example.com today",
        "call +1 415-555-0132 now",
        "call (415) 555-0132 now",
        "or +44 20 7946 0958",
        "ssn 123-45-6789",
        "card 4111 1111 1111 1111",
        "card 4111-1111-1111-1111",
        "iban GB82 WEST 1234 5698 7654 32",
        "iban DE89370400440532013000",
    ],
)
def test_pii_detectors_find_obvious_leaks(leak):
    assert "possible PII" in C.evaluate(C.no_pii(), leak)


@pytest.mark.parametrize(
    "clean",
    [
        "order 20240315 shipped",
        "card 1234 5678 9012 3456",  # fails the Luhn checksum
        "iban GB00 WEST 1234 5698 7654 32",  # fails mod-97
        "ssn 000-12-3456 and 666-12-3456 and 900-12-3456",  # invalid area numbers
        "version 3.11.4 released 2024-03-15",
        "call 12345",
        "meet at user at noon",
    ],
)
def test_pii_detectors_do_not_cry_wolf(clean):
    assert C.evaluate(C.no_pii(), clean) is None


def test_pii_reason_never_echoes_the_full_value():
    reason = C.evaluate(C.no_pii(["email"]), "mail me: jane.doe@example.com")
    assert "jane.doe@example.com" not in reason and "email" in reason


def test_pii_scans_every_string_in_structured_outputs_and_respects_kinds():
    leak = {"a": [{"note": "reach me at x@example.com"}]}
    assert C.evaluate(C.no_pii(["email"]), leak) is not None
    assert C.evaluate(C.no_pii(["phone"]), leak) is None
    with pytest.raises(SpecError, match="unknown PII kind"):
        C.no_pii(["dna"])


@pytest.mark.parametrize(
    "secret", [AWS_KEY, GITHUB_TOKEN, OPENAI_STYLE_KEY, PEM_HEADER, JWT, "xox" + "b-" + "1" * 12]
)
def test_secret_detectors(secret):
    reason = C.evaluate(C.no_secrets(), f"debug dump: {secret} end")
    assert reason is not None and secret not in reason


def test_secrets_no_false_positives_and_kinds():
    assert (
        C.evaluate(C.no_secrets(), "The sky is blue; sk-learn is a library; AKIA alone is nothing")
        is None
    )
    assert C.evaluate(C.no_secrets(["aws_access_key"]), GITHUB_TOKEN) is None
    with pytest.raises(SpecError):
        C.no_secrets(["password"])


def test_registry_entries_are_constructible_with_no_required_state():
    assert callable(C.REGISTRY["no_pii"]()) and callable(C.REGISTRY["probability"]())


def test_length_checks_reject_values_without_a_length():
    assert "no length" in C.evaluate(C.min_length(1), 42)
    with pytest.raises(SpecError):
        C.all_of()
