import random
import re

import pytest

from assay import transforms as T
from assay.errors import ContractError, SpecError

TEXT = "My account was charged twice for the premium plan, please refund the money."


def rng(seed=1):
    return random.Random(seed)


@pytest.mark.parametrize(
    "transform",
    [
        T.typo(0.1),
        T.whitespace(),
        T.homoglyph(),
        T.zero_width(),
        T.suffix("a", "b"),
        T.prefix("a", "b"),
    ],
)
def test_transforms_are_deterministic_for_a_seed(transform):
    assert transform(TEXT, rng(5)) == transform(TEXT, rng(5))


def test_different_seeds_give_different_perturbations():
    outputs = {T.typo(0.1)(TEXT, rng(seed)) for seed in range(20)}
    assert len(outputs) > 5


def test_typos_leave_first_letters_and_short_words_alone():
    for seed in range(50):
        out = T.typo(0.3)("It is an old cat and a big dog", rng(seed))
        assert out.split()[:3] == ["It", "is", "an"]  # words under 3 letters untouched
    for seed in range(50):
        words = T.typo(0.5)(TEXT, rng(seed)).split()
        assert all(w[0] == o[0] for w, o in zip(words, TEXT.split(), strict=False))


def test_typo_makes_at_least_one_edit_and_nothing_to_edit_is_a_noop():
    assert all(T.typo(0.001)(TEXT, rng(s)) != TEXT for s in range(10))
    assert T.typo(0.5)("a b c 1 2 3", rng()) == "a b c 1 2 3"


@pytest.mark.parametrize(
    ("mode", "expected"),
    [("upper", "ABC DEF"), ("lower", "abc def"), ("title", "Abc Def"), ("swap", "aBC DeF")],
)
def test_case_modes(mode, expected):
    assert T.case(mode)("Abc dEf", rng()) == expected


def test_case_rejects_unknown_mode():
    with pytest.raises(SpecError):
        T.case("sarcastic")


def test_whitespace_changes_only_whitespace():
    for seed in range(30):
        out = T.whitespace()(TEXT, rng(seed))
        assert out != TEXT
        assert out.split() == TEXT.split()


def test_homoglyph_is_visually_identical_but_not_equal():
    folded = str.maketrans("аесіјорѕхуАЕСІЈОРЅХУ", "aecijopsxyAECIJOPSXY")
    for seed in range(20):
        out = T.homoglyph(0.5)(TEXT, rng(seed))
        assert out != TEXT
        assert out.translate(folded) == TEXT


def test_zero_width_inserts_only_invisible_characters():
    out = T.zero_width(0.3)(TEXT, rng())
    assert out != TEXT and out.replace("​", "") == TEXT


def test_replace_is_whole_word_and_case_preserving():
    swap = T.replace({"bad": "poor", "very bad": "terrible"})
    assert swap("The bad Bad BAD badge, very bad", rng()) == "The poor Poor POOR badge, terrible"


def test_replace_whole_word_off_and_case_sensitive():
    assert T.replace({"cat": "dog"}, whole_word=False)("concatenate", rng()) == "condogenate"
    assert T.replace({"Cat": "Dog"}, ignore_case=False)("Cat cat", rng()) == "Dog cat"


def test_replace_needs_a_mapping():
    with pytest.raises(SpecError):
        T.replace({})


def test_swap_is_simultaneous_so_it_cannot_undo_itself():
    assert (
        T.swap({"he": "she", "she": "he"})("He told she that he left", rng())
        == "She told he that she left"
    )
    assert T.swap({"he": "she"}, both_ways=True)("he and she", rng()) == "she and he"


def test_symmetric_detects_conflicts():
    assert T.symmetric({"a": "b"}) == {"a": "b", "b": "a"}
    with pytest.raises(SpecError, match="conflicting"):
        T.symmetric({"a": "b", "c": "b"})


def test_gender_swap_handles_common_pronouns_and_titles():
    out = T.gender_swap()("He said Mr Lee told his wife that she is a woman.", rng())
    assert out == "She said Ms Lee told her husband that he is a man."


def test_suffix_and_prefix_pick_from_the_given_texts():
    seen = {T.suffix("X", "Y", "Z")("hi", rng(s)) for s in range(30)}
    assert seen == {"hi X", "hi Y", "hi Z"}
    assert T.prefix("X", sep=": ")("hi", rng()) == "X: hi"
    for factory in (T.suffix, T.prefix):
        with pytest.raises(SpecError):
            factory()


def test_text_transforms_can_target_a_field_of_a_dict():
    row = {"message": "hello there", "id": 7}
    out = T.case("upper", field="message")(row, rng())
    assert out == {"message": "HELLO THERE", "id": 7} and row["message"] == "hello there"


def test_text_transform_on_wrong_input_explains_the_fix():
    with pytest.raises(ContractError, match="pass field"):
        T.typo()({"a": 1}, rng())
    with pytest.raises(ContractError, match="str field 'message'"):
        T.case("upper", field="message")({"a": 1}, rng())


def test_tabular_transforms_do_not_mutate_their_input():
    row = {"income": 100.0, "gender": "F", "nested": {"a": [1]}}
    snapshot = {"income": 100.0, "gender": "F", "nested": {"a": [1]}}
    for t in (T.shift_field("income", 5), T.scale_field("income", 2), T.set_field("gender", "M"),
              T.swap_field("gender", {"F": "M"}), T.jitter_field("income"), T.drop_field("income")):  # fmt: skip
        out = t(row, rng())
        out["nested"]["a"].append(2)
        assert row == snapshot


def test_tabular_transform_semantics():
    row = {"income": 100.0, "gender": "F"}
    assert T.shift_field("income", -10)(row, rng())["income"] == 90.0
    assert T.scale_field("income", 1.5)(row, rng())["income"] == 150.0
    assert T.set_field("gender", "X")(row, rng())["gender"] == "X"
    assert {
        T.set_field("gender", choices=["A", "B"])(row, rng(s))["gender"] for s in range(20)
    } == {"A", "B"}
    assert T.swap_field("gender", {"F": "M"})(row, rng())["gender"] == "M"
    assert T.swap_field("gender", {"M": "F"})(row, rng()) == row  # unmatched value: no-op
    assert "income" not in T.drop_field("income")(row, rng())
    assert T.shift_field("missing", 1)(row, rng()) == row


def test_jitter_stays_within_its_band():
    for seed in range(100):
        value = T.jitter_field("x", 0.02)({"x": 1000.0}, rng(seed))["x"]
        assert 980.0 <= value <= 1020.0


def test_numeric_transforms_reject_non_numbers():
    with pytest.raises(ContractError, match="numeric"):
        T.shift_field("x", 1)({"x": "abc"}, rng())
    with pytest.raises(ContractError, match="dict input"):
        T.shift_field("x", 1)("not a dict", rng())


def test_set_field_needs_a_value():
    with pytest.raises(SpecError):
        T.set_field("x")


def test_chain_applies_in_order_with_one_random_stream():
    both = T.chain(T.case("upper"), T.suffix("!"))
    assert both("hi", rng()) == "HI !"
    assert both.name == "case(upper)+suffix"
    with pytest.raises(SpecError):
        T.chain()


@pytest.mark.parametrize("factory", [T.typo, T.homoglyph, T.zero_width])
def test_rates_are_validated(factory):
    for bad in (0, -0.1, 1.5):
        with pytest.raises(SpecError):
            factory(bad)
    with pytest.raises(SpecError):
        T.jitter_field("x", 1.0)


def test_registry_covers_every_public_factory():
    assert {"typo", "case", "whitespace", "swap", "gender_swap", "swap_field", "chain"} - set(
        T.REGISTRY
    ) == {"chain"}
    assert re.fullmatch(r"typo\(rate=0.05\)", T.typo().name)


def test_a_buggy_custom_transform_is_reported_as_a_contract_error():
    broken = T.Transform("my-transform", lambda value, rng: 1 / 0)
    with pytest.raises(
        ContractError, match=r"transform 'my-transform' failed on str input: ZeroDivisionError"
    ):
        broken("text", rng())
    assert repr(broken) == "Transform(my-transform)"


def test_unicode_transforms_leave_text_without_candidates_untouched():
    assert T.homoglyph()("12345 !!!", rng()) == "12345 !!!"
    assert T.zero_width()("12345 !!!", rng()) == "12345 !!!"
