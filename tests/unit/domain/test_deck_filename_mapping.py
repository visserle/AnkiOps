import pytest

from ankiops.collection import (
    deck_name_in_scope,
    deck_name_to_file_stem,
    file_stem_to_deck_name,
)


def test_subdeck_separator_uses_double_underscore():
    assert deck_name_to_file_stem("A::B") == "A__B"
    assert file_stem_to_deck_name("A__B") == "A::B"


def test_literal_double_underscore_is_roundtrip_safe():
    deck_name = "A__B"
    stem = deck_name_to_file_stem(deck_name)

    assert stem == "A%5F%5FB"
    assert file_stem_to_deck_name(stem) == deck_name


def test_mixed_subdeck_and_literal_underscores_is_roundtrip_safe():
    deck_name = "A::B__C"
    stem = deck_name_to_file_stem(deck_name)

    assert stem == "A__B%5F%5FC"
    assert file_stem_to_deck_name(stem) == deck_name


def test_percent_literal_is_roundtrip_safe():
    deck_name = "A%20B"
    stem = deck_name_to_file_stem(deck_name)

    assert stem == "A%2520B"
    assert file_stem_to_deck_name(stem) == deck_name


def test_path_separators_are_roundtrip_safe():
    deck_name = "A/B\\C"
    stem = deck_name_to_file_stem(deck_name)

    assert stem == "A%2FB%5CC"
    assert file_stem_to_deck_name(stem) == deck_name


@pytest.mark.parametrize("deck_name", ["A_::B", "A::_B", "A_::_B"])
def test_underscores_adjacent_to_subdeck_separator_are_rejected(deck_name):
    with pytest.raises(ValueError, match="Ambiguous deck name"):
        deck_name_to_file_stem(deck_name)


def test_deck_name_scope_includes_subdecks_by_default():
    assert deck_name_in_scope("Parent", deck="Parent", no_subdecks=False)
    assert deck_name_in_scope("Parent::Child", deck="Parent", no_subdecks=False)
    assert not deck_name_in_scope("Other", deck="Parent", no_subdecks=False)


def test_deck_name_scope_can_exclude_subdecks():
    assert deck_name_in_scope("Parent", deck="Parent", no_subdecks=True)
    assert not deck_name_in_scope("Parent::Child", deck="Parent", no_subdecks=True)
