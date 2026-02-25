from ankiops.config import deck_name_to_file_stem, file_stem_to_deck_name


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
