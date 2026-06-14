from ankiops.markdown import write_deck_file


def test_write_deck_file_adds_final_newline_to_non_empty_content(tmp_path):
    path = tmp_path / "Deck.md"

    write_deck_file(path, "Q: Question")

    assert path.read_text(encoding="utf-8") == "Q: Question\n"


def test_write_deck_file_keeps_existing_final_newline(tmp_path):
    path = tmp_path / "Deck.md"

    write_deck_file(path, "Q: Question\n")

    assert path.read_text(encoding="utf-8") == "Q: Question\n"


def test_write_deck_file_keeps_empty_content_empty(tmp_path):
    path = tmp_path / "Deck.md"

    write_deck_file(path, "")

    assert path.read_text(encoding="utf-8") == ""
