from ankiops.log import clickable_path


def test_clickable_path_encodes_file_uri(tmp_path, monkeypatch):
    file_path = tmp_path / "Deck A #1.md"
    file_path.write_text("x", encoding="utf-8")
    monkeypatch.delenv("NO_COLOR", raising=False)

    rendered = clickable_path(file_path)
    file_uri = file_path.resolve().as_uri()

    assert file_uri in rendered
    assert "%20" in file_uri
    assert "%23" in file_uri
    assert f"FILE {file_path.resolve()}" in rendered


def test_clickable_path_returns_plain_text_with_no_color(tmp_path, monkeypatch):
    file_path = tmp_path / "Deck A.md"
    file_path.write_text("x", encoding="utf-8")
    monkeypatch.setenv("NO_COLOR", "1")

    rendered = clickable_path(file_path)

    assert rendered == f"FILE {file_path.resolve()}"


def test_clickable_path_handles_missing_path(tmp_path, monkeypatch):
    file_path = tmp_path / "Missing Deck.md"
    monkeypatch.delenv("NO_COLOR", raising=False)

    rendered = clickable_path(file_path)

    assert file_path.resolve().as_uri() in rendered
    assert f"FILE {file_path.resolve()}" in rendered
