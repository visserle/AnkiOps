import io
import logging
import sys

from ankiops.log import clickable_path, close_root_logging, configure_logging


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


def test_configure_logging_quiets_sdk_logs_but_keeps_ankiops_debug(monkeypatch):
    stream = io.StringIO()
    quiet_logger_names = ("openai", "httpx", "httpcore", "urllib3.connectionpool")
    original_levels = {
        name: logging.getLogger(name).level for name in quiet_logger_names
    }
    monkeypatch.setattr(sys, "stdout", stream)

    try:
        configure_logging(stream_level=logging.DEBUG)

        logging.getLogger("ankiops.llm.runner").debug("runner debug")
        logging.getLogger("openai._base_client").debug("sdk debug")

        rendered = stream.getvalue()
        assert "runner debug" in rendered
        assert "sdk debug" not in rendered
        assert logging.getLogger("openai").level == logging.WARNING
        assert logging.getLogger("httpx").level == logging.WARNING
        assert logging.getLogger("httpcore").level == logging.WARNING
        assert logging.getLogger("urllib3.connectionpool").level == logging.WARNING
    finally:
        close_root_logging()
        for name, level in original_levels.items():
            logging.getLogger(name).setLevel(level)


def test_configure_logging_compact_uses_rich_level_column(monkeypatch):
    stream = io.StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    try:
        configure_logging(stream_level=logging.INFO)

        logging.info("compact info message")
        logging.warning("compact warning message")

        rendered = stream.getvalue()
        assert "compact info message" in rendered
        assert "compact warning message" in rendered
        assert "INFO" in rendered
        assert "WARNING" in rendered
        assert "WARNING:" not in rendered
    finally:
        close_root_logging()
