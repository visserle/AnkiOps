"""AnkiOps Anki add-on: Import / Export buttons in the top toolbar.

Adds two links to Anki's main toolbar that shell out to the `ankiops`
CLI (installed separately via `pipx install ankiops`).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

from aqt import gui_hooks, mw
from aqt.qt import (
    QDialog,
    QDialogButtonBox,
    QFontDatabase,
    QPlainTextEdit,
    QVBoxLayout,
)
from aqt.toolbar import Toolbar
from aqt.utils import showCritical, tooltip

ADDON_NAME = "AnkiOps"


def _config() -> dict[str, Any]:
    return mw.addonManager.getConfig(__name__) or {}


def _resolve_ankiops_binary() -> str | None:
    configured = (_config().get("ankiops_path") or "").strip()
    if configured:
        path = Path(configured).expanduser()
        if path.exists():
            return str(path)

    candidates = [
        shutil.which("ankiops"),
        str(Path.home() / ".local" / "bin" / "ankiops"),
        "/opt/homebrew/bin/ankiops",
        "/usr/local/bin/ankiops",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return None


def _collection_dir() -> Path | None:
    configured = (_config().get("collection_dir") or "").strip()
    if not configured:
        return None
    path = Path(configured).expanduser()
    return path if path.is_dir() else None


def _run_ankiops(command: str) -> None:
    binary = _resolve_ankiops_binary()
    if binary is None:
        showCritical(
            "Could not find the AnkiOps CLI. Install it with "
            "`pipx install ankiops`, or set `ankiops_path` in "
            "Tools → Add-ons → AnkiOps → Config."
        )
        return

    cwd = _collection_dir()
    if cwd is None:
        showCritical(
            "Set `collection_dir` in Tools → Add-ons → AnkiOps "
            "→ Config to point to your AnkiOps collection."
        )
        return

    tooltip(f"Running ankiops {command}…")

    def task() -> subprocess.CompletedProcess:
        return subprocess.run(
            [binary, command],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=False,
        )

    def on_done(future) -> None:
        try:
            result = future.result()
        except Exception as error:
            showCritical(f"Failed to run ankiops: {error}")
            return

        output = ((result.stdout or "") + (result.stderr or "")).strip()
        if result.returncode == 0:
            _show_output(f"ankiops {command} — done", output)
            if command == "ma":
                mw.reset()
        else:
            _show_output(
                f"ankiops {command} — failed (exit {result.returncode})", output
            )

    mw.taskman.run_in_background(task, on_done)


def _show_output(title: str, body: str) -> None:
    dialog = QDialog(mw)
    dialog.setWindowTitle(title)
    dialog.resize(600, 400)

    layout = QVBoxLayout(dialog)

    view = QPlainTextEdit(dialog)
    view.setReadOnly(True)
    view.setPlainText(body or "(no output)")
    view.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
    view.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
    layout.addWidget(view)

    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok, dialog)
    buttons.accepted.connect(dialog.accept)
    layout.addWidget(buttons)

    dialog.exec()


def _on_toolbar_init(links: list[str], toolbar: Toolbar) -> None:
    links.insert(
        -2,
        toolbar.create_link(
            cmd="ankiops_anki_to_markdown",
            label="am",
            func=lambda: _run_ankiops("am"),
            tip="ankiops anki-to-markdown",
            id="ankiops-anki-to-markdown",
        ),
    )
    links.insert(
        -2,
        toolbar.create_link(
            cmd="ankiops_markdown_to_anki",
            label="ma",
            func=lambda: _run_ankiops("ma"),
            tip="ankiops markdown-to-anki",
            id="ankiops-markdown-to-anki",
        ),
    )


gui_hooks.top_toolbar_did_init_links.append(_on_toolbar_init)
