"""Toolbar command runner for the AnkiOps add-on."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

from aqt.qt import (
    QDialog,
    QDialogButtonBox,
    QFontDatabase,
    QPlainTextEdit,
    QVBoxLayout,
)
from aqt.utils import showCritical, tooltip


class AnkiOpsCommandRunner:
    def __init__(self, *, mw, addon_module_name: str) -> None:
        self._mw = mw
        self._addon_module_name = addon_module_name

    def run(self, command: str) -> None:
        binary = self._resolve_ankiops_binary()
        if binary is None:
            showCritical(
                "Could not find the AnkiOps CLI. Install it with "
                "`pipx install ankiops`, or set `ankiops_path` in "
                "Tools → Add-ons → AnkiOps → Config."
            )
            return

        cwd = self._collection_dir()
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
                self._show_output(f"ankiops {command} — done", output)
                if command == "fa":
                    self._mw.reset()
            else:
                self._show_output(
                    f"ankiops {command} — failed (exit {result.returncode})", output
                )

        self._mw.taskman.run_in_background(task, on_done)

    def _config(self) -> dict[str, Any]:
        return self._mw.addonManager.getConfig(self._addon_module_name) or {}

    def _resolve_ankiops_binary(self) -> str | None:
        configured = (self._config().get("ankiops_path") or "").strip()
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

    def _collection_dir(self) -> Path | None:
        configured = (self._config().get("collection_dir") or "").strip()
        if not configured:
            return None
        path = Path(configured).expanduser()
        return path if path.is_dir() else None

    def _show_output(self, title: str, body: str) -> None:
        dialog = QDialog(self._mw)
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


def add_toolbar_links(links: list[str], toolbar, runner: AnkiOpsCommandRunner) -> None:
    links.insert(
        -2,
        toolbar.create_link(
            cmd="ankiops_anki_to_files",
            label="af",
            func=lambda: runner.run("af"),
            tip="ankiops anki-to-files",
            id="ankiops-anki-to-files",
        ),
    )
    links.insert(
        -2,
        toolbar.create_link(
            cmd="ankiops_files_to_anki",
            label="fa",
            func=lambda: runner.run("fa"),
            tip="ankiops files-to-anki",
            id="ankiops-files-to-anki",
        ),
    )
