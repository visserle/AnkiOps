"""Regression tests for the lightweight CLI startup path."""

import argparse
import subprocess
import sys
from typing import Any


def test_cli_import_does_not_load_openai() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import ankiops.cli, sys; assert 'openai' not in sys.modules",
        ],
        check=True,
    )


def test_llm_handler_loads_and_delegates(monkeypatch: Any) -> None:
    from ankiops.llm import commands
    from ankiops.llm_parser import run_llm

    received: list[argparse.Namespace] = []
    monkeypatch.setattr(commands, "run_llm", received.append)
    args = argparse.Namespace(task_name="example")
    run_llm(args)
    assert received == [args]


def test_llm_help_keeps_model_registry_path() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "ankiops.cli", "llm", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "llm/_models.yaml" in result.stdout
