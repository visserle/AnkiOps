from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from ankiops.collab.errors import RepositoryCollisionError
from ankiops.collab.publish import _prepare_publish_plan, _publish_remote_head
from ankiops.deck_sources import DeckSource
from tests.support.deck_files import DeckFileHarness


class _ObservedRemote:
    def __init__(
        self: _ObservedRemote, result: subprocess.CompletedProcess[str]
    ) -> None:
        self.result = result
        self.calls: list[tuple[list[str], bool]] = []

    def run(
        self: _ObservedRemote, args: list[str], *, check: bool = True
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append((args, check))
        return self.result


@pytest.mark.parametrize(
    ("stdout", "expected"),
    [
        ("", None),
        (
            "prepared\tHEAD\nprepared\trefs/heads/main\n",
            "prepared",
        ),
    ],
)
def test_publish_remote_is_observed_once(
    tmp_path: Path, stdout: str, expected: str | None
) -> None:
    remote = _ObservedRemote(
        subprocess.CompletedProcess(["git", "ls-remote"], 0, stdout, "")
    )

    assert (
        _publish_remote_head(
            remote,
            DeckSource.collab(tmp_path, "owner/repo"),
            "prepared",
        )
        == expected
    )
    assert remote.calls == [(["ls-remote", "upstream"], False)]


def test_publish_remote_refuses_unrelated_content_after_one_observation(
    tmp_path: Path,
) -> None:
    remote = _ObservedRemote(
        subprocess.CompletedProcess(
            ["git", "ls-remote"],
            0,
            "other\tHEAD\nother\trefs/heads/main\n",
            "",
        )
    )

    with pytest.raises(RepositoryCollisionError, match="unrelated content"):
        _publish_remote_head(
            remote,
            DeckSource.collab(tmp_path, "owner/repo"),
            "prepared",
        )

    assert remote.calls == [(["ls-remote", "upstream"], False)]


def test_publish_remote_reports_unavailable_after_one_observation(
    tmp_path: Path,
) -> None:
    remote = _ObservedRemote(
        subprocess.CompletedProcess(
            ["git", "ls-remote"],
            1,
            "",
            "network unavailable",
        )
    )

    with pytest.raises(ValueError, match="network unavailable"):
        _publish_remote_head(
            remote,
            DeckSource.collab(tmp_path, "owner/repo"),
            "prepared",
        )

    assert remote.calls == [(["ls-remote", "upstream"], False)]


def test_prepare_publish_plan_strips_leech_tags(tmp_path: Path) -> None:
    harness = DeckFileHarness()
    harness.eject_default_note_types(tmp_path / "note_types")
    (tmp_path / "Shared.md").write_text(
        "\n".join(
            [
                "<!-- note_key: p1 -->",
                "<!-- note_type: AnkiOpsQA -->",
                "<!-- tags: leech shared -->",
                "Q: Q1",
                "A: A1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    plan = _prepare_publish_plan(
        tmp_path,
        "Shared",
        DeckSource.collab(tmp_path, "owner/repo"),
    )

    content = plan.files[0].content
    assert "<!-- tags: shared -->" in content
    assert "<!-- tags: leech" not in content
    assert "<!-- note_type: collab/owner/repo/AnkiOpsQA -->" in content
