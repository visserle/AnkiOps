from __future__ import annotations

import subprocess

import pytest

from ankiops.collab.errors import RepositoryCollisionError
from ankiops.collab.publish import _publish_remote_head
from ankiops.deck_sources import DeckSource


class _ObservedRemote:
    def __init__(self, result: subprocess.CompletedProcess[str]):
        self.result = result
        self.calls = []

    def run(self, args, *, check=True):
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
def test_publish_remote_is_observed_once(tmp_path, stdout, expected):
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


def test_publish_remote_refuses_unrelated_content_after_one_observation(tmp_path):
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


def test_publish_remote_reports_unavailable_after_one_observation(tmp_path):
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
