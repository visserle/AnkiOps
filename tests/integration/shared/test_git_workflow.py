from __future__ import annotations

import subprocess
from types import SimpleNamespace

from ankiops.deck_sources import DeckSource
from ankiops.git import CollectionGit
from ankiops.shared import run_submit, run_update
from tests.support.deck_files import DeckFileHarness


def _git(cwd, *args):
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=True,
    )


def _configure_git(cwd, name):
    _git(cwd, "config", "user.name", name)
    _git(cwd, "config", "user.email", f"{name.lower()}@example.invalid")


def _replace(path, old, new):
    content = path.read_text(encoding="utf-8").replace(old, new)
    path.write_text(content, encoding="utf-8")


def test_two_rounds_of_concurrent_shared_collaboration(tmp_path, monkeypatch):
    collection = tmp_path / "collection"
    shared_root = collection / "shared" / "owner" / "repo"
    DeckFileHarness().eject_default_note_types(shared_root / "note_types")
    (shared_root / "Deck.md").write_text(
        "<!-- note_key: deck-key -->\n"
        "<!-- note_type: shared/owner/repo/AnkiOpsQA -->\n"
        "Q: root\nA: deck\n",
        encoding="utf-8",
    )
    (shared_root / "Child.md").write_text(
        "<!-- note_key: child-key -->\n"
        "<!-- note_type: shared/owner/repo/AnkiOpsQA -->\n"
        "Q: child\nA: deck\n",
        encoding="utf-8",
    )
    private = collection / "Private.md"
    private.write_text("private baseline\n", encoding="utf-8")
    _git(collection, "init", "-b", "main")
    _configure_git(collection, "Collector")
    _git(collection, "add", "-A")
    _git(collection, "commit", "-m", "Initialize collection")

    source = DeckSource.shared(collection, "owner", "repo")
    repo = CollectionGit(collection)
    initial_split = repo.split_subtree(source)
    repo.rejoin_subtree(
        source,
        initial_split,
        "AnkiOps: initialize subtree history for shared/owner/repo",
    )
    remote = tmp_path / "remote.git"
    subprocess.run(
        ["git", "init", "--bare", "--initial-branch=main", remote],
        text=True,
        capture_output=True,
        check=True,
    )
    repo.push_ref(str(remote), initial_split, "refs/heads/main")

    collaborator = tmp_path / "collaborator"
    _git(tmp_path, "clone", remote, collaborator)
    _configure_git(collaborator, "Collaborator")
    monkeypatch.setattr(
        "ankiops.deck_sources.DeckSource.github_url",
        property(lambda _source: str(remote)),
    )
    monkeypatch.setattr(
        "ankiops.shared.commands.require_collection_dir",
        lambda: collection,
    )
    submitted = []

    def push_submission(repo, _source, branch, *, title):
        sha = repo.run(["rev-parse", branch]).stdout.strip()
        repo.push_ref(str(remote), branch, f"refs/heads/{branch}")
        submitted.append((branch, sha, title))
        return True

    monkeypatch.setattr(
        "ankiops.shared.commands.open_pr_if_possible",
        push_submission,
    )

    for round_number in (1, 2):
        old_child = "A: deck" if round_number == 1 else "A: remote round 1"
        _replace(
            collaborator / "Child.md",
            old_child,
            f"A: remote round {round_number}",
        )
        _git(collaborator, "add", "Child.md")
        _git(
            collaborator,
            "commit",
            "-m",
            f"Child: collaborator round {round_number}",
        )
        _git(collaborator, "push", "origin", "main")
        collaboration_base = _git(collaborator, "rev-parse", "HEAD^").stdout.strip()

        old_root = "A: deck" if round_number == 1 else "A: local round 1"
        _replace(
            shared_root / "Deck.md",
            old_root,
            f"A: local round {round_number}",
        )
        private.write_text(
            f"private round {round_number}\n",
            encoding="utf-8",
        )
        _git(collection, "add", "Private.md")

        run_submit(
            SimpleNamespace(
                repo="owner/repo",
                message=f"Clarify shared history round {round_number}",
                commit=True,
            )
        )

        branch, split_sha, title = submitted[-1]
        assert title == f"Clarify shared history round {round_number}"
        assert repo.run(["branch", "--list", branch]).stdout == ""
        assert _git(collection, "status", "--short").stdout == "M  Private.md\n"
        _git(collaborator, "fetch", "origin", branch)
        assert (
            _git(collaborator, "merge-base", "main", f"origin/{branch}").stdout.strip()
            == collaboration_base
        )
        remote_split = _git(
            collaborator,
            "rev-parse",
            f"origin/{branch}",
        ).stdout.strip()
        assert remote_split == split_sha
        _git(
            collaborator,
            "merge",
            "--no-ff",
            f"origin/{branch}",
            "-m",
            f"Merge local round {round_number}",
        )
        _git(collaborator, "push", "origin", "main")

        private.write_text(
            f"private round {round_number}\nunstaged private detail\n",
            encoding="utf-8",
        )
        private_status = _git(collection, "status", "--short").stdout
        private_index = _git(collection, "show", ":Private.md").stdout
        private_worktree = private.read_bytes()
        run_update(SimpleNamespace(repo="owner/repo"))
        assert _git(collection, "status", "--short").stdout == private_status
        assert _git(collection, "show", ":Private.md").stdout == private_index
        assert private.read_bytes() == private_worktree
        _git(
            collection,
            "commit",
            "-m",
            f"Private: round {round_number}",
            "--",
            "Private.md",
        )
        remote_head = _git(collaborator, "rev-parse", "HEAD").stdout.strip()
        assert (
            repo.run(
                [
                    "rev-parse",
                    "HEAD:shared/owner/repo",
                ]
            ).stdout.strip()
            == repo.run(["rev-parse", f"{remote_head}^{{tree}}"]).stdout.strip()
        )

    submission_count = len(submitted)
    head = repo.head()
    run_submit(SimpleNamespace(repo="owner/repo", message=None))
    assert len(submitted) == submission_count
    assert repo.head() == head
    subjects = repo.run(
        ["log", "--first-parent", "--format=%s", "-7"]
    ).stdout.splitlines()
    assert "AnkiOps: update shared/owner/repo from GitHub" in subjects
    assert "AnkiOps: record submission history for shared/owner/repo" in subjects
    assert "Private: round 2" in subjects
