# Git operations triggered by AnkiOps

An AnkiOps collection has one required root Git repository for local decks.
The root ignores `/collab/`. Every directory at `collab/<owner>/<repo>` is a
separate Git repository. Together, the collection root (`.`) and these canonical
relative paths are the filesystem source registry. Git is the implementation
engine; collab-command output describes decks, updates, contributions, and pull
requests.

Opening the collection root in VS Code shows all Markdown files together. VS
Code's Source Control view shows the root and nested repositories separately.

## Collection sync checkpoints

Before `ankiops af` or `ankiops fa`, AnkiOps validates the complete source set.
It then commits local pending changes in the root repository and checkpoints
each dirty collab repository independently:

```text
AnkiOps: snapshot before anki-to-files
AnkiOps: snapshot before files-to-anki
```

The root snapshot uses explicit local paths. It never stages `/collab/`.

## Collab commands

`collab publish` moves a local deck tree into a new independent repository,
commits it, creates a public GitHub repository with authenticated `gh`, and
pushes `main`. A failed retry reuses the local repository and commit.

`collab subscribe` establishes an ongoing local copy at
`collab/<owner>/<repo>` without requiring authentication for public
repositories. Its local `ankiops/journal` branch has no upstream tracking
branch. `refs/ankiops/integrated`, `refs/ankiops/submission`, and
`refs/ankiops/uploaded` record the integrated upstream commit, prepared
contribution, and last confirmed upload.

`collab status` previews local changes, available updates, pending submissions,
conflict recovery state, live pull request state, and the next collab command.
It compares content trees rather than local commit topology and does not report
Anki application state. Validation runs automatically in every command that
needs it.

`collab update` commits all non-ignored local changes directly in the subscribed
repository before fetching. It performs the content merge there and records the
upstream commit in `refs/ankiops/integrated`. Its explicit content merge base
preserves edits made after an uploaded contribution, including after a squash
merge.

`collab submit` checkpoints all tracked, untracked, staged, unstaged, and deleted
non-ignored files in only the selected source, integrates upstream, and compares
content trees. Identical trees are a no-op. If changes remain, it uploads a
single synthetic commit on `ankiops/contribution` and creates or reuses a native
GitHub pull request. Later drafts replace that commit with force-with-lease while
retaining the same pull request. A contributor without write permission receives
or reuses a fork.

## Conflict and retry behavior

An overlapping update leaves the subscribed repository at its committed local
checkpoint. It records the frozen base, local, and upstream commits under
`refs/ankiops/conflict/` and writes editable files plus evidence under
`.ankiops/conflicts/<owner>/<repo>/`. Edit the reported conflict copy and rerun
the exact command. The retry uses those frozen commits even if GitHub advances;
newer work is integrated by the next update. Failed uploads and pull-request
creation retain one reusable submission, so retrying does not duplicate either.

An unexpected database schema or corrupt database is moved to
`.ankiops.db.corrupt`, then AnkiOps creates the current schema and retries once.
The recovery is logged before mutation and tells the user to run `ankiops init`
to relink the Anki profile. If creating the replacement also fails, that second
failure is propagated.
