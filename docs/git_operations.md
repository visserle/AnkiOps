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
branch. `refs/ankiops/integrated` and `refs/ankiops/uploaded` record the content
bases needed for updates and submissions.

`collab status` previews local changes, available updates, pending submissions,
conflict recovery state, live pull request state, and the next collab command.
It compares content trees rather than local commit topology and does not report
Anki application state. Validation runs automatically in every command that
needs it.

`collab update` prepares the integration in an isolated transaction. It changes
the subscribed repository only after integration succeeds. Its explicit content
merge base preserves edits made after an uploaded contribution, including after
a squash merge. It recommends `ankiops fa` only when top-level deck
Markdown or files under `media/` or `note_types/` changed.

`collab submit` checkpoints all tracked, untracked, staged, unstaged, and deleted
non-ignored files in only the selected source, integrates upstream, and compares
content trees. Identical trees are a no-op. If changes remain, it uploads a
single synthetic commit and creates or reuses a native GitHub pull request.
Later drafts replace that commit with force-with-lease while retaining the same
pull request. A contributor without write permission receives or reuses a fork.

## Conflict and retry behavior

An overlapping update does not alter the subscribed repository. It writes an
editable conflict file plus base, local, and upstream copies under
`.ankiops/conflicts/`. Edit the reported conflict file, remove its markers, and
rerun the same `collab update` command. Failed uploads and failed pull-request
creation retain one reusable submission; rerunning `collab submit` does not
duplicate the contribution or pull request.

An unexpected database schema or corrupt database is moved to
`.ankiops.db.corrupt`, then AnkiOps creates the current schema and retries once.
The recovery is logged before mutation and tells the user to run `ankiops init`
to relink the Anki profile. If creating the replacement also fails, that second
failure is propagated.
