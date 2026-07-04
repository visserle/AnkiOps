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
`collab/<owner>/<repo>`.

`collab status` previews local changes, available updates, pending submissions,
Anki application state, conflict recovery state, pull request URLs, and the
next command. Validation runs automatically in every command that needs it.

`collab update` prepares the integration in an isolated transaction. It changes
the subscribed repository only after integration succeeds, changes Markdown
only, and tells the user to run `ankiops fa` after review.

`collab submit` automatically commits dirty files in only the selected source,
integrates upstream, and compares content trees. Identical trees are a no-op. If
changes remain, it uploads one reusable contribution and creates or reuses a
native GitHub pull request. A contributor without write permission receives or
reuses a fork.

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
