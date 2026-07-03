# Git operations triggered by AnkiOps

An AnkiOps collection has one required root Git repository for private decks.
The root ignores `/shared/`. Every directory at `shared/<owner>/<repo>` is a
separate Git repository. Git is the implementation engine; shared-command
output describes decks, updates, contributions, and pull requests.

Opening the collection root in VS Code shows all Markdown files together. VS
Code's Source Control view shows the root and nested repositories separately.

## Collection sync checkpoints

Before `ankiops af` or `ankiops fa`, AnkiOps validates the complete source set.
It then commits private pending changes in the root repository and checkpoints
each dirty shared repository independently:

```text
AnkiOps: snapshot before anki-to-files
AnkiOps: snapshot before files-to-anki
```

The root snapshot uses explicit private paths. It never stages `/shared/`.

## Shared commands

`shared publish` moves a private deck tree into a new independent repository,
commits it, creates the GitHub repository with authenticated `gh`, and pushes
`main`. A failed retry reuses the local repository and commit.

`shared subscribe` establishes an ongoing local copy at
`shared/<owner>/<repo>`.

`shared status` previews local changes, available updates, pending submissions,
Anki application state, conflict recovery state, pull request URLs, and the
next command. Validation runs automatically in every command that needs it.

`shared update` prepares the integration in an isolated transaction. It changes
the subscribed repository only after integration succeeds, changes Markdown
only, and tells the user to run `ankiops fa` after review.

`shared submit` automatically commits dirty files in only the selected source,
integrates upstream, and compares content trees. Identical trees are a no-op. If
changes remain, it uploads one reusable contribution and creates or reuses a
native GitHub pull request. A contributor without write permission receives or
reuses a fork.

## Conflict and retry behavior

An overlapping update does not alter the subscribed repository. It writes an
editable conflict file plus base, local, and upstream copies under
`.ankiops/conflicts/`. Edit the reported conflict file, remove its markers, and
rerun the same `shared update` command. Failed uploads and failed pull-request
creation retain one reusable submission; rerunning `shared submit` does not
duplicate the contribution or pull request.

Unexpected database schemas and corrupt databases are rejected without mutation.
Initialize a fresh collection instead.
