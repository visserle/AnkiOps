# Git operations triggered by the AnkiOps CLI

This page lists the Git commands that AnkiOps can run from the `ankiops`
command line. It covers automatic snapshots, collection initialization, and the
GitHub-facing collaboration workflow.

The implementation lives mainly in `ankiops/git.py`, with call sites in
`ankiops/init.py`, `ankiops/cli.py`, `ankiops/llm/runner.py`, and
`ankiops/collab/`.

## Snapshot convention

AnkiOps uses pre-operation snapshots. A command runs read-only validation or
planning first, then snapshots the relevant paths, then starts mutating files or
external state.

Snapshot commits use this subject:

```text
AnkiOps: snapshot before <action>
```

Examples:

- `AnkiOps: snapshot before export`
- `AnkiOps: snapshot before import`
- `AnkiOps: snapshot before deserializing`
- `AnkiOps: snapshot before fixing image widths`
- `AnkiOps: snapshot before LLM task grammar`

## Snapshot Git sequence

Snapshot callers pass an explicit path scope to `git_snapshot`. The scoped
sequence is:

```text
git rev-parse --git-dir
git add -A -- <paths>
git diff --cached --quiet -- <paths>
git commit -m "AnkiOps: snapshot before <action>" -- <paths>
```

When AnkiOps must use the temporary broad fallback, the caller passes the
collection directory as the only path. That gives Git the collection root as the
pathspec:

```text
git add -A -- .
git diff --cached --quiet -- .
git commit -m "AnkiOps: snapshot before <action>" -- .
```

Important behavior:

- Scoped snapshots include pending changes only under the supplied paths.
- A clean scope creates no commit, even if unrelated files are dirty.
- Deleted tracked files inside the scope are included.
- If the directory is not a Git repository, Git is missing, or the commit fails,
  AnkiOps skips the snapshot or logs a warning and continues.
- Commands use the broad fallback when collab sources are present. Source-aware
  collab snapshots are future work.

## Commands that snapshot

| CLI command | Snapshot action | Snapshot scope |
| --- | --- | --- |
| `ankiops anki-to-markdown` / `ankiops am` | `export` | Root Markdown deck files plus `media/`. |
| `ankiops markdown-to-anki` / `ankiops ma` | `import` | Root Markdown deck files plus `media/` and `note_types/`. |
| `ankiops deserialize` | `deserializing` | Local target Markdown deck files derived from the input JSON after validation. |
| `ankiops fix-image-widths` | `fixing image widths` | Local Markdown deck files selected by `--deck` and `--no-subdecks`. |
| `ankiops llm <task> --run` | `LLM task <task>` | Local Markdown deck files containing queued candidate notes. |

Each command accepts `--no-auto-commit` / `-n`, except that LLM only accepts it
with `<task> --run`.

`ankiops collab pull --to-anki` runs `ankiops ma` after pulling. `ankiops collab
contribute --from-anki` runs `ankiops am` before preparing the contribution.
Until collab-aware scoping exists, those nested snapshots use the broad fallback
when collab sources are present.

## Other Git operations

| CLI command | Git or GitHub operations |
| --- | --- |
| `ankiops init` | Checks whether the collection directory is already inside a Git repo with `git rev-parse --git-dir`. If not, runs `git init`. |
| `ankiops collab publish <deck> <owner>/<repo>` | Requires a Git-backed collection, checks for staged changes, optionally checks or creates the GitHub repo, commits the deck move into `collab/<owner>/<repo>`, splits that subtree to a temporary branch, and pushes it to `main` on the target repo. |
| `ankiops collab subscribe <owner>/<repo>` | Requires a Git-backed collection and runs `git subtree add` for `https://github.com/<owner>/<repo>.git` at `collab/<owner>/<repo>` from branch `main`. |
| `ankiops collab pull [owner/repo]` | Requires a Git-backed collection and runs `git subtree pull` for one collab source, or every known collab source when the repo is omitted. |
| `ankiops collab contribute <owner>/<repo>` | Requires a Git-backed collection, commits changes under `collab/<owner>/<repo>`, splits that subtree to a temporary branch, pushes that branch when possible, and opens a PR with `gh` when available. |

## Collaboration details

All collaboration commands use `collab/<owner>/<repo>` as the local subtree
prefix and `https://github.com/<owner>/<repo>.git` as the remote URL. The
collab branch is `main`.

### `collab publish`

`publish` validates that the collection is a Git repo and that the Git index has
no staged changes:

```text
git rev-parse --git-dir
git diff --cached --quiet
```

It checks whether the target GitHub repository exists. If `gh` is installed, it
uses `gh repo view <owner>/<repo>` and can create a missing repo with
`gh repo create <owner>/<repo> --private` or `--public`. Without `gh`, it checks
the remote with:

```text
git ls-remote https://github.com/<owner>/<repo>.git
```

After writing the collab source files, it commits the move:

```text
git add -A -- <new-collab-paths>
git rm --cached -f --ignore-unmatch -- <original-deck-paths>
git diff --cached --quiet
git commit -m "AnkiOps: publish <deck> to collab/<owner>/<repo>"
```

Then it publishes the subtree:

```text
git subtree split --prefix collab/<owner>/<repo> -b ankiops-collab-<owner>-<repo>-<id>
git push https://github.com/<owner>/<repo>.git <branch>:main
```

If publish fails after creating a commit or temporary branch, AnkiOps attempts to
roll back with the appropriate subset of:

```text
git reset --mixed <initial-head>
git update-ref -d HEAD
git branch -D <branch>
git reset HEAD -- <paths>
git rm -r --cached --ignore-unmatch -- <paths>
```

### `collab subscribe`

`subscribe` refreshes the index and adds the GitHub repository as a subtree:

```text
git rev-parse --git-dir
git update-index -q --refresh
git subtree add --prefix collab/<owner>/<repo> https://github.com/<owner>/<repo>.git main
```

### `collab pull`

`pull` refreshes the index and pulls one or more collab subtrees:

```text
git rev-parse --git-dir
git update-index -q --refresh
git subtree pull --prefix collab/<owner>/<repo> https://github.com/<owner>/<repo>.git main
```

### `collab contribute`

`contribute` commits local collab source changes, creates a subtree branch, and
tries to prepare a pull request:

```text
git rev-parse --git-dir
git add -A -- collab/<owner>/<repo>
git diff --cached --quiet -- collab/<owner>/<repo>
git commit -m "AnkiOps: contribute collab/<owner>/<repo>" -- collab/<owner>/<repo>
git subtree split --prefix collab/<owner>/<repo> -b ankiops-collab-<owner>-<repo>-<id>
git push https://github.com/<owner>/<repo>.git <branch>:<branch>
gh pr create --repo <owner>/<repo> --head <branch> --base main --fill
```

If `gh` is not installed, or if the push fails, AnkiOps leaves the branch name in
the log so the user can push it and open a PR manually.

## Commands that do not run Git

The following CLI paths do not trigger Git operations directly:

- `ankiops serialize`
- `ankiops note-types`
- `ankiops collab status`
- `ankiops llm` status output
- `ankiops llm <task>` dry-run planning
- `ankiops llm --job <job_id|latest>`
- `ankiops --help` and `ankiops --version`
