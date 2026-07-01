# Git operations triggered by the AnkiOps CLI

This page lists the Git commands that AnkiOps can run from the `ankiops`
command line. It covers automatic snapshots, collection initialization, and the
GitHub-facing shared workflow.

The implementation lives mainly in `ankiops/git.py`, with call sites in
`ankiops/init.py`, `ankiops/cli.py`, `ankiops/llm/runner.py`, and
`ankiops/shared/`.

## Snapshot convention

AnkiOps uses pre-operation snapshots. A command runs read-only validation or
planning first, then snapshots the relevant paths, then starts mutating files or
external state.

Snapshot commits use this subject:

```text
AnkiOps: snapshot before <action>
```

Examples:

- `AnkiOps: snapshot before anki-to-files`
- `AnkiOps: snapshot before files-to-anki`
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
- Commands use the broad fallback when shared sources are present. Source-aware
  shared snapshots are future work.

## Commands that snapshot

| CLI command | Snapshot action | Snapshot scope |
| --- | --- | --- |
| `ankiops anki-to-files` / `ankiops af` | `anki-to-files` | Root Markdown deck files plus `media/`. |
| `ankiops files-to-anki` / `ankiops fa` | `files-to-anki` | Root Markdown deck files plus `media/` and `note_types/`. |
| `ankiops deserialize` | `deserializing` | Local target Markdown deck files derived from the input JSON after validation. |
| `ankiops fix-image-widths` | `fixing image widths` | Local Markdown deck files selected by `--deck` and `--no-subdecks`. |
| `ankiops llm <task> --run` | `LLM task <task>` | Local Markdown deck files containing queued candidate notes. |

Each command accepts `--no-auto-commit` / `-n`, except that LLM only accepts it
with `<task> --run`.

`ankiops shared update --to-anki` runs `ankiops fa` after updating and currently
uses the broad snapshot fallback when shared sources are present. `ankiops shared
submit --from-anki` runs `ankiops af` with auto-commit disabled; only an explicit
`shared submit --commit` may create the submission commit.

## Other Git operations

| CLI command | Git or GitHub operations |
| --- | --- |
| `ankiops init` | Checks whether the collection directory is already inside a Git repo with `git rev-parse --git-dir`. If not, runs `git init`. |
| `ankiops shared create <deck> <owner>/<repo>` | Requires a Git-backed collection, checks for staged changes, optionally checks or creates the GitHub repo, commits the deck move into `shared/<owner>/<repo>`, splits that subtree to a temporary branch, and pushes it to `main` on the target repo. |
| `ankiops shared add <owner>/<repo>` | Requires a Git-backed collection and runs `git subtree add` for `https://github.com/<owner>/<repo>.git` at `shared/<owner>/<repo>` from branch `main`. |
| `ankiops shared update [owner/repo]` | Requires a Git-backed collection and runs `git subtree pull` for one shared source, or every known shared source when the repo is omitted. |
| `ankiops shared status <owner>/<repo>` | Shows shared and private working-tree changes, fetches remote `main`, compares committed subtree history, and explains what `submit` will do. |
| `ankiops shared submit <owner>/<repo>` | Requires shared changes to be committed unless `--commit` is explicit, splits and rejoins the subtree history, pushes a temporary branch when changes exist, and opens a PR with `gh` when available. |

## Shared details

All shared commands use `shared/<owner>/<repo>` as the local subtree
prefix and `https://github.com/<owner>/<repo>.git` as the remote URL. The
shared branch is `main`.

### `shared create`

`create` validates that the collection is a Git repo and that the Git index has
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

After writing the shared source files, it commits the move:

```text
git add -A -- <new-shared-paths>
git rm --cached -f --ignore-unmatch -- <original-deck-paths>
git diff --cached --quiet
git commit -m "AnkiOps: create shared/<owner>/<repo> from <deck>"
```

Then it splits the subtree, records a metadata-only history join, and pushes the
split commit:

```text
git subtree split --prefix shared/<owner>/<repo>
git commit-tree <HEAD-tree> -p HEAD -p <split-sha> \
  -m "AnkiOps: initialize subtree history for shared/<owner>/<repo>" \
  -m "<git-subtree trailers>"
git update-ref HEAD <rejoin-commit> <old-head>
git branch ankiops-shared-<owner>-<repo>-<id> <split-sha>
git push https://github.com/<owner>/<repo>.git <branch>:main
git branch -D <branch>
```

The rejoin commit has the same tree as its first parent. It records the split as
its second parent and adds `git-subtree-dir`, `git-subtree-mainline`, and
`git-subtree-split` trailers. This preserves remote ancestry without staging or
committing unrelated working-tree changes.

If create fails after creating a commit or temporary branch, AnkiOps attempts to
roll back with the appropriate subset of:

```text
git reset --mixed <initial-head>
git update-ref -d HEAD
git branch -D <branch>
git reset HEAD -- <paths>
git rm -r --cached --ignore-unmatch -- <paths>
```

### `shared add`

`add` refreshes the index and adds the GitHub repository as a subtree:

```text
git rev-parse --git-dir
git update-index -q --refresh
git subtree add --prefix shared/<owner>/<repo> \
  --message "AnkiOps: add shared/<owner>/<repo> from GitHub" \
  https://github.com/<owner>/<repo>.git main
```

### `shared update`

`update` refreshes the index and pulls one or more shared subtrees:

```text
git rev-parse --git-dir
git update-index -q --refresh
git subtree pull --prefix shared/<owner>/<repo> \
  --message "AnkiOps: update shared/<owner>/<repo> from GitHub" \
  https://github.com/<owner>/<repo>.git main
```

When the pull creates no commit, AnkiOps reports that the source is already up
to date. Because `git subtree pull` rejects any dirty worktree, AnkiOps
temporarily stashes changes outside the shared prefix and restores their exact
staged and unstaged state with `git stash pop --index`. Existing stashes are
left untouched; dirty files inside the shared prefix still block the pull.

### `shared status`

`status` is read-only with respect to collection files and history. It shows
short Git status lines for the selected shared prefix and for private paths,
fetches GitHub `main`, and classifies the committed shared state as current,
remote-ahead, local-ahead, same-content, diverged, or incompatible. Its final
line states whether `submit` will stop, do nothing, or open a pull request.

### `shared submit`

`submit` accepts `-m` / `--message`. The message becomes the pull request title;
with `--commit`, the collection commit receives an `AnkiOps:` prefix. Without
the option, the title is `Update shared deck <owner>/<repo>`.

Dirty shared files stop submission without changing history. Commit them
yourself, or pass `--commit` to give AnkiOps explicit permission to run:

```text
git add -A -- shared/<owner>/<repo>
git diff --cached --quiet -- shared/<owner>/<repo>
git commit -m "AnkiOps: <message>" -- shared/<owner>/<repo>
```

AnkiOps then fetches remote `main` and splits the subtree without creating a
branch:

```text
git fetch https://github.com/<owner>/<repo>.git main
git subtree split --prefix shared/<owner>/<repo>
```

If the split is already an ancestor of remote `main`, or both trees are equal,
the command reports that there are no changes and stops. Otherwise it requires
a common ancestor, records a metadata-only rejoin commit, and publishes the
split:

```text
git commit-tree <HEAD-tree> -p HEAD -p <split-sha> \
  -m "AnkiOps: record submission history for shared/<owner>/<repo>" \
  -m "<git-subtree trailers>"
git update-ref HEAD <rejoin-commit> <old-head>
git branch ankiops-shared-<owner>-<repo>-<id> <split-sha>
git push https://github.com/<owner>/<repo>.git <branch>:<branch>
gh pr create --repo <owner>/<repo> --head <branch> --base main \
  --title "<message>" --body "Submitted by AnkiOps from shared/<owner>/<repo>."
git branch -D <branch>
```

The local temporary branch is removed after a successful push. A failed push
keeps it for manual recovery. When `gh` is unavailable or PR creation fails, the
successfully pushed remote branch remains available for creating the PR by hand.

Shared sources created by older experimental AnkiOps versions do not have the
required ancestry metadata. Update and submit reject them with instructions to
recreate or re-add the source; there is no automatic migration.

## Commands that do not run Git

The following CLI paths do not trigger Git operations directly:

- `ankiops serialize`
- `ankiops note-types`
- `ankiops shared list`
- `ankiops llm` status output
- `ankiops llm <task>` dry-run planning
- `ankiops llm --job <job_id|latest>`
- `ankiops --help` and `ankiops --version`
