# CLI output design

AnkiOps uses Rich for human-facing terminal output. Commands should share the
same hierarchy, language, spacing, and status symbols.

## Output hierarchy

Commands print these sections when they contain useful information:

1. One outcome line with a status symbol and a past-tense result.
2. Aligned summary rows for the command's main subjects.
3. Details for changed or failed items.
4. A `Next` section when the user needs to act.

Successful commands start with `✓`. Commands that finish with recoverable
problems start with `!`. Operational failures start with a red `×`, write to
stderr, and return a non-zero exit code.

Default output describes results. Diagnostic events such as connections, cache
hits, snapshots, and internal phases belong in `--debug` output.

## Language

- Use `Anki → files` and `Files → Anki` for sync directions.
- Use short count phrases separated by `·`.
- Use `up to date` when a checked subject has no changes.
- Name the affected item before its counts.
- Print commands without quotes so users can copy them.
- Do not tell users to edit `note_key` or `note_type` comments. AnkiOps owns
  both comments.
- `fa` assigns note keys when it imports keyless Markdown notes. Report those
  notes as successful creations, for example `2 created (keys assigned)`.

## Detail policy

Clean runs stop after the summary rows. Runs with changes list changed decks.
Warnings name the affected deck, file, or note group and include one recovery
step. Errors include the source item and the reason.

Do not list unchanged decks. Do not print empty sections.

Reserve structured output for normal results and common safeguards. Rare
preflight failures use one concise red `Error:` message with the action when it
is clear. They do not need summary rows or a `Next` section:

```text
Error: --no-subdecks requires --deck
```

```text
Error: Collection has no linked Anki profile. Run ankiops init.
```

Invalid arguments exit with code 2. Missing state and environment failures exit
with code 1. Put exception details and tracebacks under `--debug`.

## `af`: Anki to files

Changed run:

```text
✓ Synced Anki to files
  Notes       3 decks · 128 notes · 1 created · 4 updated
  Media       6 checked · 2 pulled

  Changed decks
  Biology                 1 created · 1 updated
  German::Verbs           3 updated
```

Clean run:

```text
✓ Synced Anki to files
  Notes       3 decks · 128 notes · up to date
  Media       6 files checked · up to date
```

An `af` run may preserve local Markdown notes that have not gone through their
first `fa` import. The warning should explain that `fa` assigns their keys:

```text
! Anki → files completed with warnings
  Notes       3 decks · 126 synced · 2 waiting for import

  2 local notes kept safely
  They are waiting for their first import into Anki.

  Next
  Run ankiops fa; AnkiOps will assign their note keys automatically.
```

## `fa`: Files to Anki

Changed run:

```text
✓ Synced files to Anki
  Notes        3 decks · 128 notes · 2 created (keys assigned) · 5 updated
  Note types   7 checked · 1 updated
  Media        6 checked · 1 hashed · 1 synced

  Changed decks
  Biology                 2 created · 1 updated
  German::Verbs           4 updated
```

Clean run:

```text
✓ Synced files to Anki
  Notes        3 decks · 128 notes · up to date
  Note types   7 checked · up to date
  Media        6 files checked · up to date
```

A recoverable problem should leave successful work visible and explain the
remaining action:

```text
! Files → Anki completed with warnings
  Notes        3 decks · 128 notes · 2 created (keys assigned) · 5 updated
  Note types   7 checked · up to date
  Media        6 checked · 1 synced · 1 missing locally

  1 referenced media file not found
  Biology references media/cell.png, but the file is missing locally.

  Next
  Add the file or remove the reference, then run ankiops fa again.
```

## Rich presentation

- Use bold text for the outcome and section headings.
- Use green for success symbols, yellow for warnings, and red for errors.
- Keep labels and unchanged context dim.
- Use a Rich table without borders for aligned summary and deck rows.
- Preserve readable output when color is disabled or stdout is not a terminal.
- Keep clickable file links where a path helps the user act.

## `collab status`

`collab status` prints one collection result followed by a stable table. The
table uses four columns: `Repository`, `Local`, `Upstream`, and `Submission`.
Users can compare several subscriptions without learning separate layouts for
clean, changed, and submitted decks.

Clean state:

```text
✓ Checked collab decks
  2 decks · up to date

  Repository              Local    Upstream    Submission
  owner/anatomy           clean    up to date  —
  study-group/biology     clean    up to date  —
```

An available update adds one next command:

```text
! Checked collab decks
  2 decks · 1 update available

  Repository              Local    Upstream           Submission
  owner/anatomy           clean    1 update available —
  study-group/biology     clean    up to date         —

  Next
  ankiops collab update owner/anatomy
```

Use file language for working-tree changes. Print `2 files changed`, not
`2 changed paths`. Translate Git status codes in the detail rows:

```text
  owner/anatomy
  modified  Deck.md
  new       media/diagram.png
```

Use `contribution ready` for committed local work. A commit count does not tell
the user how many files or notes changed.

If AnkiOps cannot reach GitHub, show the local state and mark the remote columns
as unavailable:

```text
! Checked local collab state
  2 decks · GitHub status unavailable

  Repository              Local          Upstream     Submission
  owner/anatomy           1 file changed unavailable  unknown
  study-group/biology     clean           unavailable  unknown

  Status
  Could not reach GitHub. Local state is shown above.
```

Do not print `Next` for an unavailable remote. Repeating `collab status` cannot
fix the network, credentials, or GitHub service state. Print `Next` only when
AnkiOps can name a command that advances the current workflow.

## `collab update`

`collab update owner/repository` requires one repository and mutates only that
subscription. Use `collab status` without a repository to inspect all
subscriptions.

Successful runs start with `Collab update complete`. Failed runs start with
`Collab update failed`. Keep the repository name in a summary row so every
scenario uses the same heading and alignment.

Separate upstream changes from local work. `collab update` checkpoints local
files before it integrates GitHub changes, so a run may report both groups:

```text
✓ Collab update complete
  Repository   owner/anatomy
  Upstream     3 files integrated
  Local work   2 files committed · ready to submit
  Anki         unchanged

  Upstream files
  modified     Deck.md
  new          media/heart.png
  modified     README.md

  Local contribution
  modified     Mnemonics.md
  new          media/diagram.png

  Next
  Review the integrated changes in VS Code

  Apply the reviewed changes to Anki
    ankiops fa

  Submit your local contribution
    ankiops collab submit owner/anatomy
```

The `Anki` row reports state. `collab update` changes files and leaves Anki
unchanged. Put recommendations under `Next`.

Each `Next` entry uses an action sentence. Put its command on the following
indented line. An action without a command, such as reviewing changes in VS
Code, stays on one line. This layout keeps descriptions useful without creating
long command lines.

Documentation-only updates still recommend a review but omit `fa`. Clean runs
omit `Next`. A conflict reports that subscribed files remain unchanged, names
the conflict file and recovery directory, then gives the retry command after
the resolution step:

```text
× Collab update failed
  Repository         owner/anatomy
  Repository files   reset to the committed local checkpoint
  Conflict           Deck.md
  Resolve in         .ankiops/conflicts/owner/anatomy/

  Next
  Resolve the conflict in the file above

  Retry the update
    ankiops collab update owner/anatomy
```

## `collab submit`

`collab submit` reports the local contribution, GitHub upload, and pull-request
state as separate rows. Successful runs use `Collab submit complete` for new,
updated, unchanged, and open pull requests.

New pull request:

```text
✓ Collab submit complete
  Repository     owner/anatomy
  Contribution   2 files submitted
  Commit         Clarify cardiac anatomy
  Pull request   opened
  GitHub         github.com/owner/anatomy/pull/24

  Contribution files
  modified       Deck.md
  new            media/diagram.png
```

An open pull request completes the workflow, so the command omits `Next`. The
GitHub URL remains visible and clickable in the summary.

Use `no changes` when the local content matches upstream. An existing current
pull request reuses the prepared commit and managed branch; a retry may repeat
the force-push but must not create another pull request.

Failures show how far the operation reached. If an upload fails after AnkiOps
commits the contribution, report the local commit and the failed GitHub state:

```text
× Collab submit failed
  Repository           owner/anatomy
  Local contribution   committed safely
  GitHub               not uploaded
  Error                GitHub authentication failed

  Next
  Sign in to GitHub
    gh auth login

  Retry the submission
    ankiops collab submit owner/anatomy
```

AnkiOps owns the remote `ankiops/contribution` branch. Every non-noop submit or
retry force-replaces it with the prepared snapshot, then reuses or creates the
pull request. Do not report remote divergence or branch cleanup: GitHub-side
edits to this managed branch are outside the supported workflow.

## `collab subscribe`

Successful subscriptions report the repository, destination, downloaded
content, and Anki state:

```text
✓ Collab subscribe complete
  Repository   owner/anatomy
  Saved to     collab/owner/anatomy/
  Content      4 decks · 326 notes · 18 media files
  Anki         unchanged

  Next
  Review the subscribed decks in VS Code

  Apply the subscribed decks to Anki
    ankiops fa
```

The destination path should remain clickable. AnkiOps already parses the deck
files to validate note keys, so the success output can count decks and notes.

Failed subscriptions report the final filesystem state. Use
`Local folder  not created` after AnkiOps cleans up an incomplete clone. Do not
say that AnkiOps removed local files; users may read that as data loss outside
the new subscription folder.

If the subscription exists, report `Existing files  left unchanged`. If a fork
declares another repository as its canonical source, name both repositories and
give the canonical subscribe command under `Next`.

## `collab publish`

`collab publish` moves a local deck tree into an independent collab repository,
publishes that repository on GitHub, and transfers sync ownership. Report those
results as separate rows:

```text
✓ Collab publish complete
  Deck         Psychology
  Repository   owner/psychology
  Moved to     collab/owner/psychology/
  Content      3 decks · 326 notes · 18 media files
  GitHub       public · github.com/owner/psychology
  Anki         unchanged

  Published decks
  Psychology
  Psychology::Learning
  Psychology::Memory

  Next
  Review the moved deck in VS Code
    collab/owner/psychology/

  Apply the moved deck to Anki
    ankiops fa
```

The `Next` section repeats the new collab folder because the deck no longer
lives at its original collection path. Keep that path clickable.

Failed runs must distinguish the source deck from a prepared copy. Use
`Local deck  left unchanged` when publish stops before the handoff. If AnkiOps
prepared a collab repository before an upload failure, report its path as
`Prepared copy` and keep the retry command under `Next`.

Missing note keys block publishing. Recommend `ankiops fa`, which assigns the
keys, before the publish retry. Repository-name collisions report that both the
local deck and existing GitHub repository remain unchanged.

## `note-types`

The list command prints two borderless tables. The first table explains how
AnkiOps identifies each note type and which optional labels it accepts:

```text
Note types                                      7 configured · 30 unique labels

Note type                  Identified by                    Optional labels
AnkiOpsChoice              Q:, A: + any C1:–C8:             E:, M:, AI:
AnkiOpsCloze               T:                               E:, M:, AI:
AnkiOpsClozeHideAll        THA:                             E:, M:, AI:
AnkiOpsImageOcclusion      IO_ID:, IO_IM:, IO_QM:,          IO_H:, IO_F:, IO_R:,
                           IO_AM:, IO_OM:                    IO_S:, IO_E1:, IO_E2:
AnkiOpsInput               Q:, I:                           E:, M:, AI:
AnkiOpsQA                  Q:, A:                           E:, M:, AI:
AnkiOpsReversed            F:, B:                           D:, E:, M:, AI:
```

The second table provides the inverse lookup from labels to fields and note
types. Keep the `Label`, `Field`, `Role`, and `Used by` columns. Group numbered
families such as `C1:–C8:` and `IO_E1:–IO_E2:` to reduce repetition without
hiding mappings.

`note-types --add` uses `Note type add complete` or `Note type add failed`.
Successful copies report field and template counts, the destination, and each
created file. Keep destination and configuration paths clickable. If a local
note type exists, report `Existing files  left unchanged`. If Anki does not
contain the requested type, list the available Anki types and report
`Local folder  not created`.

## `init`

`init` reports the collection state that users may want to verify. Omit Git,
workspace, media, and configuration setup details.

```text
✓ AnkiOps initialized
  Collection     ~/Anki/
  Anki profile   Default
  Note types     7 installed
  Anki           unchanged

  Next
  If you use the AnkiOps add-on, set ankiops_dir to
    ~/Anki/
```

`init --tutorial` adds one summary row:

```text
  Tutorial       AnkiOps Tutorial.md
```

Keep add-on configuration as the only `Next` item after a successful run. Do
not recommend opening VS Code or running `fa` here. A connection failure names
Anki as the cause and tells the user to start Anki before retrying `init`.

## `serialize`

Successful runs report the exported content and JSON destination. The saved
file completes the command, so omit `Next`.

```text
✓ Serialized collection
  Content    3 decks · 128 notes
  Saved to   AnkiCollection.json
```

A deck-scoped run adds `Scope`. State whether the scope includes subdecks.
Keep the output path clickable.

Validation failures report that AnkiOps did not create the output file. For a
duplicate note key, list the two clickable Markdown files without note ordinal
numbers:

```text
× Serialization failed
  Output   AnkiCollection.json not created
  Error    duplicate note key nk-42

  Found in
  Biology.md
  Biology__Cells.md
```

## `deserialize`

Use `JSON import complete`, `JSON import complete with warnings`, and
`JSON import failed` as the outcome lines. Successful runs report the input,
JSON content, file results, and Anki state:

```text
✓ JSON import complete
  Input     AnkiCollection.json
  Content   3 decks · 128 notes
  Files     3 created
  Anki      unchanged

  Created files
  created   Biology.md          42 notes
  created   Biology__Cells.md   44 notes
  created   German.md           42 notes

  Next
  Review the imported decks in VS Code

  Apply the reviewed changes to Anki
    ankiops fa
```

`--overwrite` reports created and replaced files as separate counts. Without
that option, keep existing files unchanged and report them under `Skipped
files`. The recovery step tells the user to review those files before running
`ankiops deserialize --overwrite`.

AnkiOps validates the full JSON document before it writes deck files. A failed
validation reports `Files  left unchanged`, the input path, and the error. Keep
input and Markdown paths clickable.

## `fix-image-widths`

Changed runs report the selected mode, deck scope, checked counts, changed
counts, and Anki state:

```text
✓ Image widths updated
  Mode      normalize within ±5 px
  Scope     all decks
  Checked   3 decks · 128 notes · 42 images
  Changed   2 decks · 5 notes · 7 images
  Anki      unchanged

  Changed decks
  Biology                4 images
  German::Vocabulary     3 images

  Next
  Review the updated widths in VS Code

  Apply the reviewed changes to Anki
    ankiops fa
```

Use `force 500 px` for `--width 500`. A deck scope states whether it includes
subdecks. Clean runs use `Image widths up to date`, omit changed counts and
deck details, and stop without `Next`.

A changed run edits Markdown files and leaves Anki unchanged. Keep the review
step before `fa`. If the selected deck does not exist, report `Files  left
unchanged` and name the missing deck.

## `llm`: status

`ankiops llm` validates the task catalog and shows recent job results. Replace
the internal field-rule count with each task's scope. Replace the `persisted`
flag with `files changed`, `no changes`, `failed`, or `running`.

```text
✓ LLM configuration ready
  Tasks         5 configured
  Recent jobs   4 shown

  Tasks
  Task           Model           Scope
  fix-grammar    gpt-5.4-mini    all decks
  fix-html       gpt-5.4-mini    all decks
  review         gpt-5.4-mini    all decks
  tag            gpt-5.4-nano    all decks
  translate      gpt-5.4-mini    all decks

  Recent jobs
  Job   Task           Model           Result          Created
  #18   review         gpt-5.4-mini    running         Today 14:41
  #17   fix-grammar    gpt-5.4-mini    files changed   Today 14:32
  #16   tag            gpt-5.4-nano    no changes      Today 13:08
  #15   translate      gpt-5.4-mini    failed          Yesterday 18:41
```

Omit `Next` because the dashboard does not select a task or job. An invalid
catalog uses `LLM configuration invalid`, counts ready and invalid tasks, and
names each invalid task file. Put the clickable task path under `Next` when the
user needs to edit it.

## `llm`: task plan

`ankiops llm <task>` creates a dry-run plan. Report the task, model, deck scope,
eligible notes, request estimate, cost estimate, and prompt files. The `Dry run`
row confirms that AnkiOps did not call the API or edit files.

```text
✓ LLM task plan ready
  Task       fix-grammar
  Model      gpt-5.4-mini
  Scope      all decks
  Notes      128 checked · 116 eligible · 12 skipped
  Requests   39 estimated · up to 3 notes each · low reasoning
  Cost       $0.18 estimated
  Prompts    llm/fix-grammar.yaml · llm/_system_prompt.md
  Dry run    API not called · files unchanged

  Field access
  Note type       Notes   Editable               Read-only   Hidden                  Tags
  AnkiOpsQA       62      Question, Answer       —           Extra, More, AI Notes   hidden
  AnkiOpsChoice   36      Question, Choice 1–8   Answer      Extra, More, AI Notes   hidden
  AnkiOpsCloze    18      Text                   —           Extra, More, AI Notes   hidden

  Next
  Run this task
    ankiops llm fix-grammar --run
```

Keep prompt paths clickable. Do not print the full prompt in default output;
the task and prompt files provide the review surface in VS Code. Preserve model
and deck overrides in the run command under `Next`.

If the plan finds no eligible notes, use `No notes eligible for LLM task`, show
the skipped count and reason, and omit the run command.

## `llm`: task run

`ankiops llm <task> --run` uses one transient Rich progress bar. Spell out
`updated`, `unchanged`, `skipped`, and `errors`; avoid abbreviated progress
labels. Remove the progress bar when the run finishes.

A successful run with edits leaves this summary:

```text
✓ LLM task complete
  Job        #18
  Task       fix-grammar
  Model      gpt-5.4-mini
  Notes      128 checked · 42 updated · 74 unchanged · 12 skipped
  Requests   39 · 84,200 input tokens · 21,600 output tokens
  Cost       $0.16
  Files      2 decks updated
  Anki       unchanged

  Changed decks
  Biology                27 notes
  German::Vocabulary     15 notes

  Next
  Review the LLM changes in VS Code

  Apply the reviewed changes to Anki
    ankiops fa
```

A successful job without edits reports `Files  unchanged` and omits `Next`.
Failed jobs report errors and canceled notes, the incurred cost, and
`Files  left unchanged`. AnkiOps writes no accumulated edits when a job fails.
Put the job inspection command under `Next`, along with a recovery step when
the error suggests one.

## `llm`: job detail

`ankiops llm --job <id>` reports the durable state and result of one job. A
completed job with edits groups changed fields and note counts by deck:

```text
✓ LLM job #18 complete
  Task       fix-grammar
  Model      gpt-5.4-mini
  Timing     Today 14:32 · 18s
  Notes      128 checked · 42 updated · 74 unchanged · 12 skipped
  Requests   39 completed · 84,200 input · 21,600 output tokens
  Cost       $0.16
  Files      2 decks updated
  Anki       unchanged

  Changed decks
  Deck                    Notes   Fields
  Biology                 27      Question, Answer
  German::Vocabulary      15      Question
```

Do not list every unchanged or skipped note. Keep successful request payloads
out of default output. A completed job without edits reports `Files  unchanged`
and omits the changed-deck table.

A failed job reports errors and canceled notes, then lists only the failed
requests. Put a concrete recovery step under `Next` when one is available. A
running job reports progress, results so far, active requests, and `Files
unchanged while running`. Completed, unchanged, and running states do not need
a `Next` section.

## Welcome screen

The bare `ankiops` command is the only output that uses an `=` banner. It is a
compact command index rather than a command result, so it has no status symbol
or `Next` section:

```text
============================================================
AnkiOps 0.6.6
A bidirectional bridge between Anki and the filesystem
============================================================

Core workflow
  ankiops af                  Anki → files
  ankiops fa                  Files → Anki

Collection
  ankiops init                Initialize this folder
  ankiops note-types          Show note types and labels
  ankiops serialize           Export decks to JSON
  ankiops deserialize         Import JSON as deck files
  ankiops fix-image-widths    Normalize Markdown image widths

Extensions
  ankiops llm                 Plan and run LLM tasks
  ankiops collab              Share decks through GitHub

Help
  ankiops <command> --help     Show command help
  ankiops --debug <command>    Enable debug logging
```

Read the version dynamically. The global `--debug` flag comes before the
command. Do not repeat the command index as a separate examples section.

## Shared failures

Commands that need Anki use the same connection failure. Name the attempted
operation, state that Anki is unavailable, and confirm which local state
AnkiOps left unchanged:

```text
× Files → Anki failed
  Anki     unavailable
  Files    left unchanged

  Could not connect to Anki
  Make sure Anki is running and AnkiOpsConnect or AnkiConnect is enabled.

  Next
  Retry Files → Anki
    ankiops fa
```

Use the user's original command under `Next`, including its options. Put
connection exceptions and HTTP details under `--debug`.

Commands that require a collection name the current folder and the output they
did not create:

```text
× Serialization failed
  Collection   not initialized
  Output       AnkiCollection.json not created

  No AnkiOps collection found in ~/Biology/

  Next
  Initialize this folder
    ankiops init

  Then retry the JSON export
    ankiops serialize
```

Keep the folder clickable. Preserve the original command and its options in
the retry step.

If Anki has a different profile open, stop before reading or writing sync
content:

```text
× Files → Anki stopped
  Collection       ~/Anki/
  Linked profile   Default
  Open profile     Work
  Files            left unchanged
  Anki             unchanged

  Next
  Open Default in Anki

  Then retry Files → Anki
    ankiops fa
```

Use `stopped` because the profile check protects both sides from changes. Keep
the collection path clickable and preserve the original command in the retry
step.

## Implementation boundary

Domain and sync code should return structured results. A shared presentation
layer should turn those results into Rich renderables. Command orchestration
should not assemble styled output through logging calls.
