# Deleted Deck Git Snapshot Bug

Date: 2026-06-16
Status: Reported before fix

## Summary

Deleted Markdown deck files are not included in automatic pre-sync Git snapshot
commits for syncs between Anki and the filesystem.

## Impact

When a tracked deck file is deleted locally before a sync, `ankiops af` or
`ankiops fa` can create a snapshot commit that omits the deletion. The following
sync operation may recreate or otherwise mutate deck files, leaving the Git
history without the user's explicit deck deletion.

## Expected Behavior

Automatic sync snapshots should include deletions of tracked Markdown deck
files in the sync scope, matching the documented behavior in
`docs/git_operations.md`: deleted tracked files inside the scope are included.

## Observed Cause

The sync CLI builds its snapshot scope from `DeckSource.local(...).deck_files()`.
That helper discovers only Markdown files that currently exist on disk. A deck
file that has already been deleted is absent from the scoped path list before
`git_snapshot()` runs, so Git never receives that deleted path as a pathspec.

The lower-level `git_snapshot()` implementation already handles deleted tracked
paths when the caller supplies the deleted path explicitly; the failure is in
the sync snapshot scope construction.

## Reproduction Plan

1. Create and commit `DeletedDeck.md` in a temporary AnkiOps collection.
2. Delete `DeletedDeck.md` from the working tree.
3. Run a sync command with automatic snapshots enabled.
4. Inspect the generated snapshot commit.

The snapshot commit should contain `D DeletedDeck.md`. Current behavior is
expected to omit that deletion.

## Fix Direction

Make sync snapshot path discovery include tracked root Markdown deck files even
when they no longer exist on disk, while preserving the current exclusions for
reserved Markdown files and unrelated non-deck files.
