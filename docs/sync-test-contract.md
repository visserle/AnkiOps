# Sync Test Contract

This document defines the behavior contract that the database and sync test suites must enforce.
It is intentionally explicit so future changes can be judged against stable expectations.

## Scope

The suite covers note and deck synchronization via:

- `import_collection` (Markdown -> Anki)
- `export_collection` (Anki -> Markdown)
- Round-trip flows (`import -> export -> import` and `export -> import -> export`)

The suite must validate behavior for three collection states:

- Fresh collection: no prior mappings and no managed markdown files
- Running collection: mappings/files exist and are internally consistent
- Corrupted collection: SQLite mapping DB is unreadable or structurally invalid

## Canonical Source Of Truth

Direction determines the winner when the two sides disagree:

- Import: markdown is canonical
- Export: Anki is canonical
- Round-trip: the most recent directional sync defines canonical state

Sync mode is always full-sync in both directions. Partial-sync toggles were removed.

This policy is required to keep behavior deterministic until explicit conflict tracking is implemented.

## Operation Families

Each direction/state combination must cover these operation families.

- Create: note exists only on canonical side and is created on the target side
- Update: note exists on both sides and canonical fields overwrite target fields
- Move: note remains the same entity but changes deck/file location
- Delete: note removed on canonical side is removed from target side
- Conflict: incompatible concurrent edits requiring explicit policy
- Drift: metadata or mapping drift between DB, markdown, and Anki state

## Drift Taxonomy

The suite distinguishes drift modes so failures are diagnosable.

- D1 Missing note mapping: note key exists but DB has no key->id row
- D2 Stale note mapping: DB maps key->id but id no longer matches the true note
- D3 Missing deck mapping: deck exists in Anki but DB has no deck row
- D4 Stale deck mapping: DB has prior deck name for current deck id (rename)
- D5 Data loss drift: DB was recreated after corruption and mappings must be rebuilt

## Conflict Taxonomy

- C1 Duplicate `note_key` values in markdown across files
- C2 Directional content divergence for the same `note_key` (both sides changed)
- C3 Mapping conflict where one id is associated with a different key than expected

Current required policy:

- C1 must hard-fail import with a clear error
- C2 must resolve deterministically by direction (import=markdown wins, export=Anki wins)
- C3 must not silently duplicate notes

## Core Invariants

All tests in this suite should assert at least one invariant.

1. A managed note has exactly one stable `note_key`.
2. No operation may duplicate a note for an existing `note_key`.
3. Import then immediate re-import with no edits is idempotent.
4. Export then immediate re-export with no edits is idempotent.
5. Move operations preserve note identity (`note_key`, note id).
6. Delete operations remove stale entities from the target side.
7. Deck rename operations keep content but update filename and deck mapping.
8. DB corruption recovery creates `<db>.corrupt` backup and resumes with a clean DB.
9. After corruption recovery, mappings can be rebuilt from embedded `AnkiOps Key`.
10. Round-trip cycles do not introduce unexpected creates/updates/deletes.

## Scenario Naming

Scenario IDs use this shape:

`<DIR>-<STATE>-<OP>-<NNN>`

- `DIR`: `IMP`, `EXP`, `RT`
- `STATE`: `FRESH`, `RUN`, `CORR`
- `OP`: `CREATE`, `UPDATE`, `MOVE`, `DELETE`, `CONFLICT`, `DRIFT`
- `NNN`: sequence number

Examples:

- `IMP-FRESH-CREATE-001`
- `EXP-RUN-DRIFT-003`
- `RT-CORR-DELETE-002`

## Minimum Matrix For Phase 1

Phase 1 must include deterministic tests for:

- 3 directions (`IMP`, `EXP`, `RT`)
- 3 states (`FRESH`, `RUN`, `CORR`)
- 7 operation families (`CREATE`, `UPDATE`, `MOVE`, `DELETE`, `CONFLICT`, `DRIFT`, `RENAME`)

Target: 63 deterministic scenarios.

## Gating

- Conflict/drift behaviors not implemented yet should be represented with `xfail(strict=True)`.
- Once implementation lands, flip to required-pass tests and remove `xfail`.
- This suite is a release gate for sync/database changes.
