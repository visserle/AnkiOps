# Sync performance notes

## Measurement (Apple M2, Anki 26.9.4)

Measurements used a snapshot of a real collection: 93 decks, 2,672 managed
notes and 2,620 referenced media files. Times are elapsed process time,
including CLI startup and the default pre-sync Git snapshot, with warm OS,
Git and AnkiOps caches. They are not a guarantee for cold caches, large
changes, slow storage, or media transfers.

| Scenario | Elapsed time |
| --- | --- |
| Anki to files, unchanged (5 runs) | 0.68–0.75 s |
| Files to Anki, unchanged (5 runs) | 0.80–0.82 s |
| One file-side note edited | 0.83 s |
| One Anki-side note edited | 0.77 s |

The unchanged runs used the installed Python 3.12 CLI and the running Anki
profile through a read-only RPC proxy, with copies of the filesystem
collection and Anki media. The proxy rejected write actions. The one-note
measurements used independent copies of both collections and a headless Anki
backend, allowing the changes to be applied without touching the live profile.
Both tests included the default Git checkpoint. No live notes or media were
changed by the benchmarks.

Profiling identified unnecessary LLM imports on CLI startup, repeated deck and
note-type lookups in the add-on's card-info action, and unnecessary content
fingerprints when an exported deck has no keyless notes. The optimizations keep
Anki's existing note and card APIs and do not replace their behavior with
assumptions about the database's representation or card ordering.

For a meaningful comparison, measure both directions end-to-end with the same
collection snapshot, Anki version, Python executable, Git snapshot setting and
cache state. Include changed-note cases, not just no-op syncs. Run benchmarks
against isolated copies: both sync directions can delete or modify data.
