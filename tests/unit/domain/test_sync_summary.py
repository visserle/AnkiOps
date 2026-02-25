"""Tests for sync summary aggregation and projection."""

from ankiops.models import (
    Change,
    ChangeType,
    CollectionResult,
    SyncResult,
)


def test_note_summary_prefers_higher_priority_change_per_entity():
    result = SyncResult.for_notes(name="Deck", file_path=None)
    result.changes = [
        Change(ChangeType.SKIP, 1, "note_key: one"),
        Change(ChangeType.UPDATE, 1, "note_key: one"),
        Change(ChangeType.CREATE, 2, "note_key: two"),
        Change(ChangeType.DELETE, 3, "note_key: three"),
    ]

    summary = result.summary

    assert summary.updated == 1
    assert summary.created == 1
    assert summary.skipped == 0
    assert summary.deleted == 1
    assert summary.total == 2


def test_media_summary_tracks_sync_and_hash_with_deduplication():
    result = SyncResult.for_media()
    result.changes = [
        Change(ChangeType.HASH, "a.png", "a.png"),
        Change(ChangeType.SYNC, "a.png", "a.png"),
        Change(ChangeType.SKIP, "b.png", "b.png"),
    ]

    summary = result.summary

    assert summary.synced == 1
    assert summary.hashed == 0
    assert summary.skipped == 1
    assert summary.total == 2


def test_collection_export_extra_changes_update_counters_not_total():
    deck_result = SyncResult.for_notes(name="Deck", file_path=None)
    deck_result.changes = [Change(ChangeType.CREATE, 1, "note_key: one")]
    export_result = CollectionResult.for_export(
        results=[deck_result],
        extra_changes=[Change(ChangeType.DELETE, None, "file: Deck.md")],
    )

    summary = export_result.summary

    assert summary.created == 1
    assert summary.deleted == 1
    assert summary.total == 1
