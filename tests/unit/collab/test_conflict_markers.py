from ankiops.collab.commands import _contains_conflict_markers


def test_conflict_marker_detection_requires_a_complete_marker_triplet(tmp_path):
    explanation = tmp_path / "Deck.md"
    explanation.write_text(
        "A: The values are ======= by definition.\n", encoding="utf-8"
    )

    assert not _contains_conflict_markers(explanation.read_bytes())

    explanation.write_text(
        "<<<<<<< local\nA: local\n=======\nA: upstream\n>>>>>>> upstream\n",
        encoding="utf-8",
    )

    assert _contains_conflict_markers(explanation.read_bytes())
