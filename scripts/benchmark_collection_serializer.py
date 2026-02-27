"""Benchmark collection serializer throughput with deterministic fixtures."""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

from ankiops.collection_serializer import (
    deserialize_collection_data,
    deserialize_collection_from_json,
    serialize_collection,
    serialize_collection_to_json,
)
from ankiops.config import deck_name_to_file_stem
from ankiops.db import SQLiteDbAdapter
from ankiops.fs import FileSystemAdapter

_SCENARIOS = (
    "serialize-memory",
    "serialize-json",
    "deserialize-memory",
    "deserialize-json",
)
_DEFAULT_NOTES = 5000
_DEFAULT_DECKS = 10


def _prepare_empty_collection(collection_dir: Path) -> None:
    db = SQLiteDbAdapter.load(collection_dir)
    try:
        db.save()
    finally:
        db.close()

    fs = FileSystemAdapter()
    fs.eject_builtin_note_types(collection_dir / "note_types")


def _deck_note_counts(total_notes: int, decks: int) -> list[int]:
    if total_notes <= 0:
        raise ValueError("--notes must be greater than zero")
    if decks <= 0:
        raise ValueError("--decks must be greater than zero")

    base, extra = divmod(total_notes, decks)
    return [base + (1 if index < extra else 0) for index in range(decks)]


def _seed_markdown_decks(collection_dir: Path, *, total_notes: int, decks: int) -> int:
    counts = _deck_note_counts(total_notes, decks)
    note_counter = 0

    for deck_index, note_count in enumerate(counts):
        if note_count == 0:
            continue

        deck_name = f"PerfDeck::{deck_index:03d}"
        file_path = collection_dir / f"{deck_name_to_file_stem(deck_name)}.md"
        lines: list[str] = []

        for _ in range(note_count):
            note_counter += 1
            lines.append(f"<!-- note_key: perf-{note_counter} -->")
            lines.append(f"Q: Question {note_counter}?")
            lines.append(f"A: Answer {note_counter}.")
            lines.append("")
            lines.append("---")
            lines.append("")

        while lines and lines[-1] in ("", "---"):
            lines.pop()

        file_path.write_text("\n".join(lines), encoding="utf-8")

    return note_counter


@contextmanager
def _serializer_paths(collection_dir: Path) -> Iterator[None]:
    note_types_dir = collection_dir / "note_types"
    with (
        patch(
            "ankiops.collection_serializer.get_collection_dir",
            lambda: collection_dir,
        ),
        patch(
            "ankiops.collection_serializer.get_note_types_dir",
            lambda: note_types_dir,
        ),
    ):
        yield


def _measure_once(*, scenario: str, notes: int, decks: int) -> float:
    with tempfile.TemporaryDirectory() as temp_dir:
        root_dir = Path(temp_dir)
        source_dir = root_dir / "source"
        source_dir.mkdir()
        _prepare_empty_collection(source_dir)
        _seed_markdown_decks(source_dir, total_notes=notes, decks=decks)

        if scenario == "serialize-memory":
            with _serializer_paths(source_dir):
                started_at = time.perf_counter()
                serialize_collection(source_dir)
                return time.perf_counter() - started_at

        if scenario == "serialize-json":
            output_json = root_dir / "serialized.json"
            with _serializer_paths(source_dir):
                started_at = time.perf_counter()
                serialize_collection_to_json(source_dir, output_json)
                return time.perf_counter() - started_at

        with _serializer_paths(source_dir):
            serialized_data = serialize_collection(source_dir)

        target_dir = root_dir / "target"
        target_dir.mkdir()
        _prepare_empty_collection(target_dir)

        if scenario == "deserialize-memory":
            with _serializer_paths(target_dir):
                started_at = time.perf_counter()
                deserialize_collection_data(serialized_data, overwrite=True)
                return time.perf_counter() - started_at

        if scenario == "deserialize-json":
            input_json = root_dir / "serialized.json"
            input_json.write_text(json.dumps(serialized_data), encoding="utf-8")
            with _serializer_paths(target_dir):
                started_at = time.perf_counter()
                deserialize_collection_from_json(input_json, overwrite=True)
                return time.perf_counter() - started_at

        raise ValueError(f"Unsupported scenario: {scenario}")


def _summary(samples: list[float]) -> dict[str, float | list[float]]:
    stdev = statistics.stdev(samples) if len(samples) > 1 else 0.0
    return {
        "samples_seconds": samples,
        "mean_seconds": statistics.mean(samples),
        "min_seconds": min(samples),
        "max_seconds": max(samples),
        "median_seconds": statistics.median(samples),
        "stdev_seconds": stdev,
    }


def _format_ms(seconds: float) -> str:
    return f"{seconds * 1000:.2f}"


def _print_table(
    *,
    notes: int,
    decks: int,
    repeats: int,
    summaries: dict[str, dict[str, float | list[float]]],
) -> None:
    print(
        "Collection serializer benchmark "
        f"(notes={notes}, decks={decks}, repeats={repeats})"
    )
    print(
        f"{'scenario':<20} {'mean(ms)':>10} {'median(ms)':>11} "
        f"{'min(ms)':>9} {'max(ms)':>9} {'stdev(ms)':>10}"
    )
    for scenario in _SCENARIOS:
        if scenario not in summaries:
            continue
        row = summaries[scenario]
        print(
            f"{scenario:<20} "
            f"{_format_ms(row['mean_seconds']):>10} "
            f"{_format_ms(row['median_seconds']):>11} "
            f"{_format_ms(row['min_seconds']):>9} "
            f"{_format_ms(row['max_seconds']):>9} "
            f"{_format_ms(row['stdev_seconds']):>10}"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark AnkiOps collection serializer throughput."
    )
    parser.add_argument(
        "--notes",
        type=int,
        default=_DEFAULT_NOTES,
        help=f"Total notes generated across decks (default: {_DEFAULT_NOTES}).",
    )
    parser.add_argument(
        "--decks",
        type=int,
        default=_DEFAULT_DECKS,
        help=f"Number of generated decks (default: {_DEFAULT_DECKS}).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of repeated runs per scenario (default: 3).",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        choices=_SCENARIOS,
        help=(
            "Benchmark scenario to run (repeatable). "
            "Defaults to all scenarios."
        ),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional output path for JSON benchmark results.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.repeats <= 0:
        raise ValueError("--repeats must be greater than zero")

    scenarios = args.scenario or list(_SCENARIOS)
    summaries: dict[str, dict[str, float | list[float]]] = {}

    for scenario in scenarios:
        samples = [
            _measure_once(scenario=scenario, notes=args.notes, decks=args.decks)
            for _ in range(args.repeats)
        ]
        summaries[scenario] = _summary(samples)

    _print_table(
        notes=args.notes,
        decks=args.decks,
        repeats=args.repeats,
        summaries=summaries,
    )

    if args.json_output is not None:
        payload: dict[str, Any] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "notes": args.notes,
            "decks": args.decks,
            "repeats": args.repeats,
            "scenarios": scenarios,
            "results": summaries,
        }
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote JSON results to {args.json_output}")


if __name__ == "__main__":
    main()
