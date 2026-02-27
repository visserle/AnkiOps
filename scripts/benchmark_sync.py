"""Benchmark import/export sync throughput with deterministic local fixtures."""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

from ankiops.anki import AnkiAdapter
from ankiops.config import get_note_types_dir
from ankiops.db import SQLiteDbAdapter
from ankiops.export_notes import export_collection
from ankiops.fs import FileSystemAdapter
from ankiops.import_notes import import_collection

_SCENARIOS = ("export-cold", "export-warm", "import-warm")
_DEFAULT_DECK = "PerfDeck"


def _load_mock_anki_class():
    script_dir = Path(__file__).resolve().parent
    fake_anki_path = script_dir.parent / "tests" / "support" / "fake_anki.py"
    spec = importlib.util.spec_from_file_location("ankiops_fake_anki", fake_anki_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load fake Anki module from {fake_anki_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MockAnki


MockAnki = _load_mock_anki_class()


def _seed_mock_anki(mock_anki: MockAnki, *, deck_name: str, notes: int) -> None:
    if deck_name not in mock_anki.decks:
        mock_anki.invoke("createDeck", deck=deck_name)

    for idx in range(notes):
        mock_anki.invoke(
            "addNote",
            note={
                "deckName": deck_name,
                "modelName": "AnkiOpsQA",
                "fields": {
                    "Question": f"Q{idx}",
                    "Answer": f"A{idx}",
                    "AnkiOps Key": f"perf-key-{idx}",
                },
            },
        )


def _new_ports(collection_dir: Path) -> tuple[AnkiAdapter, FileSystemAdapter, SQLiteDbAdapter]:
    anki = AnkiAdapter()
    fs = FileSystemAdapter()
    fs.set_configs(fs.load_note_type_configs(get_note_types_dir()))
    db = SQLiteDbAdapter.load(collection_dir)
    return anki, fs, db


def _measure_once(*, scenario: str, notes: int, deck_name: str) -> float:
    mock_anki = MockAnki()
    with tempfile.TemporaryDirectory() as temp_dir:
        collection_dir = Path(temp_dir)
        anki, fs, db = _new_ports(collection_dir)
        try:
            with (
                patch("ankiops.anki_client.invoke", side_effect=mock_anki.invoke),
                patch("ankiops.anki.invoke", side_effect=mock_anki.invoke),
            ):
                _seed_mock_anki(mock_anki, deck_name=deck_name, notes=notes)

                if scenario == "export-cold":
                    started_at = time.perf_counter()
                    export_collection(
                        anki_port=anki,
                        fs_port=fs,
                        db_port=db,
                        collection_dir=collection_dir,
                        note_types_dir=get_note_types_dir(),
                    )
                    return time.perf_counter() - started_at

                export_collection(
                    anki_port=anki,
                    fs_port=fs,
                    db_port=db,
                    collection_dir=collection_dir,
                    note_types_dir=get_note_types_dir(),
                )

                started_at = time.perf_counter()
                if scenario == "export-warm":
                    export_collection(
                        anki_port=anki,
                        fs_port=fs,
                        db_port=db,
                        collection_dir=collection_dir,
                        note_types_dir=get_note_types_dir(),
                    )
                elif scenario == "import-warm":
                    import_collection(
                        anki_port=anki,
                        fs_port=fs,
                        db_port=db,
                        collection_dir=collection_dir,
                        note_types_dir=get_note_types_dir(),
                    )
                else:
                    raise ValueError(f"Unsupported scenario: {scenario}")
                return time.perf_counter() - started_at
        finally:
            db.close()


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
    repeats: int,
    summaries: dict[str, dict[str, float | list[float]]],
) -> None:
    print(f"Sync benchmark (notes={notes}, repeats={repeats})")
    print(
        f"{'scenario':<12} {'mean(ms)':>10} {'median(ms)':>11} "
        f"{'min(ms)':>9} {'max(ms)':>9} {'stdev(ms)':>10}"
    )
    for scenario in _SCENARIOS:
        if scenario not in summaries:
            continue
        row = summaries[scenario]
        print(
            f"{scenario:<12} "
            f"{_format_ms(row['mean_seconds']):>10} "
            f"{_format_ms(row['median_seconds']):>11} "
            f"{_format_ms(row['min_seconds']):>9} "
            f"{_format_ms(row['max_seconds']):>9} "
            f"{_format_ms(row['stdev_seconds']):>10}"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark AnkiOps import/export sync throughput."
    )
    parser.add_argument(
        "--notes",
        type=int,
        default=5000,
        help="Number of notes seeded into the mock deck (default: 5000).",
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
    parser.add_argument(
        "--deck-name",
        default=_DEFAULT_DECK,
        help=f"Deck name used for generated notes (default: {_DEFAULT_DECK}).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    scenarios = args.scenario or list(_SCENARIOS)
    summaries: dict[str, dict[str, float | list[float]]] = {}

    for scenario in scenarios:
        samples = [
            _measure_once(scenario=scenario, notes=args.notes, deck_name=args.deck_name)
            for _ in range(args.repeats)
        ]
        summaries[scenario] = _summary(samples)

    _print_table(notes=args.notes, repeats=args.repeats, summaries=summaries)

    if args.json_output is not None:
        payload: dict[str, Any] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "notes": args.notes,
            "repeats": args.repeats,
            "scenarios": scenarios,
            "results": summaries,
        }
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote JSON results to {args.json_output}")


if __name__ == "__main__":
    main()
