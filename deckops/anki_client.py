"""Shared AnkiConnect client and state used by both import/export modules."""

import logging
from dataclasses import dataclass
from typing import Any

import requests
from ankiops.config import ANKI_CONNECT_URL, SUPPORTED_NOTE_TYPES

logger = logging.getLogger(__name__)


def invoke(action: str, **params) -> Any:
    """Send a request to AnkiConnect and return the result.

    Raises an Exception when AnkiConnect returns an error.
    """
    response = requests.post(
        ANKI_CONNECT_URL,
        json={"action": action, "version": 6, "params": params},
        timeout=10,
    )
    result = response.json()
    if result.get("error"):
        raise Exception(f"AnkiConnect error: {result['error']}")
    return result["result"]


@dataclass
class AnkiState:
    """All Anki-side data, fetched once.

    Built by ``AnkiState.fetch()`` with 3-4 API calls:
      1. deckNamesAndIds
      2. findCards  (all AnkiOps cards)
      3. cardsInfo  (details for found cards)
      4. notesInfo  (details for discovered note IDs)

    Notes and cards are stored as raw AnkiConnect dicts.
    """

    deck_names_and_ids: dict[str, int]
    id_to_deck_name: dict[int, str]
    notes: dict[int, dict]  # note_id -> raw AnkiConnect note dict
    cards: dict[int, dict]  # card_id -> raw AnkiConnect card dict
    deck_note_ids: dict[str, set[int]]  # deck_name -> {note_id, ...}

    @staticmethod
    def fetch() -> "AnkiState":
        deck_names_and_ids = invoke("deckNamesAndIds")
        id_to_deck_name = {v: k for k, v in deck_names_and_ids.items()}

        query = " OR ".join(f"note:{nt}" for nt in SUPPORTED_NOTE_TYPES)
        all_card_ids = invoke("findCards", query=query)

        cards: dict[int, dict] = {}
        deck_note_ids: dict[str, set[int]] = {}
        all_note_ids: set[int] = set()

        if all_card_ids:
            for card in invoke("cardsInfo", cards=all_card_ids):
                cards[card["cardId"]] = card
                deck_note_ids.setdefault(card["deckName"], set()).add(card["note"])
                all_note_ids.add(card["note"])

        notes: dict[int, dict] = {}
        if all_note_ids:
            for note in invoke("notesInfo", notes=list(all_note_ids)):
                if not note:
                    continue
                model = note.get("modelName")
                if model and model not in SUPPORTED_NOTE_TYPES:
                    raise ValueError(
                        f"Safety check failed: Note {note['noteId']} has template "
                        f"'{model}' but expected a AnkiOps template. "
                        f"AnkiOps will never modify notes with non-AnkiOps templates."
                    )
                notes[note["noteId"]] = note

        return AnkiState(
            deck_names_and_ids=deck_names_and_ids,
            id_to_deck_name=id_to_deck_name,
            notes=notes,
            cards=cards,
            deck_note_ids=deck_note_ids,
        )
