"""Shared fixtures for AnkiOps tests."""

from typing import Any
from unittest.mock import patch

import pytest

from ankiops.anki_client import AnkiConnectError


class MockAnki:
    """Stateful mock of AnkiConnect (Unified)."""

    def __init__(self):
        # State
        self.decks = {"Default": 1}  # Name -> ID
        self.notes = {}  # ID -> Note dict (Anki format)
        self.cards = {}  # ID -> Card dict
        self.next_note_id = 100
        self.next_card_id = 1000
        self.next_deck_id = 10

        # Operation log for assertions
        self.calls = []

    def invoke(self, action: str, **params) -> Any:
        self.calls.append((action, params))

        match action:
            case "deckNamesAndIds":
                return self.decks

            case "findCards":
                query = params.get("query", "")
                found_cards = []

                # Parse query: "part1 OR part2 OR part3"
                # A card matches if it satisfies ANY of the parts.
                # Each part consists of space-separated terms (AND condition).
                or_groups = query.split(" OR ")

                for cid, card in self.cards.items():
                    # Check if card matches ANY of the OR groups
                    card_matches_any_group = False

                    for group in or_groups:
                        # Inside a group, terms are AND-ed
                        # e.g. "note:Basic deck:Default" -> must match note type AND deck
                        terms = group.strip().split()
                        if not terms:
                            continue

                        group_matches_all_terms = True
                        for term in terms:
                            if term.startswith("note:"):
                                model = term.split(":", 1)[1]
                                # Note might not exist if deleted but card remains
                                # (shouldn't happen in valid state but good to be safe)
                                note = self.notes.get(card["note"])
                                if not note or note["modelName"] != model:
                                    group_matches_all_terms = False
                                    break
                            elif term.startswith("deck:"):
                                deck_name = term.split(":", 1)[1]
                                if card["deckName"] != deck_name:
                                    group_matches_all_terms = False
                                    break
                            # Ignore unknown terms for now or fail?
                            # Failing is safer for tests to catch unexpected queries.
                            # But for now let's assume valid queries.

                        if group_matches_all_terms:
                            card_matches_any_group = True
                            break

                    if card_matches_any_group:
                        found_cards.append(cid)

                return found_cards

            case "cardsInfo":
                card_ids = params.get("cards", [])
                return [self.cards[cid] for cid in card_ids if cid in self.cards]

            case "notesInfo":
                note_ids = params.get("notes", [])
                return [self.notes[nid] for nid in note_ids if nid in self.notes]

            case "multi":
                actions = params.get("actions", [])
                results = []
                for act in actions:
                    res = self.invoke(act["action"], **act.get("params", {}))
                    results.append(res)
                return results

            case "createDeck":
                name = params["deck"]
                new_id = self.next_deck_id
                self.next_deck_id += 1
                self.decks[name] = new_id
                return new_id

            case "addNote":
                note_data = params["note"]
                if "deckName" in note_data:
                    deck_name = note_data["deckName"]
                    if deck_name not in self.decks:
                        raise AnkiConnectError(f"deck was not found: {deck_name}")

                new_id = self.next_note_id
                self.next_note_id += 1

                # Create cards (mock 1 card per note)
                card_id = self.next_card_id
                self.next_card_id += 1

                self.notes[new_id] = {
                    "noteId": new_id,
                    "modelName": note_data["modelName"],
                    "fields": {k: {"value": v} for k, v in note_data["fields"].items()},
                    "cards": [card_id],
                }
                self.cards[card_id] = {
                    "cardId": card_id,
                    "note": new_id,
                    "deckName": note_data["deckName"],
                    "modelName": note_data["modelName"],
                }
                return new_id

            case "updateNoteFields":
                note_info = params["note"]
                nid = note_info["id"]
                if nid in self.notes:
                    for k, v in note_info["fields"].items():
                        self.notes[nid]["fields"][k] = {"value": v}
                return None

            case "deleteNotes":
                nids = params["notes"]
                for nid in nids:
                    if nid in self.notes:
                        # Remove associated cards
                        card_ids = self.notes[nid]["cards"]
                        for cid in card_ids:
                            if cid in self.cards:
                                del self.cards[cid]
                        del self.notes[nid]
                return None

            case "changeDeck":
                # cards, deck
                cards = params["cards"]
                deck = params["deck"]
                for cid in cards:
                    if cid in self.cards:
                        self.cards[cid]["deckName"] = deck
                return None

            case "findNotes":
                # "AnkiOps Key:xyz"
                query = params.get("query", "")
                if "AnkiOps Key:" in query:
                    key = query.split(":")[1].strip('"')
                    found = []
                    for nid, note in self.notes.items():
                        if note["fields"].get("AnkiOps Key", {}).get("value") == key:
                            found.append(nid)
                    return found
                return []

            case _:
                return None

    # -- Setup helpers --
    def add_note(self, deck_name: str, note_type: str, fields: dict):
        if deck_name not in self.decks:
            self.invoke("createDeck", deck=deck_name)

        self.invoke(
            "addNote",
            note={"deckName": deck_name, "modelName": note_type, "fields": fields},
        )


@pytest.fixture
def mock_anki():
    return MockAnki()


@pytest.fixture(autouse=True)
def mock_input():
    """Always answer 'y' to confirmation prompts."""
    with patch("builtins.input", return_value="y"):
        yield


@pytest.fixture
def run_ankiops(mock_anki):
    """Fixture to run ankiops with mocked invoke."""
    # We must patch where it's imported in all touched modules
    with (
        patch("ankiops.anki_client.invoke", side_effect=mock_anki.invoke),
        patch("ankiops.models.invoke", side_effect=mock_anki.invoke),
        patch("ankiops.markdown_to_anki.invoke", side_effect=mock_anki.invoke),
        patch("ankiops.note_types.invoke", side_effect=mock_anki.invoke),
    ):
        yield
