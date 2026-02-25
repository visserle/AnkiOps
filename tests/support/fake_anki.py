"""Stateful fake AnkiConnect backend used in tests."""

from __future__ import annotations

from typing import Any

from ankiops.anki_client import AnkiConnectError


class MockAnki:
    """Stateful mock of AnkiConnect (unified action dispatcher)."""

    def __init__(self):
        self.decks = {"Default": 1}  # Name -> ID
        self.notes = {}  # ID -> Note dict
        self.cards = {}  # ID -> Card dict
        self.next_note_id = 100
        self.next_card_id = 1000
        self.next_deck_id = 10
        self.calls = []

    def invoke(self, action: str, **params) -> Any:
        self.calls.append((action, params))

        match action:
            case "deckNamesAndIds":
                return self.decks

            case "findCards":
                query = params.get("query", "")
                found_cards = []
                or_groups = query.split(" OR ")

                for cid, card in self.cards.items():
                    card_matches_any_group = False

                    for group in or_groups:
                        terms = group.strip().split()
                        if not terms:
                            continue

                        group_matches_all_terms = True
                        for term in terms:
                            if term.startswith("note:"):
                                model = term.split(":", 1)[1]
                                note = self.notes.get(card["note"])
                                if not note or note["modelName"] != model:
                                    group_matches_all_terms = False
                                    break
                            elif term.startswith("deck:"):
                                deck_name = term.split(":", 1)[1]
                                if card["deckName"] != deck_name:
                                    group_matches_all_terms = False
                                    break

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
                return [
                    self.notes[note_id] for note_id in note_ids if note_id in self.notes
                ]

            case "multi":
                actions = params.get("actions", [])
                results = []
                for act in actions:
                    res = self.invoke(act["action"], **act.get("params", {}))
                    results.append(res)
                return results

            case "modelNames":
                return [
                    "AnkiOpsQA",
                    "AnkiOpsReversed",
                    "AnkiOpsCloze",
                    "AnkiOpsInput",
                    "AnkiOpsChoice",
                ]

            case "modelFieldNames":
                model = params.get("modelName")
                if model == "AnkiOpsQA":
                    return [
                        "Question",
                        "Answer",
                        "Extra",
                        "More",
                        "Source",
                        "AI Notes",
                        "AnkiOps Key",
                    ]
                elif model == "AnkiOpsReversed":
                    return [
                        "Front",
                        "Back",
                        "Extra",
                        "More",
                        "Source",
                        "AI Notes",
                        "AnkiOps Key",
                    ]
                elif model == "AnkiOpsCloze":
                    return [
                        "Text",
                        "Extra",
                        "More",
                        "Source",
                        "AI Notes",
                        "AnkiOps Key",
                    ]
                elif model == "AnkiOpsInput":
                    return [
                        "Question",
                        "Input",
                        "Answer",
                        "Extra",
                        "More",
                        "Source",
                        "AI Notes",
                        "AnkiOps Key",
                    ]
                elif model == "AnkiOpsChoice":
                    return [
                        "Question",
                        "Choice 1",
                        "Choice 2",
                        "Choice 3",
                        "Choice 4",
                        "Choice 5",
                        "Answer",
                        "Extra",
                        "More",
                        "Source",
                        "AI Notes",
                        "AnkiOps Key",
                    ]
                return []

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

            case "addNotes":
                notes = params.get("notes", [])
                results = []
                for note_data in notes:
                    results.append(self.invoke("addNote", note=note_data))
                return results

            case "updateNoteFields":
                note_info = params["note"]
                note_id = note_info["id"]
                if note_id in self.notes:
                    for k, v in note_info["fields"].items():
                        self.notes[note_id]["fields"][k] = {"value": v}
                return None

            case "deleteNotes":
                note_ids = params["notes"]
                for note_id in note_ids:
                    if note_id in self.notes:
                        card_ids = self.notes[note_id]["cards"]
                        for cid in card_ids:
                            if cid in self.cards:
                                del self.cards[cid]
                        del self.notes[note_id]
                return None

            case "changeDeck":
                cards = params["cards"]
                deck = params["deck"]
                for cid in cards:
                    if cid in self.cards:
                        self.cards[cid]["deckName"] = deck
                return None

            case "findNotes":
                query = params.get("query", "")

                if "AnkiOps Key:" in query:
                    note_key = query.split(":")[1].strip('"')
                    found = []
                    for note_id, note in self.notes.items():
                        if (
                            note["fields"].get("AnkiOps Key", {}).get("value")
                            == note_key
                        ):
                            found.append(note_id)
                    return found

                or_groups = query.split(" OR ")
                found_notes = []
                for note_id, note in self.notes.items():
                    for group in or_groups:
                        terms = group.strip().split()
                        match_all = True
                        for term in terms:
                            if term.startswith("note:"):
                                model = term.split(":", 1)[1]
                                if note["modelName"] != model:
                                    match_all = False
                                    break
                            elif term.startswith("deck:"):
                                deck_match = False
                                target_deck = term.split(":", 1)[1]
                                for card in self.cards.values():
                                    if (
                                        card["note"] == note_id
                                        and card["deckName"] == target_deck
                                    ):
                                        deck_match = True
                                        break
                                if not deck_match:
                                    match_all = False
                                    break
                        if match_all:
                            found_notes.append(note_id)
                            break
                return found_notes

            case _:
                return None

    def add_note(self, deck_name: str, note_type: str, fields: dict):
        """Convenience helper for test setup."""
        if deck_name not in self.decks:
            self.invoke("createDeck", deck=deck_name)

        self.invoke(
            "addNote",
            note={"deckName": deck_name, "modelName": note_type, "fields": fields},
        )
