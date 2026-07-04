from __future__ import annotations

from anki_addon.actions import dispatch_action
from tests.unit.addon.fakes import _FakeCollection


def test_dispatch_action_reads_model_state():
    col = _FakeCollection()
    col.old_model["flds"][0]["description"] = "Prompt"
    col.old_model["flds"][0]["size"] = 14

    assert dispatch_action(col, "modelNames", {}) == [
        "AnkiOpsQA",
        "collab/owner/repo/AnkiOpsQA",
    ]
    assert dispatch_action(
        col,
        "modelFieldNames",
        {"modelName": "AnkiOpsQA"},
    ) == ["Question", "Answer", "AnkiOps Key"]
    assert dispatch_action(
        col,
        "modelStyling",
        {"modelName": "AnkiOpsQA"},
    ) == {"name": "AnkiOpsQA", "css": ".card { color: black; }"}
    assert dispatch_action(
        col,
        "modelTemplates",
        {"modelName": "AnkiOpsQA"},
    ) == {"Card 1": {"Front": "{{Question}}", "Back": "{{Answer}}"}}
    assert dispatch_action(
        col,
        "modelFieldDescriptions",
        {"modelName": "AnkiOpsQA"},
    ) == ["Prompt", "", ""]
    assert dispatch_action(
        col,
        "modelFieldFonts",
        {"modelName": "AnkiOpsQA"},
    ) == {
        "Question": {"font": "Arial", "size": 14},
        "Answer": {"font": "Arial", "size": 20},
        "AnkiOps Key": {"font": "Arial", "size": 20},
    }


def test_dispatch_action_creates_and_updates_model_state():
    col = _FakeCollection()

    assert (
        dispatch_action(
            col,
            "createModel",
            {
                "modelName": "AnkiOpsNew",
                "inOrderFields": ["Question", "Answer"],
                "css": ".card { color: green; }",
                "isCloze": False,
                "cardTemplates": [
                    {"Name": "Card 1", "Front": "{{Question}}", "Back": "{{Answer}}"}
                ],
            },
        )
        is None
    )
    assert dispatch_action(
        col,
        "modelFieldNames",
        {"modelName": "AnkiOpsNew"},
    ) == ["Question", "Answer"]

    assert (
        dispatch_action(
            col,
            "modelFieldAdd",
            {"modelName": "AnkiOpsNew", "fieldName": "Extra"},
        )
        is None
    )
    assert (
        dispatch_action(
            col,
            "modelFieldReposition",
            {"modelName": "AnkiOpsNew", "fieldName": "Extra", "index": 1},
        )
        is None
    )
    assert (
        dispatch_action(
            col,
            "modelFieldSetDescription",
            {"modelName": "AnkiOpsNew", "fieldName": "Extra", "description": "Details"},
        )
        is None
    )
    assert (
        dispatch_action(
            col,
            "modelFieldSetFontSize",
            {"modelName": "AnkiOpsNew", "fieldName": "Extra", "fontSize": 15},
        )
        is None
    )
    assert (
        dispatch_action(
            col,
            "updateModelStyling",
            {"model": {"name": "AnkiOpsNew", "css": ".card { color: blue; }"}},
        )
        is None
    )
    assert (
        dispatch_action(
            col,
            "modelTemplateRename",
            {
                "modelName": "AnkiOpsNew",
                "oldTemplateName": "Card 1",
                "newTemplateName": "Review",
            },
        )
        is None
    )
    assert (
        dispatch_action(
            col,
            "modelTemplateAdd",
            {
                "modelName": "AnkiOpsNew",
                "template": {
                    "Name": "Extra",
                    "Front": "{{Extra}}",
                    "Back": "{{Answer}}",
                },
            },
        )
        is None
    )
    assert (
        dispatch_action(
            col,
            "updateModelTemplates",
            {
                "model": {
                    "name": "AnkiOpsNew",
                    "templates": {
                        "Review": {"Front": "{{Question}}?", "Back": "{{Answer}}!"},
                        "Extra": {"Front": "{{Extra}}?", "Back": "{{Answer}}!"},
                    },
                }
            },
        )
        is None
    )

    assert dispatch_action(
        col,
        "modelFieldNames",
        {"modelName": "AnkiOpsNew"},
    ) == ["Question", "Extra", "Answer"]
    assert dispatch_action(
        col,
        "modelFieldDescriptions",
        {"modelName": "AnkiOpsNew"},
    ) == ["", "Details", ""]
    assert (
        dispatch_action(
            col,
            "modelFieldFonts",
            {"modelName": "AnkiOpsNew"},
        )["Extra"]["size"]
        == 15
    )
    assert (
        dispatch_action(
            col,
            "modelStyling",
            {"modelName": "AnkiOpsNew"},
        )["css"]
        == ".card { color: blue; }"
    )
    assert dispatch_action(
        col,
        "modelTemplates",
        {"modelName": "AnkiOpsNew"},
    ) == {
        "Review": {"Front": "{{Question}}?", "Back": "{{Answer}}!"},
        "Extra": {"Front": "{{Extra}}?", "Back": "{{Answer}}!"},
    }
