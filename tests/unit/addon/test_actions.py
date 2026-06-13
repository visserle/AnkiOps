from __future__ import annotations

import pytest

from anki_addon.actions import AnkiOpsConnectActionError, dispatch_action


def test_dispatch_action_routes_version():
    assert dispatch_action(None, "version", {}) == 1


def test_dispatch_action_multi_collects_results_and_errors():
    results = dispatch_action(
        None,
        "multi",
        {
            "actions": [
                {"action": "version", "params": {}},
                {"action": "missing", "params": {}},
            ]
        },
    )

    assert results == [1, "Unknown AnkiOpsConnect action: missing"]


def test_dispatch_action_blocks_unknown_action():
    with pytest.raises(
        AnkiOpsConnectActionError,
        match="Unknown AnkiOpsConnect action",
    ):
        dispatch_action(None, "missing", {})
