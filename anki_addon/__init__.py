"""AnkiOps Anki add-on.

Adds toolbar commands and hosts the local AnkiOpsConnect.
"""

from __future__ import annotations

from aqt import gui_hooks, mw
from aqt.toolbar import Toolbar

from .ankiops_connect_host import AnkiOpsConnectHost
from .command_runner import AnkiOpsCommandRunner, add_toolbar_links

_command_runner = AnkiOpsCommandRunner(mw=mw, addon_module_name=__name__)
_ankiops_connect_host = AnkiOpsConnectHost(
    get_collection=lambda: mw.col,
    run_on_main=mw.taskman.run_on_main,
)


def _on_toolbar_init(links: list[str], toolbar: Toolbar) -> None:
    add_toolbar_links(links, toolbar, _command_runner)


def _start_ankiops_connect() -> None:
    _ankiops_connect_host.start()


gui_hooks.top_toolbar_did_init_links.append(_on_toolbar_init)
if hasattr(gui_hooks, "profile_did_open"):
    gui_hooks.profile_did_open.append(_start_ankiops_connect)
else:
    _start_ankiops_connect()
