"""AnkiOps Anki add-on entrypoint."""

from __future__ import annotations


def _setup() -> None:
    from aqt import gui_hooks, mw

    from .host import AnkiOpsConnectHost
    from .toolbar import AnkiOpsCommandRunner, add_toolbar_links

    toolbar_runner = AnkiOpsCommandRunner(mw=mw, addon_module_name=__name__)
    connect_host = AnkiOpsConnectHost(
        get_collection=lambda: mw.col,
        run_on_main=mw.taskman.run_on_main,
    )

    def on_toolbar_init(links, toolbar) -> None:
        add_toolbar_links(links, toolbar, toolbar_runner)

    def start_ankiops_connect() -> None:
        connect_host.start()

    gui_hooks.top_toolbar_did_init_links.append(on_toolbar_init)
    if hasattr(gui_hooks, "profile_did_open"):
        gui_hooks.profile_did_open.append(start_ankiops_connect)
    else:
        start_ankiops_connect()


try:
    _setup()
except ModuleNotFoundError as error:
    if error.name != "aqt":
        raise
