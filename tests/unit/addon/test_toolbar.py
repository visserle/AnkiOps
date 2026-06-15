from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_toolbar_module(monkeypatch):
    aqt = ModuleType("aqt")
    qt = ModuleType("aqt.qt")
    utils = ModuleType("aqt.utils")

    class _DummyDialog:
        def __init__(self, *args, **kwargs):
            pass

        def setWindowTitle(self, title):
            pass

        def resize(self, width, height):
            pass

        def exec(self):
            pass

    class _DummyButtonBox:
        class StandardButton:
            Ok = object()

        def __init__(self, *args, **kwargs):
            self.accepted = self

        def connect(self, callback):
            pass

    class _DummyFontDatabase:
        class SystemFont:
            FixedFont = object()

        @staticmethod
        def systemFont(font):
            return object()

    class _DummyPlainTextEdit:
        class LineWrapMode:
            NoWrap = object()

        def __init__(self, *args, **kwargs):
            pass

        def setReadOnly(self, value):
            pass

        def setPlainText(self, value):
            pass

        def setFont(self, value):
            pass

        def setLineWrapMode(self, value):
            pass

    class _DummyLayout:
        def __init__(self, *args, **kwargs):
            pass

        def addWidget(self, widget):
            pass

    qt.QDialog = _DummyDialog
    qt.QDialogButtonBox = _DummyButtonBox
    qt.QFontDatabase = _DummyFontDatabase
    qt.QPlainTextEdit = _DummyPlainTextEdit
    qt.QVBoxLayout = _DummyLayout
    utils.showCritical = lambda *args, **kwargs: None
    utils.tooltip = lambda *args, **kwargs: None

    monkeypatch.setitem(sys.modules, "aqt", aqt)
    monkeypatch.setitem(sys.modules, "aqt.qt", qt)
    monkeypatch.setitem(sys.modules, "aqt.utils", utils)
    module_path = Path(__file__).parents[3] / "anki_addon" / "toolbar.py"
    spec = importlib.util.spec_from_file_location(
        "anki_addon_toolbar_test", module_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeToolbar:
    def __init__(self):
        self.handlers = {}

    def create_link(self, *, cmd, label, func, tip=None, id=None):
        self.handlers[cmd] = func
        title_attr = f'title="{tip}"' if tip else ""
        id_attr = f'id="{id}"' if id else ""
        return (
            f'<a class=hitem tabindex="-1" aria-label="{label}" '
            f"{title_attr} {id_attr} href=# onclick=\"return pycmd('{cmd}')\">"
            f"{label}</a>"
        )


class _FakeRunner:
    def __init__(self):
        self.commands = []

    def run(self, command):
        self.commands.append(command)


def test_add_toolbar_links_groups_af_and_fa_with_a_tight_separator(monkeypatch):
    toolbar_module = _load_toolbar_module(monkeypatch)
    links = ["Decks", "Add", "Browse", "Stats", "Sync"]
    toolbar = _FakeToolbar()
    runner = _FakeRunner()

    toolbar_module.add_toolbar_links(links, toolbar, runner)

    compact_link = next(link for link in links if "ankiops-toolbar-pair" in link)
    assert links.count(compact_link) == 1
    assert compact_link.count("class=hitem") == 2
    assert ">af</a><span" in compact_link
    assert ">|</span><a " in compact_link
    assert ">fa</a>" in compact_link
    assert "margin-left:0" in compact_link
    assert "padding-right:1px" in compact_link

    toolbar.handlers["ankiops_anki_to_files"]()
    toolbar.handlers["ankiops_files_to_anki"]()
    assert runner.commands == ["af", "fa"]
