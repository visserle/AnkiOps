from anki_addon.highlights import add_highlight_shortcut


class _FakeWebView:
    def __init__(self):
        self.scripts = []

    def eval(self, script):
        self.scripts.append(script)


class _FakeEditor:
    def __init__(self):
        self.web = _FakeWebView()


def test_add_highlight_shortcut_marks_selected_editor_html():
    shortcuts = []
    editor = _FakeEditor()

    add_highlight_shortcut(shortcuts, editor)

    sequence, callback = shortcuts[0]
    assert sequence == "Ctrl+H"

    callback()

    assert editor.web.scripts == [
        """
const activeElement = document.activeElement;
const shadowRoot = activeElement && activeElement.shadowRoot;
const selection = shadowRoot && shadowRoot.getSelection();
if (!selection || selection.rangeCount === 0 || selection.isCollapsed) {
    return;
}

const selected = document.createElement("div");
for (let index = 0; index < selection.rangeCount; index++) {
    selected.append(selection.getRangeAt(index).cloneContents());
}
document.execCommand("insertHTML", false, `<mark>${selected.innerHTML}</mark>`);
globalThis.saveNow();
"""
    ]
