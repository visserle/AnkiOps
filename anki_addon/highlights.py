"""Editor highlighting shortcut for the AnkiOps add-on."""

_HIGHLIGHT_SELECTION_SCRIPT = """
(function () {
    const activeElement = document.activeElement;
    const shadowRoot = activeElement && activeElement.shadowRoot;
    const selection = shadowRoot && shadowRoot.getSelection();
    if (!selection || selection.rangeCount === 0 || selection.isCollapsed) {
        return;
    }

    const range = selection.getRangeAt(0);

    const closestMark = (node) => {
        const el = node.nodeType === Node.TEXT_NODE ? node.parentElement : node;
        return el ? el.closest("mark") : null;
    };

    const startMark = closestMark(range.startContainer);
    const endMark = closestMark(range.endContainer);

    if (startMark && startMark === endMark) {
        // toggle off: selection sits entirely inside one existing <mark> -> unwrap it
        const parent = startMark.parentNode;
        const movedNodes = [];
        while (startMark.firstChild) {
            const child = startMark.firstChild;
            parent.insertBefore(child, startMark);
            movedNodes.push(child);
        }
        parent.removeChild(startMark);

        selection.removeAllRanges();
        if (movedNodes.length > 0) {
            const newRange = document.createRange();
            newRange.setStartBefore(movedNodes[0]);
            newRange.setEndAfter(movedNodes[movedNodes.length - 1]);
            selection.addRange(newRange);
        }
    } else {
        // toggle on: wrap the selection in a new <mark>
        const mark = document.createElement("mark");
        mark.appendChild(range.extractContents());
        range.insertNode(mark);

        selection.removeAllRanges();
        const newRange = document.createRange();
        newRange.selectNodeContents(mark);
        selection.addRange(newRange);
    }

    const editable = shadowRoot.activeElement || activeElement;
    editable.dispatchEvent(new Event("input", { bubbles: true, composed: true }));
})();
"""


def add_highlight_shortcut(shortcuts: list, editor) -> None:
    """Register Ctrl+J to toggle a <mark> around the selected editor text."""
    shortcuts.append(("Ctrl+J", lambda: editor.web.eval(_HIGHLIGHT_SELECTION_SCRIPT)))
