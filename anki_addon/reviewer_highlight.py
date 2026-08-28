"""Injects the Ctrl+J highlight-toggle shortcut into EFDRC's review-mode editor."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Optional

_aqt = import_module("aqt")
gui_hooks: Any = _aqt.gui_hooks
Reviewer: Any = import_module("aqt.reviewer").Reviewer
WebContent: Any = import_module("aqt.webview").WebContent

try:
    MultiCardPreviewer: Any = import_module("aqt.previewer").MultiCardPreviewer
except ImportError:  # pragma: no cover - not present on older Anki versions
    MultiCardPreviewer = None


_HIGHLIGHT_SHORTCUT_SCRIPT = """
<script>
(function () {
    function register() {
        if (
            typeof window.EFDRC === "undefined" ||
            typeof window.EFDRC.registerShortcut !== "function"
        ) {
            return false;
        }
        window.EFDRC.registerShortcut("Ctrl+J", function (event, elem) {
            event.preventDefault();

            var selection = window.getSelection();
            if (!selection || selection.rangeCount === 0 || selection.isCollapsed) {
                return;
            }

            var range = selection.getRangeAt(0);

            function closestMark(node) {
                var el = node.nodeType === Node.TEXT_NODE ? node.parentElement : node;
                return el ? el.closest("mark") : null;
            }

            var startMark = closestMark(range.startContainer);
            var endMark = closestMark(range.endContainer);

            if (startMark && startMark === endMark) {
                var parent = startMark.parentNode;
                var movedNodes = [];
                while (startMark.firstChild) {
                    var child = startMark.firstChild;
                    parent.insertBefore(child, startMark);
                    movedNodes.push(child);
                }
                parent.removeChild(startMark);

                selection.removeAllRanges();
                if (movedNodes.length > 0) {
                    var newRange = document.createRange();
                    newRange.setStartBefore(movedNodes[0]);
                    newRange.setEndAfter(movedNodes[movedNodes.length - 1]);
                    selection.addRange(newRange);
                }
            } else {
                var mark = document.createElement("mark");
                mark.appendChild(range.extractContents());
                range.insertNode(mark);

                selection.removeAllRanges();
                var newRange2 = document.createRange();
                newRange2.selectNodeContents(mark);
                selection.addRange(newRange2);
            }
        });
        return true;
    }

    if (!register()) {
        var attempts = 0;
        var timer = setInterval(function () {
            attempts += 1;
            if (register() || attempts > 40) {  // ~2s at 50ms
                clearInterval(timer);
            }
        }, 50);
    }
})();
</script>
"""


def _on_webview_will_set_content(web_content: Any, context: Optional[Any]) -> None:
    if isinstance(context, Reviewer):
        web_content.body += _HIGHLIGHT_SHORTCUT_SCRIPT
    elif MultiCardPreviewer is not None and isinstance(context, MultiCardPreviewer):
        web_content.body += _HIGHLIGHT_SHORTCUT_SCRIPT


def add_reviewer_highlight_shortcut() -> None:
    """Register the Ctrl+J highlight-toggle shortcut for EFDRC's review-mode editor."""
    gui_hooks.webview_will_set_content.append(_on_webview_will_set_content)
