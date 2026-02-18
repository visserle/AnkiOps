Q: Welcome to AnkiOps!
A: Anki decks ↔ Markdown files, in perfect sync
E: Edit in your text editor, review in Anki.

![](sync_arrows.png)
M: Neat!

---

T: AnkiOps supports {{c1::cloze deletions}} and even {{c1::multiple}} {{c2::clozes}} in one card.

---

Q: What is this?
C1: A multiple choice question
C2: with
C3: automatically randomized answers.
A: 1,3
E: The order changes dynamically between each review.

---

Q: What Markdown features are supported?
A: **Bold text**, *italic text*, ==highlighted text==, ~~strikethrough~~, `inline code`,

> blockquotes,
> another one,

1. ordered lists,
   - even nested unordered items,

| Beautiful | Tables, |
| --- | --- |
| inline formula: | \(E = mc^2\), and |

\[\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi},\] and, of course, code blocks:

```python
def truth():
    print("AnkiOps is awesome")

```
E: *Everything renders beautifully on desktop and mobile.*

---

Q: How do I get started?
A: Run `ankiops ma` to import Markdown --> Anki

Run `ankiops am` to export Anki --> Markdown
E: Check the README for detailed documentation!