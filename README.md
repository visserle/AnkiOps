# DeckOps

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![PyPI version](https://badge.fury.io/py/deckops.svg)](https://badge.fury.io/py/deckops) 

**Anki decks ↔ Markdown files, in perfect sync**

Editing flashcards in Anki's UI is tedious when you could be using your favorite text editor, AI tools, and Git. Currently available Markdown → Anki tools only go one way, where edits in Anki don't sync back. 

**DeckOps** is a bidirectional Anki ↔ Markdown bridge. Each deck is a Markdown file. Changes sync bidirectionally, allowing work in either Anki or your text editor. This brings AI assistance, batch editing, and version control to your flashcard workflow:

<video src="https://github.com/user-attachments/assets/f0b12979-f41a-4da9-b7fb-8587ca48329a" controls width="100%">
  Your browser does not support the video tag. Please refer to showcase.mp4.
</video>

## Features

- Fully round-trip, bidirectional sync that handles cards identites, moves, deletions, drifts, and conflicts.
- Support for Base (Q&A) and Cloze notes using DeckOps templates
- Markdown support with nearly all features (including syntax-highlighted code blocks, supported on desktop and mobile)
- Built-in Git integration with autocommit for tracking all changes
- Image support via VS Code where images are directly copied into your Anki media folder (automatically set up)
- Simple CLI interface: after initialization, only two commands are needed for daily use

## Getting Started


1. **Install DeckOps via [pipx](https://github.com/pypa/pipx)**: Pipx will make DeckOps globally available in your terminal.
```bash
pipx install deckops
```
2. **Initialize DeckOps**: Make sure that Anki is running, with the [AnkiConnect add-on](https://ankiweb.net/shared/info/2055492159) enabled. Initialize DeckOps in any empty directory of your choosing. This is where your text-based decks will live. The additional tutorial flag creates a sample Markdown deck.
```bash
deckops init --tutorial
```
3. **Execute DeckOps**: Import the tutorial deck into Anki using:
```bash
deckops ma # markdown to anki (import)
```
4. **Keep everything in sync**: When editing your Markdown files, sync Markdown → Anki (and vice versa), as each sync makes one side match the other. After reviewing and editing your cards in Anki, you can sync Anki → Markdown using the following command:
```bash
deckops am # anki to markdown (export)
```

## FAQ

### How is this different from other Markdown or Obsidian tools?

Available tools are one-way importers: you write in Markdown or Obsidian and push to Anki, but edits in Anki don't sync back. DeckOps is fully bidirectional: you can edit in either Anki or Markdown and sync in both directions. Additionally, DeckOps uses a one-file-per-deck structure, making your collection easier to navigate and manage than approaches that use one file per card. Note that DeckOps requires custom note types (`DeckOpsQA` and `DeckOpsCloze`).

### Is it safe to use?

Yes, DeckOps will never modify cards with non-DeckOps templates. Your existing collection won't be affected and you can safely mix managed and unmanaged cards in the same deck. Further, DeckOps only syncs if the activated profiles matches the one it was initialized with. When orpahned DeckOps cards are detected, you will be prompted to confirm their deletion. Concerning your Markdown files, DeckOps automatically creates a Git commit of your collection folder before every sync, so you can always roll your files back if needed.

### How do I create new cards?

Create a new Markdown file in your initialized DeckOps folder. For the first import, the file name will act as the deck name. Subdecks are supported via two underscores `__` (Anki's `::` is not supported in the file system). Start by writing your cards in Markdown. For each card, you can decide whether to use the QA or cloze format. Cards must be separated by a new line, three dashes `---`, and another new line. You can add new cards anywhere in an existing file.

```markdown
Q: Question text here
A: Answer text here
E: Extra information (optional)
M: Content behind a "more" button (optional)

---

T: Text with {{c1::multiple}} {{c2::cloze deletions}}.
E: ![image with set width](im.png){width=700}

---

And so on…
```

### Which characters or symbols cannot be used?

Since cards are separated by horizontal lines (`---`), they cannot be used within the content fields of your cards. This includes all special Markdown characters that render these lines (`***`, `___`), and `<hr>`.

### How does it work?

On first import, DeckOps assigns IDs from Anki to each deck/note/card for tracking. They are represented by a single-line HTML tag (e.g., `<!-- card_id: 1770487991522 -->`) above a card in the Markdown. With the IDs in place, we can track what is new, changed, moved between decks, or deleted, and DeckOps will sync accordingly. Content is automatically converted between Anki's HTML format and Markdown during sync operations. Note that one DeckOps folder represents an entire Anki profile.

### Why do some cards have a `card_id` and others a `note_id`?

`DeckOpsQA` cards have a `card_id`, while `DeckOpsCloze` cards have a `note_id` HTML tag. This is because cloze notes can generate multiple cards. If you want to transform a Cloze note into a QA card in Markdown, make sure to change the prefix from `T:` to `Q:` & `A:` and delete the old `note_id`. DeckOps will assign a new `card_id` in the next import.


### What is the recommended workflow?

We recommend using VS Code. It has excellent AI integration, a great [add-on](https://marketplace.visualstudio.com/items?itemName=shd101wyy.markdown-preview-enhanced) for Markdown previews, and supports image pasting (which will be saved in your Anki media folder by default).

### How can I share my DeckOps deck with others?

You can either export your deck using Anki's native export feature (`.apkg` file) and share that, or directly share your Markdown files along with the `media/DeckOpsMedia` folder. Make sure to remove all ID tags from your Markdown files first, as they are profile-specific.

### How can I migrate my existing cards into DeckOps?

While it is doable, the migration can be tricky. If you convert your cards into DeckOps' note types, then all you need to do is export your cards from Anki to Markdown. In the first re-import, some formatting may be changed because the original HTML from Anki may not follow the CommonMark standard. If your note format does not work with the DeckOps format, you will have to adapt the code to your needs.

### How can I develop DeckOps locally?

Fork this repository and initialize the tutorial in your root folder (make sure Anki is running). This will create a folder called `collection` with the sample Markdown in it. Paths will adapt automatically to the development environment. You can run DeckOps locally using the main script.

```bash
git clone https://github.com/visserle/deckops.git
cd deckops
uv sync
uv run python -m main init --tutorial
uv run python -m main ma
```

### What commands and flags are available in the CLI?

**Global:**
- `--debug` - Enable debug logging
- `--help` - Show help message

**`init`:**
- `--no-auto-commit` - Disable automatic git commits
- `--tutorial` - Create tutorial markdown file

**`anki-to-markdown` / `am`:**
- `--deck`, `-d` - Export single deck by name
- `--keep-orphans` - Keep deck files/cards that no longer exist in Anki
- `--no-auto-commit`, `-n` - Skip automatic git commit

**`markdown-to-anki` / `ma`:**
- `--file`, `-f` - Import single file
- `--only-add-new` - Only add new cards, skip existing
- `--no-auto-commit`, `-n` - Skip automatic git commit

---

### How does DeckOps solve bidirectional sync? (Claude's answer)

DeckOps handles the core challenges of bidirectional synchronization between markdown and Anki:

#### 1. Identity

**Solution**: Embed immutable IDs directly in markdown as HTML comments

- **Deck Identity**: `<!-- deck_id: 1234567890 -->` on first line of file
- **Card Identity** (QA cards): `<!-- card_id: 1770487991522 -->` before each card
- **Note Identity** (Cloze): `<!-- note_id: 1770487991521 -->` before each note

IDs are Anki's native IDs (timestamps in milliseconds), written to Markdown on first sync and persisting across all future syncs, enabling bidirectional linking.


#### 2. Moves (Between Decks)

**Solution**: Move detection + automatic deck correction

**Import (Markdown → Anki)**:
When you move a card between markdown files:
1. Cut card from `DeckA.md` (keeping its ID)
2. Paste into `DeckB.md`
3. Import detects card in wrong deck → **auto-moves to DeckB**
4. **Review history preserved** 


**Export (Anki → Markdown)**:
When you move a card between decks in Anki:
1. Export detects card disappeared from DeckA, appeared in DeckB
2. Reports as move (not deletion + creation)
3. Card appears in correct Markdown file

Note: Deck renaming is only possible via export (Anki → Markdown). While import (Markdown → Anki) can be used to create new decks named after the file, renaming decks should always happen via export. Since the `deck_id` is not dependent on the file name, there is no conflict when the Markdown file name differs from a deck's name in Anki.

#### 3. Conflict Resolution

**Solution**: **Last sync direction wins** (no merging)

**Import (Markdown → Anki)**:
- Markdown content **overwrites** Anki content
- Updates existing cards with markdown content
- If fields match → skip (optimization)
- If fields differ → markdown wins

**Export (Anki → Markdown)**:
- Anki content **overwrites** Markdown content
- Existing blocks replaced with Anki's current state
- Deck renames reflected in file renames

This simple approach requires discipline: always sync in the same direction for a given edit session.

#### 4. Drift Detection & Recovery

**Solution**: "Stale card" detection with automatic re-creation

**What is drift?**
- Card exists in Markdown with `card_id: 123`
- But ID 123 no longer exists in Anki (manually deleted)

**How it's resolved (import)**:
1. Phase 1: Try to update card 123 → fails
2. Mark as "stale"
3. Phase 3: Re-create in Anki with new ID (e.g., 456)
4. Phase 4: Update Markdown: `<!-- card_id: 123 -->` → `<!-- card_id: 456 -->`

**Result**: Drift is automatically healed. Content preserved, but review history lost (new card).

#### 5. Deletions

**Solution**: Bidirectional orphan cleanup

**Markdown → Anki (Import)**:
- Cards in Anki deck but NOT in Markdown file → deleted from Anki
- Exception: Cards claimed by other files are moved, not deleted

**Anki → Markdown (Export)**:
- **Orphaned decks**: File has `deck_id` but deck doesn't exist → delete file
- **Orphaned cards**: Card has `card_id` but card doesn't exist → remove block

Deletions propagate in both directions to maintain consistency.

#### Summary

| Challenge | Solution | Preserves History? |
|-----------|----------|-------------------|
| **Identity** | Embed Anki IDs in Markdown |  Yes |
| **Moves** | Auto-move + global tracking |  Yes |
| **Conflicts** | Last sync wins (no merge) |  Yes |
| **Drift** | Stale card detection + re-creation | No |
| **Deletions** | Bidirectional orphan cleanup | N/A |

---

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/visserle)
