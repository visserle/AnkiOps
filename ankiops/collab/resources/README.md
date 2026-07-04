# {{DECK_NAME}}

`{{DECK_NAME}}` is a shared Anki deck with a Git history, managed through [AnkiOps](https://github.com/visserle/AnkiOps). AnkiOps is a bidirectional bridge between Anki and your filesystem. In this collaborative repository, cards are written in Markdown, changes are reviewed through pull requests, and the accepted version is kept in sync with Anki.

> [!NOTE]
> AnkiOps collaboration is experimental. Check the [AnkiOps documentation](https://github.com/visserle/AnkiOps#how-does-collaboration-work-experimental) before upgrading an existing collection or changing the repository layout.

## Subscribe to this deck

Follow the [AnkiOps installation and initialization guide](https://github.com/visserle/AnkiOps#how-to-get-started) first. AnkiOps collab commands further require the [GitHub CLI](https://cli.github.com/) with an authenticated GitHub account:

```bash
gh auth login
```

Run these commands from the AnkiOps collection root:

```bash
ankiops collab subscribe {{REPOSITORY}}
ankiops fa
```

`collab subscribe` clones this repository into `collab/{{REPOSITORY}}/` as an independent Git repository. `ankiops fa` syncs its deck files, media, and note types to Anki. Your AnkiOps collection can hold private decks and other shared repositories alongside it. Open Anki to confirm that `{{DECK_NAME}}` and its subdecks appear.

## Keep the deck current

Run these commands from the AnkiOps collection root:

```bash
ankiops collab status {{REPOSITORY}}
ankiops collab update {{REPOSITORY}}
ankiops fa
```

`collab status` reports available GitHub changes and local work. `collab update` brings GitHub changes into the local Markdown source. Review them, then run `ankiops fa` to apply them to Anki.

If an update overlaps with a local edit, AnkiOps leaves the subscribed repository unchanged and reports the location of preserved base, local, and upstream copies. Resolve the marked Markdown file there, remove its conflict markers, and run the reported update command again.

## Contribute to this deck

Choose either editing workflow:

- Edit the Markdown files under `collab/{{REPOSITORY}}/`.
- Edit cards in Anki, then run `ankiops af` to write those changes back to the deck files.

Deck files follow the [AnkiOps Markdown syntax](https://github.com/visserle/AnkiOps#markdown-files). Keep the `note_key` and `note_type` comments unchanged; you may edit fields and tags.

Review and submit your changes from the AnkiOps collection root:

```bash
ankiops collab status {{REPOSITORY}}
ankiops collab submit {{REPOSITORY}} --message "Clarify the explanation of spaced repetition"
```

`collab submit` commits changes from this shared repository and opens a pull request. It excludes private decks and changes from other subscribed repositories. If your GitHub account lacks write access, AnkiOps creates or reuses a fork for the submission.

Use an issue for questions about the deck's scope or content. Use a pull request for a concrete card, media, or note-type change.

## Maintain this deck as a publisher

<!--
Publisher checklist:
- Add a LICENSE file before accepting contributions.
- Replace the notice in the License section below.
- Remove this comment.
-->

Review incoming changes through GitHub pull requests. After merging a pull request, update your local subscription before making more edits:

```bash
ankiops collab update {{REPOSITORY}}
ankiops fa
```

Use the same `status` and `submit` workflow for maintainer-authored changes.

## Command Reference

Global options:

- `--debug`: Enable debug logging.
- `--version`: Show the installed version.
- `--help`: Show help.

Collab commands:

- `ankiops init [--tutorial]`: Initialize the current directory as an AnkiOps collection.
- `ankiops anki-to-files` or `ankiops af`: Sync Anki changes into local files.
- `ankiops files-to-anki` or `ankiops fa`: Sync local file changes into Anki.
- `ankiops collab publish <deck> <owner>/<repo>`: Publish a deck to GitHub.
- `ankiops collab subscribe <owner>/<repo>`: Subscribe to a deck on GitHub.
- `ankiops collab status [owner/repo]`: Show the status of a subscribed deck.
- `ankiops collab update [owner/repo]`: Update a subscribed deck with upstream changes.
- `ankiops collab submit <owner>/<repo> [-m|--message text]`: Submit local changes to a subscribed deck as a PR.

See the [AnkiOps README](https://github.com/visserle/AnkiOps) for further details. 