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

`collab status` reports available GitHub changes and local work. `collab update` brings this repository's GitHub changes into its local Markdown files. Review them, then run `ankiops fa` to sync the whole collection to Anki.

If an update overlaps with a local edit, AnkiOps leaves the subscribed repository unchanged and reports the location of preserved base, local, and upstream copies. Resolve the marked Markdown file there, remove its conflict markers, and run the reported update command again.

## Contribute to this deck

Edit your subscribednotes with either workflow:

- Edit the Markdown files under `collab/{{REPOSITORY}}/`.
- Edit cards in Anki, then run `ankiops fa` to write those changes back to the deck files.

Review and submit your changes from the AnkiOps collection root:

```bash
ankiops collab status {{REPOSITORY}}
ankiops collab submit {{REPOSITORY}} --message "Clarify the explanation of spaced repetition"
```

`collab submit` commits changes from this shared repository and opens a pull request. It excludes private decks and changes from other subscribed repositories. If your GitHub account lacks write access, AnkiOps creates or reuses a fork for the submission.

Use an issue for questions about the deck's scope or content. Use a pull request for a concrete card, media, or note-type change.

## Maintain this deck as a publisher

Before inviting subscribers, replace the opening paragraph with information specific to this deck:

- Describe its topics, intended audience, and prerequisite knowledge.
- Name its sources or credits and explain which issues and contributions you accept.

Remove this checklist after updating the paragraph.

Review incoming changes through GitHub pull requests. After merging a pull request, update your local subscription before making more edits:

```bash
ankiops collab update {{REPOSITORY}}
ankiops fa
```

Use the same `status` and `submit` workflow for maintainer-authored changes.

See the [AnkiOps README](https://github.com/visserle/AnkiOps) for the full command reference and synchronization details.
