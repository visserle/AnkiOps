# LLM Tasks

With LLM tasks, you can run programmable, customizable tasks on your collection such as content review, grammar fixes, translations, and more. The tasks are for enhancing, rewriting, completing already existing notes. In contrast to edits by automated agents (i.e. Codex), tasks divide edits into separate requests to the API. This improves attention to detail significantly.

## Architecture of a task

LLM tasks read notes from your Markdown decks and write selected fields or tags back to those files. Each `.yaml` file (except `_models.yaml`) defines one task. The filename becomes the task name, so `fix-grammar.yaml` runs as `ankiops llm fix-grammar`.

```yaml
# fix-grammar.yaml
model: gpt-5.4-mini
system_prompt: !file _system_prompt.md
user_prompt: |
  Correct grammar, spelling, and punctuation in editable fields.
  Preserve meaning, Markdown structure, cloze syntax, code fences, math, and URLs.
  Do not add facts or change correctness.

fields:
  default_access: editable
  hidden:
    "*": ["AI Notes"]
    "AnkiOpsImageOcclusion": ["*"]
  read_only:
    "AnkiOpsChoice": ["Answer"]

tags: hidden

request:
  max_notes_per_request: 3
  reasoning: low
```

Apart from the prompts, the most important part of a task is the `fields` section. It controls which fields the model can read and write. There are three access levels:

1. `editable`: Can receive full replacement values.
2. `read_only`: Response cannot change them.
3. `hidden`: Never enter the request.

Further, field access can be set for specific note types or via `*` wildcards. For example, the `AnkiOpsChoice` note type has a read-only `Answer` field in the example above. The model can read it, but cannot change it. Always use the field names from the `note_type.yaml`, not the short labels from Markdown deck files (e.g., `Question` instead of `Q:`). Tags are also controlled by access levels and default to `hidden`.

Task files accept these top-level keys:

- `model`: An alias from `_models.yaml`
- `system_prompt` and `user_prompt`: Inline text or `!file path.md` (file paths stay within this `llm` folder)
- `fields`: Sets `default_access` and field rules for `editable`, `read_only`, or `hidden` access. Rules map note-type patterns to field-name patterns. Patterns use shell-style wildcards such as `*`.
- `tags`: Sets tag access to `editable`, `read_only`, or `hidden`. Tags default to `hidden` when you omit this key.
- `request`: Requires `max_notes_per_request`. It can also set `temperature` and `reasoning`. AnkiOps accepts `none`, `low`, `medium`, `high`, or `xhigh`, but each model supports a subset.

## Running tasks

AnkiOps depends on the OpenAI Responses API with advanced features such as structured model outputs. While AnkiOps is provider-agnostic and could work with other providers or local models (see Model registry below), it uses OpenAI's models by default.

1. Set your `OPENAI_API_KEY` in your shell environment. You can get one from [OpenAI](https://platform.openai.com/account/api-keys). For example, in Bash:

```bash
export OPENAI_API_KEY="your-api-key"
```

2. Run the status command to validate the model registry and every task:

```bash
ankiops llm
```

3. Plan the task to see which notes it will affect and how much it will approximately cost (assuming number of input tokens equals output tokens):

```bash
ankiops llm fix-grammar
```

4. Run the task after the plan looks right by adding `--run`:

```bash
ankiops llm fix-grammar --run
```

5. Review the all changes in the Git diff and sync your improved notes to Anki via `ankiops fa`.

## Additional options

There are additional options for running tasks from the command line. You can limit the task to a specific deck with `--deck`, override the model with `--model`, and inspect a completed or failed job with `--job`.

Limit tasks to one decks with `--deck`:

```bash
ankiops llm translate --deck "Deck1::Subdeck1"
ankiops llm translate --deck "Deck1::Subdeck1" --run
```

Override the task's model with `--model`:

```bash
ankiops llm review --model gpt-5.4 --run
```

Inspect a completed or failed job with:

```bash
ankiops llm --job latest
ankiops llm --job 12
```

## Model registry

`_models.yaml` contains the aliases available to tasks and `--model`. Each entry uses these fields:

- `model`: Required. The local alias used in task files and commands.
- `model_id`: Required. The model identifier sent to the API.
- `base_url`: Required. The API base URL without `/responses`.
- `api_key`: Required. A key or an environment-variable reference such as `$OPENAI_API_KEY`.
- `concurrency`: Optional. The maximum number of requests AnkiOps sends at once. It defaults to `8`. Lower `concurrency` if your provider reports rate-limit errors.
- `input_usd_per_mtok` and `output_usd_per_mtok`: Optional prices used for plan and job cost estimates.

A sample entry for `_models.yaml`:
```yaml
- model: gpt-5.4-mini
  model_id: gpt-5.4-mini
  base_url: https://api.openai.com/v1
  api_key: $OPENAI_API_KEY
  concurrency: 8
  input_usd_per_mtok: 0.75
  output_usd_per_mtok: 4.5
```