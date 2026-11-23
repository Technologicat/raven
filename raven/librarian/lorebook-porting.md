# Porting the lorebook to another frontend

The lorebook gives an LLM the definitions of the user's own terms when a term comes up in the conversation: a
folder of Markdown files, each entry a section whose header is the term. Raven-librarian is one frontend for it.
This says what it takes to give the same lorebook to another — Claude Code, Open WebUI, LM Studio, pi — so that
one folder of definitions serves them all.

A port is three things:

- **The core**, `lorebookcore.py`: parsing, matching, choosing what fires, answering a lookup.
- **The conformance cases**, `tests/lorebook_conformance.json`, which say what the core must do.
- **The host's half**, which only the port can write: where definitions go, what was already given, the
  lookup tool, the folder.

## The core

**`lorebookcore.py` is one file, Python 3.11 or later, standard library only.** Raven's tests hold it to that.

- **A Python host copies it as it is.**
- **Any other host translates it**, and the conformance cases are how the translation is checked.

What it offers, all of it stateless:

- `parse(text, name=...)` splits one file into `Entry` records; `load(directory)` reads a folder of them, the
  folder's `README.md` excepted.
- `find_matches(entries, text, expansions=...)` says which entries' terms appear in a text.
- `select(entries, text, count_tokens=..., budget=..., max_depth=..., already_fired=..., expansions=...)` is the
  whole choice of what fires: the terms in the text, then the entries their links lead to, under a budget.
- `answer(entries, term, expansions=...)` is what the lookup tool replies with; `tool_spec()` is the tool as the
  model is shown it, an OpenAI-style function specification named `TOOL_NAME`.
- `terms(entries)` lists every term with its words; `clean_expansion` is the guard on generated forms.

Each function's docstring is the reference for its arguments.

## The conformance cases

**`tests/lorebook_conformance.json` is the spec that cannot drift from the code**: Raven's own tests run it
(`tests/test_lorebookcore.py`), and a port runs it too. The runner is short, and is the reference for what each
field means.

- **`glossaries`**: small lorebooks, each a list of lines, referred to by name from the cases. A case may also
  carry its glossary's lines itself.
- **One section per function**, each a list of cases with a `name`:
  - `parse`: the titles a glossary splits into, and facts about chosen entries.
  - `slugify`: title and anchor pairs.
  - `load`: files to write into an empty folder, and the entries and problems that come back.
  - `find_matches`: a text, the entries it fires, and the wording found.
  - `clean_expansion`: what the guard keeps of a generator's suggestions.
  - `select`: what fires, at which depth, by what wording, and through which link.
  - `lookup`: the entry found, whether exactly, the near misses offered, the entries mentioning the term, and
    the answer text.
- **Expansions are given raw**, as a generator suggests them, so a port's `clean_expansion` is tested along with
  its matcher.
- **`select` counts tokens as characters**: an entry costs `len(format_entry(entry))`, so a port needs no
  tokenizer to run the cases.

## What the host supplies

### Where the definitions go

**Ahead of the user's message, as material the model reads before answering it.** In Raven they are stored in the
chat as a round of `lorebook_lookup` calls nobody made, each answered by an entry, so that the model reads them as
looked up before the question came.

- **That shape was measured, and it matters most on local models.** Raven's measurement of injected material
  (`investigations/context-injects/`) found it best placed *before* the user's question, in an honest tool role
  with the call that asked for it — one model family would not take a tool result without that call. Across the
  local models tested, a tool round was the only shape that worked for all of them, which is why Raven gives
  definitions only on a turn where the model has tools.
- **A host that cannot store a tool round adds them as context instead**, as Claude Code's hooks do. On a local
  model, check that it works before relying on it.

### What has already been given

**Each entry is given once per conversation.** `select` takes `already_fired`, the entries the model already has,
and neither fires them again nor follows their links.

- **Raven** reads them off the branch, each stored definition naming its entry, so a branch remembers its own.
- **A host without that** keeps its own record, keyed by the conversation.
- **Compaction is the trap.** A host that summarizes old messages away loses the definitions with them, while its
  record says they were given. Either give them again after compacting, or forget them from the record.

### The scan, the budget and the cap

Raven's settings, `raven.librarian.config`, which a port can take as its defaults:

- **Scan the new message and the 4 before it**, leaving out earlier definitions and system text: a term in the
  model's last reply fires as well as one the user typed.
- **Follow links 2 deep** (`max_depth`).
- **Spend at most 2,000 tokens at once** (`budget`), and **8,000 in a whole conversation**, the cap being the
  budget's ceiling once what was already given is counted.

### The lookup tool

**Offer `tool_spec()` while the lorebook has entries, and reply with `answer(entries, term)`.** The model asks
for a term it needs that was not given; the answer is the entry, a near miss taken as meant, the near misses and
the entries mentioning the term, or a plain "not defined". The answers are written for the model to act on.

### The folder

**Follow it**, reading it again when a file changes, so an edited definition is used without a restart. `load`
returns the problems it met by file name, for the host to report.

## Inflected forms

**A term fires however it is inflected only if the matcher is given its forms.** `find_matches`, `select` and
`lookup` all take `expansions`: a mapping from a term's last word to an `Expansion(forms, stems)`. Only a term's
last word matches by them, it being the one that inflects.

A host has two ways to get them.

### The forms Raven generated

**Raven keeps them in sweep ledgers, `~/.config/raven/librarian/lorebook_expansion/expansions-*.jsonl`**, one per
model and prompt, in the format described in [`sweep-ledger.md`](sweep-ledger.md). A port on the same machine
reads them and generates nothing.

- **An answer's `id` is a term**, its title or an alias as `terms(entries)` keys it.
- **Its `answer` is `{word: {language: {"forms": [...], "stems": [...]}}}`**, the word being the term's last.
- **Its digest is of `"{term} — inflect: {word}"`**, so an answer about a term since edited is recognized as stale.
- **To build `expansions`**: for each word, gather the forms of every language read, and the stems of every
  language marked `"stems"` in `lorebook_languages` (none, by default); pass each word's through
  `clean_expansion(word, forms, stems)`; drop the words left with neither.
- **Which ledger to read is the port's choice**: Raven keeps one per model and prompt, each complete for that
  model, and an answer whose digest does not match the term as it is now is stale in any of them.

### Asking an LLM

**`raven/librarian/prompts/lorebook_expansion.md` is the prompt.** It is followed by the languages and a numbered
list of terms, each as `"{n}. {term} — inflect: {word}"`, and asks for JSON keyed by number. Raven asks ten terms
at a time, without thinking, and keeps the complete terms of a reply that runs on and never closes
(`lorebook_expansion.salvage`). The generation needs an LLM and takes minutes, so it belongs where the host can
run work in the background.

## Hosts

### Claude Code

**A plugin with two hooks**, checked against the hooks and plugins references in November 2026:

- **`UserPromptSubmit`** gets the prompt and `session_id` on stdin, and gives the definitions as
  `hookSpecificOutput.additionalContext`, which the model reads as a system reminder. **At most 10,000 characters
  per hook output**, with no setting to raise it, so the budget has to stay under that.
- **`SessionStart` matching `compact`** gives the session's definitions again after compaction, which summarizes
  hook-added context away.
- **The record of what was given** lives in `${CLAUDE_PLUGIN_DATA}`, keyed by the session id: the transcript keeps
  injected text in a record the docs do not describe.
- **The folder** is a `userConfig` setting of type `directory`, reaching the hook as an environment variable.
- **The default timeout for this hook is 30 seconds**, and one that runs out drops its context silently. The scan
  takes a fraction of a second; the forms are read from Raven's ledgers, never generated in the hook.
- **The hook runs whatever `python3` is on the path**, which has to be 3.11 or later.
- **Not verified**: whether a plugin's MCP server gives the model a lookup tool it can call, as MCP servers
  ordinarily do.

### Open WebUI

**Python, so the core goes in as it is.** Its functions are recalled to include filters that can change a
conversation before it reaches the model, which is where the definitions would go; not yet checked against its
documentation.

### LM Studio and pi

**TypeScript, so one translation of the core serves both**, checked against the conformance cases.

- **pi** (read in its extension documentation, November 2026): an extension's `before_agent_start` can return a
  message with a `customType`, stored in the session and sent to the model — the nearest thing to Raven's stored
  round — and `pi.registerTool` offers the lookup. That a port can recognize its own messages by that type on the
  current branch is inferred, not read.
- **LM Studio**'s prompt preprocessor can only prepend to the user's message, so its port marks what it injects,
  to find it again in the history.
