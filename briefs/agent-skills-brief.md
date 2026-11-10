# Brief: agent skills in Librarian

**Status: reviewed by the maintainer, 2026-11-10; ready to build.**

**Goal:** let a user describe a procedure in a Markdown file, and have Librarian's AI follow it when a task
matches. MCP supplies tools; a skill says how to use them. This is material that does not belong in a tool's
description: which of a lab's endpoints to call in what order, what a result means, what to check before
reporting it.

**Decided** (maintainer):

- The lab's December demo needs both MCP and skills (2026-10-19).
- **v1 is prose-only** (2026-11-10). No bundled script runs. See *What v1 leaves out*.

The format is the [Agent Skills](https://agentskills.io) open standard, originally Anthropic's: a folder
holding a `SKILL.md`, with YAML frontmatter and a Markdown body. Checked against its
[specification](https://agentskills.io/specification) and
[integration guide](https://agentskills.io/integrate-skills) on 2026-11-10.

---

## What the standard settles

Two questions the deferred item left open.

**Skills and the lorebook are separate things.** The guide says most implementations "rely on the model's own
judgment as the activation mechanism, rather than implementing harness-side trigger matching or keyword
detection". The model reads a catalog of names and descriptions and decides. The lorebook (brief 05) is the
reverse: deterministic keyword triggers, no model judgment, no resident catalog, and injection whether or not
the model would have asked. They share neither trigger nor injection path, so v1 shares no code with it. (The
deferred item's claim that the lorebook already provides the first two stages does not hold; it is corrected
there.)

**Not running scripts is a subset, not a violation, as far as the documents say.** The spec calls `scripts/`
optional and says supported languages depend on the implementation; the guide says "You don't need to support
every scenario". Neither has a conformance clause.

---

## The design

Activation in one sentence: the model calls `activate_skill`, and the skill's body comes back as that call's
result, which stays on the branch as an ordinary tool round.

### Where skills live

`~/.config/raven/librarian/skills/<name>/SKILL.md`, as `librarian_config.skills_dir`, beside `mcp.json`.

Not `~/.agents/skills/`, the cross-client convention, and not `~/.claude/skills/`. Skills written for a coding
agent assume a shell and a working directory that Librarian does not have. Someone who wants one symlinks it
in. There is no project scope, since Librarian has no project.

### Parsing

As the guide recommends, leniently: a name that disagrees with its directory, or is too long, warns and loads.
A missing description or unparseable YAML skips the skill and reports it. The body is read at activation, so
an edit takes effect without a reload. PyYAML is already in the tree.

### Disclosure and activation: one tool

A dedicated tool, `activate_skill`, built per session from what is installed:

- **The catalog is in its description**, with `name` constrained by an `enum` to the installed names, so the
  model cannot invent one. The guide lists this as one of its two placements.
  - It puts the catalog where the tool specifications already are, at the front of the prompt, changing only
    when the skills do. The system message's per-turn injects stay as they are, and no new root is created.
  - It shows up in the per-group token counts as any other tool does.
- **No skills, no tool.** Nothing is registered, and nothing is added to the prompt.
- **`file` is an optional second argument**, a path relative to the skill's directory, for tier-three prose
  such as `references/x.md`. Without it the model can read only `SKILL.md`, and a skill that splits its
  material as the spec advises would be half broken. A path resolving outside the skill's directory is refused.

A dedicated tool rather than file reading because Librarian has no general file-read tool, and a
filesystem MCP server is neither always present nor pointed at this directory.

### Activation by the user

A message starting with `/skill-name` activates that skill, in the GUI and in `minichat`: the harness stores a
synthetic `activate_skill` exchange before the user's message node, and the rest of the line is the message. A
`/word` naming no installed skill is ordinary text, as it may be a path.

### What the model gets back

The body with frontmatter stripped, wrapped as the guide suggests, with the bundled files listed and not read:

```
<skill_content name="lab-instruments">
...the body...

<skill_resources>
  <file>references/calibration.md</file>
</skill_resources>
</skill_content>
```

- **Returned whole.** It does not go through the document path that turns a long tool result into an
  attachment, since an excerpt of a procedure is worse than none.
- **A skill that lists `scripts/`** gets one line saying that scripts cannot be run here, so that the model
  says so instead of pretending. A `TODO` comment at that line says to remove it if script support arrives.
- **The tool message's metadata names the skill**, so that compaction, whenever it exists, can tell a skill
  from other tool results without parsing the wrapping tag.
- **Activating a skill already on the branch** answers that it is already loaded, as the guide suggests. The
  check walks the branch's tool messages, so branching back past an activation correctly allows it again.

### Switching it off

A **Skills** checkbox of its own, with a hotkey, beside the others. It gates `activate_skill`, and available
only when any skills are installed, as the MCP one is. The row of mode toggles is full, so other labels need
shortening to make room, and the F1 card's right column has no spare row either.

### Lifecycle

The MCP client's shape, for the same reasons:

- **One entry point for every frontend and script**, never called from `llmclient.setup`, which batch tools run
  and whose tool specifications a sweep's fingerprint covers.
- **`agent.turn` takes `skills_enabled=False`**, and `ai_turn` takes it as a required argument beside
  `internet_enabled` and `mcp_enabled`.
- **The apps follow the directory** and show a transient notice in the status row above the composer, as they
  do for reloads of `mcp.json`; a script does not.
- `activate_skill` is its own group in `maybe_tool_names_for_turn` and `tool_cost_by_group`, and survives
  `reconnect` as the other tool tables do.

---

## What v1 leaves out

- **Bundled scripts.** They wait for per-call approval (`TODO_DEFERRED.md`), since a script is code the model
  runs on the user's machine. Filed under "Agent skills for Librarian".
- **`allowed-tools`.** Experimental in the spec, and Librarian has no permission system for it to feed yet.
  Ignored, with a debug-level log line.
- **Autocomplete for `/skill-name`.** See *Activation by the user*.
- **Corpus scopes.** Whether a skill can restrict itself to one is a question for the drydock (brief 13), which
  is where scopes arrive.
- **Protection from compaction.** The guide says to exempt skill content. Librarian has no compaction yet; the
  wrapping tag is what will let it tell.

**What prose cannot do.** The worked example in the deferred item, classifying a paper pile, needed batching,
index-keyed answers, resumability and an escalation path computed outside the model. A skill describing that
workflow produces the naive version, which fails in ways the user cannot see. v1 does not solve this and
should not claim to: `raven.librarian.sweep` is where such a job belongs. Prose skills are for procedures a
model can follow turn by turn, which is what the lab demo needs.

**The demo's skill** says how the lab's MCP tools work together as an ecosystem, which is no single tool's
docstring's job. The lab's MCP docs are due after a meeting on 2026-11-13. Until then, a local skill chaining
several tools stands in.

---

## Acceptance

- A folder with a valid `SKILL.md` appears in `activate_skill`'s enum and description; with none, no tool exists
  and the prompt is byte-identical to one built without this feature.
- The model activates a skill on a matching task and follows it, tested live with a local model, including one
  that combines a skill with an MCP tool.
- `file` reads a bundled reference, and a path outside the skill's directory is refused.
- A malformed skill is reported and the others load.
- Adding or editing a skill while the app runs takes effect, with a notice, and existing chats are unaffected.
- A script on `agent` gets no skills unless it asks, and a sweep's fingerprint does not change.
