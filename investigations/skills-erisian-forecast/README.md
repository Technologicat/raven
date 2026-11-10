# Does a prose skill change what the model does?

A skill is only worth having if a model follows it, and does something it would not have done without it. This
is a test of that: a procedure chaining several tools in an order no tool's description suggests, with rules
that can be checked afterwards.

- `erisian-forecast/SKILL.md` is the skill. Copy the folder into `~/.config/raven/librarian/skills/`.
- `rhythmbox_mcp_server.py` is an MCP server over a music library on another machine, searching file names over
  SSH. `queue_track` only records the track to a file.
- The check: run the request with skills on and with them off, and compare the date against the arithmetic, the
  tool calls against the procedure's order, and the last sentence against the fixed text.

The server's `mcp.json` entry:

```json
"rhythmbox": {"command": "python", "args": ["rhythmbox_mcp_server.py", "--host", "HOST", "--queue-file", "/tmp/queue.txt"]}
```

## Results, 2026-11-10, Qwen 3.5 9B (Q4, 128 Ki context)

With the skill on, in all five runs the model activated it, made the tool calls in the procedure's order, queued
the track the rules select, and ended with the fixed sentence. With it off, it fetched the weather and answered
with a forecast table. So the skill does change the behaviour, for this model.

The date was the weak step. Before the skill gave a table of month offsets, neither of two runs got it right. With the
table, two of three did; the third added the offset wrongly in its head, and one of the two right ones skipped
the calculator altogether. A step the model can skip, it sometimes will.
