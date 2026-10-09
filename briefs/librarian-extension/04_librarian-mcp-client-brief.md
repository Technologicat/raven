# Brief: client-side MCP tool client in Librarian

**Relationship to the LM Studio compat brief:** independent. That one ships before the autumn
demo regardless; this one is gated on the Hindsight playground being up and is the
"interesting path," not the deadline path. They share one thing — the verified OpenAI function
spec (see §2) — and nothing else.

**Goal:** make external tools first-class **alongside** Raven's built-in ones, not in place of
them. Today `llmclient.setup` hardcodes `tools` / `tool_entrypoints` to just `websearch`.
That set is expected to grow — webfetch (phase 1), a sandboxed calculator, weather, docs-DB
access, possibly local file I/O later — and built-ins remain the right shape for capabilities
that are core to Librarian's mission and benefit from tight integration with Raven internals.
**Extend** the registry with a client-side MCP client so any configured MCP server's tools
become available *in addition* to the built-ins, **all feeding the existing
`perform_tool_calls` loop** rather than splitting it. MCP becomes one *source* of tools among
several; the scaffold, the `on_call_start`/`on_call_done` events, timing, and error handling
are common path.

**Explicitly NOT this brief:** backend-hosted MCP (LM Studio talking to MCP servers itself).
That's the quick path for *playing with* Hindsight as a tool and bypasses Raven's tool loop
entirely — fine for experiments, wrong for the product. Different job.

---

## Scope — keep it to the tools slice

The MCP spec is broad (resources, prompts, sampling, roots, multiple transports). Librarian
needs **one capability: tools, as a client.** Ignore the rest until there's a concrete need.
Consume the official `mcp` Python SDK (PyPI: `mcp`) — you are a *client* of the protocol, not an
implementer of it. The real work is a thin adapter, not a protocol.

### Which version (checked against PyPI 2026-09-29)

- **Pin `mcp>=2.2,<3`.** 2.0 shipped 2026-07-28, and 1.x is still maintained alongside it (1.30.0
  and 2.2.0 came out the same day), so a bare floor is ambiguous and 1.x-era examples online are
  misleading.
- **2.x splits the wire types into `mcp-types`** (pydantic only) and uses **`httpx2`** as its HTTP
  client, a separate package from the `httpx` 0.28 already in our tree.
- **There is no client-only install.** `mcp` also pulls in its server-side dependencies (`starlette`,
  `sse-starlette`, `uvicorn`, `python-multipart`, `pyjwt[crypto]`, `opentelemetry-api`). Several are
  already here transitively; what's new is moderate.
- **FastMCP was considered and rejected.** Its client extra (`fastmcp-slim[client]`) depends on `mcp`
  itself, so it would be a strict superset with more dependencies, and its strengths are server-side.
- **A hand-rolled sync client on `requests` + `sseclient-py` was also considered.** It would remove
  the async bridge (§1) entirely, since the tools-only slice of the protocol is small. Rejected
  because that trades a one-time bridge for tracking every spec revision ourselves, answering
  server-initiated requests correctly, and giving up OAuth for remote servers.
- **CI:** if the adapter's tests import `mcp`, add it to `requirements-ci.txt` or guard the import
  with `importorskip` (`scripts/check_ci_imports.py` says which).

---

## 0. Prerequisite — `@tool` decorator and parameter-metadata pattern

Before the MCP adapter can register MCP-sourced tools alongside built-ins, Raven's built-in
tool-spec construction needs to be unified into a registry-and-decorator pattern. This is
genuinely §0 (not §1) because the MCP adapter is the *second* consumer of the
spec-construction machinery; built-ins are the first.

**Today**: `llmclient.setup` hardcodes `tools` / `tool_entrypoints` for `websearch`,
constructing the OpenAI function spec inline. Each new built-in (webfetch, calculator,
weather, docs-DB access) means more hardcoded spec construction, with spec/implementation
drift as a permanent maintenance risk.

**Goal**: declare tool metadata *once*, at the function definition site, and derive both the
OpenAI function spec and the entrypoint registration automatically. Pattern follows the
avatar postproc GUI builder precedent: metadata on the function, machinery introspects.

### The pattern: `Annotated[T, Metadata]`

Following PEP 593 (`typing.Annotated`) — the modern Python convention used by FastAPI,
Pydantic, Typer, and others. The annotation `T` stays the real type (mypy reads it cleanly,
no decorator-induced obfuscation); metadata travels alongside, accessible via
`typing.get_type_hints(include_extras=True)`.

```python
from typing import Annotated
from raven.librarian.llmclient import tool, Param

@tool(name="websearch", description="Search the web for current information.")
def websearch(
    query: Annotated[str, Param("The search query. Use specific terms.")],
    max_results: Annotated[int, Param("Maximum results to return",
                                      constraints={"minimum": 1, "maximum": 20})] = 5,
    engine: Annotated[str, Param("Search engine to use",
                                 constraints={"enum": ["duckduckgo", "startpage"]})] = "duckduckgo",
) -> list[Part]:
    ...
```

What the type-checker sees: `query: str`, `max_results: int = 5`, `engine: str = "duckduckgo"`,
return type `list[Part]`. Fully mypy-strict-clean. The decorator returns the function
unchanged (registration is a pure side effect), preserving the call-site signature for any
direct callers.

```python
from functools import wraps
from typing import Callable, TypeVar
F = TypeVar("F", bound=Callable)

def tool(*, name: str, description: str) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        TOOL_REGISTRY.append(_build_entry(func, name, description))
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper  # type: ignore[return-value]
    return decorator
```

`@wraps(func)` preserves `__name__`, `__doc__`, `__module__`, `__wrapped__`, and friends —
needed so `help(websearch)` works correctly, IDE tooltips show the right docstring,
introspection tools see through the wrapper, and `inspect.signature(websearch)` returns the
original signature. Standard hygiene even when the wrapper is a passthrough; matters more
the moment any future decorator variant actually wraps behavior.

### What lives where

**In `llmclient` itself** (top of the module, where a developer reading `setup` finds it
immediately — discoverability matters more than file-size discipline here):

- The `@tool` decorator
- The `Param` dataclass (carries `description` and optional `constraints` dict)
- `TOOL_REGISTRY` module-level list
- `_build_entry()` helper that introspects via `get_type_hints(include_extras=True)` +
  `inspect.signature`, composes the OpenAI function spec, derives `required` from
  absence-of-default, and auto-injects `additionalProperties: false` (the §2 hardening,
  free for every registered tool)

`setup` walks `TOOL_REGISTRY` instead of hardcoding the tool list. The MCP adapter (§1)
constructs the same registry-entry shape from MCP `inputSchema` and appends to the same
registry; downstream code doesn't distinguish.

**Scope: existing built-in tools get converted as part of this phase.** `websearch` (and
`webfetch` once phase 1 lands) currently exist as hardcoded entries in `setup`; both get
rewritten to use `@tool` / `Param`. ~30 lines of conversion per tool, mechanical.

### JSON Schema generation

A small mapping covers what tool params actually look like in practice:

- `str`, `int`, `float`, `bool` → primitive schemas
- `list[T]` → `{"type": "array", "items": <T's schema>}`
- `Optional[T]` / `T | None` → optional-with-null
- `Literal[...]` → enum
- `dict[str, T]` → object with `additionalProperties`

Roll our own (~40-50 lines of dispatch) rather than pull Pydantic just for this — keeps dep
surface clean, gives full control over emitted schema, matches the lightweight precedent of
the avatar postproc utility.

### Standardizing across the codebase

The `Annotated[T, Metadata]` shape is **the convention Raven should standardize on for
parameter metadata across contexts**. Different contexts use different metadata classes
(each context's data needs differ), but the consumption pattern stays uniform:

- **Tool specs** (this brief): `Param` carries description + JSON Schema constraints.
- **Postproc settings** (future refactor): would use a `Setting` class carrying widget hints
  and live ranges. The existing hand-rolled postproc machinery predates awareness of
  `Annotated`; refactoring it to this shape is documented as the natural cleanup target but
  is *not in scope here* — postproc works as-is, the refactor is purely "make this cleaner"
  with no urgency.
- **Future contexts** (lorebook config field definitions, visualizer importer settings,
  others) inherit the convention naturally.

No shared base class for the metadata classes — they'd either be empty (no real
abstraction) or kitchen-sink (every context paying for every other context's fields). The
shared part is the *shape*, not a class. Each metadata class is independent and lives in
its own module; the import path disambiguates at call sites (`from raven.librarian.llmclient
import Param` vs `from raven.server.modules.postproc import Setting`).

The naming convention is unprefixed within each module — `Param`, not `ToolParam`. The
Pythonic pattern: let imports carry the namespace.

**Shared empty marker base** (`AnnotationMetadata`) — `Param`, `Setting`, and any future
Raven metadata classes inherit from it. The base has no fields, no methods; it exists solely
so tooling can identify Raven metadata via `isinstance` / `issubclass` without maintaining
a hand-curated class registry. (See the postproc appendix for the full rationale.) **The
line worth not crossing**: putting fields on the base — that's where kitchen-sink drift
starts. Empty marker yes; shared fields no.

---

## 1. The adapter (the bulk of the work, and it's mechanical)

Per configured MCP server:

1. Open a client session (transport per §3).
2. `list_tools()` → for each tool, translate its `inputSchema` (JSON Schema) into the OpenAI
   function spec via §0's `_build_entry()` machinery (it receives the schema from MCP rather
   than introspecting a Python function, but produces the same registry-entry shape as
   built-ins). Append to `settings.tools`.
3. Register a `tool_entrypoint` closure that calls `session.call_tool(name, args)` and returns
   the result as content parts per the content-parts brief. Add to `settings.tool_entrypoints`.

After this, `perform_tool_calls` is unchanged — it looks up the entrypoint by name and calls it.
The closure happens to dispatch over MCP instead of calling a local function. That's the whole
point of the design: the loop doesn't know or care.

The `mcp` SDK is `asyncio`; `llmclient` is sync. This is the one part of the work that needs
careful design rather than mechanical typing — see the sync/async impedance subsection below.

### Sync/async impedance — the async bridge

**Do not** thread async through `llmclient` — `invoke`, `perform_tool_calls`, and the scaffold
must stay sync. Isolate all async behind a sync facade.

**Shape: a small generic utility module, not an MCP-specific bridge.** Put it at
`raven.common.async_bridge` — one class, ~60 lines, owning a background thread + event loop and
exposing `submit(coro) -> result` (blocking) and `shutdown(timeout)`. The MCP client is its first
consumer; later asyncio consumers (the `lmstudio` SDK if it ever earns its way in, a
`playwright`-based browser tool, async arXiv clients, any other async library Raven absorbs)
reuse it. Generic to the *bridging pattern*, not generic to async paradigms — there's no good
cross-paradigm abstraction (asyncio vs. trio vs. anyio vs. threading vs. native CPS genuinely
don't compose, and every attempt at a Unified Async Theory has either lost to network effects or
gone baroque), so don't try; just keep this bridge tight enough that wiring a new asyncio
dependency through it is mechanical.

Concretely:

- Dedicated background thread runs an asyncio event loop.
- `submit(coro)` → `asyncio.run_coroutine_threadsafe(coro, loop).result()`, blocking.
- `shutdown(timeout)` cancels pending tasks, stops the loop, joins the thread (this is the
  `shutdown(timeout)` the lifecycle section relies on for force-terminating hung sessions).
- The MCP client wraps this: `list_tools()` and `call_tool()` are sync methods that internally
  `bridge.submit(session.list_tools())` etc.
- **Probe before writing the shutdown path (unverified against `mcp` 2.x):** the SDK is anyio-based,
  and anyio cancel scopes must be exited in the task that entered them. If that holds here, the
  `async with stdio_client(...)` / `ClientSession` contexts can't be opened by one `submit` and
  closed by another. Each server would then need one long-lived holder coroutine that keeps its
  session open until a shutdown event, with `call_tool` submissions from other tasks being fine.

**Why a bare daemon thread rather than `bgtask`** (the obvious question, so answered here):
`bgtask.TaskManager` groups *completable, cooperatively-cancellable* callables for cancellation —
typically several managers sharing one `ThreadPoolExecutor`. A `run_forever()` event loop is none
of those things: it never completes, it ignores the `env.cancelled` poll that cooperative
cancellation relies on (only `loop.call_soon_threadsafe(loop.stop)` breaks it), and it would pin
a pool worker for the entire app lifetime. The loop's real units of work are the coroutines,
tracked by asyncio and by the `concurrent.futures.Future` that `run_coroutine_threadsafe` returns
— not by `bgtask`. So routing it through `bgtask` would engage none of `bgtask`'s machinery while
leaking asyncio into the shared task manager. `bgtask` and `async_bridge` are *sibling*
`raven.common` primitives — one backgrounds a callable, the other crosses into asyncio — not one
built on the other. (Note thread *creation* isn't `bgtask`'s job either: it takes an executor as
a dependency.)

Everything upstream of the facade stays exactly as sync as it is now. This is also the boundary
between Raven's thread-based concurrency world (`bgtask.TaskManager` over `ThreadPoolExecutor`,
GIL-tolerated because the work is I/O- or GPU-bound) and the asyncio-shaped dependency — Raven
doesn't adopt asyncio internally, it just talks to it at this one boundary.

---

## 2. Schema translation — the tool spec is verified correct

The OpenAI function spec Raven builds was reverse-engineered from an informal DeepSeek example
and never checked against the real thing. It is now verified against the LM Studio tools doc
(local copy `00_stuff/lmstudio_api_docs/oai_03_tools_and_function_calling.md`; online at
https://lmstudio.ai/docs/developer/openai-compat/tools): the shape

```json
{"type": "function",
 "function": {"name": ..., "description": ..., "parameters": <JSON Schema object>}}
```

is exactly right. **Carry these through from MCP `inputSchema` when present** (they're worth
having, not optional polish): `additionalProperties: false` (stops the model inventing params)
and `enum` on constrained-string params. MCP tool `inputSchema` *is* a JSON Schema object, so
translation is mostly: wrap it as `function.parameters`, carry `name` and `description` across,
derive `required` from the schema. No lossy guessing. The `additionalProperties: false`
hardening applies uniformly to built-ins too, since §0's `_build_entry()` injects it for every
registered tool regardless of source.

---

## 3. Transports and configuration

Support both:

- **stdio** — local subprocess MCP servers (filesystem, sqlite, any tool that runs as a
  subprocess speaking MCP over stdin/stdout).
- **streamable HTTP / SSE** — networked servers, including the Docker-served HTTP case
  (Hindsight, and anything else that runs in a container and exposes an MCP endpoint). This is
  the v0 critical path, since the first target (Hindsight) is HTTP.

**Config: an `mcp.json` file, and nothing else.** `raven.librarian.config` names the file,
beside the API key file:

```python
mcp_config_file = librarian_userdata_dir / "mcp.json"  # will be used if it exists, ignored if not.
```

A missing file means no MCP servers, the same way a missing `api_key.txt` means no key. Raven
reads the file's top-level `mcpServers`, which is the de facto MCP-client schema (Claude Desktop,
Cursor, Cline, LM Studio, Claude Code), so an entry can be lifted from any of them unchanged. To
share LM Studio's file instead of keeping a copy, point `mcp_config_file` at
`~/.lmstudio/mcp.json` in `overrides.json`.

```json
{
  "mcpServers": {
    "hindsight": {
      "url": "http://localhost:8788/mcp",
      "headers": {"Authorization": "Bearer ..."},
      "allowed_tools": ["retain", "recall", "reflect"]
    },
    "local-filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/home/jje/notes"],
      "disabled": false
    }
  }
}
```

(The Hindsight URL is a placeholder; the actual port is per Hindsight's docs.)

Field names match the convention: `command`/`args`/`env`/`cwd` for stdio, `url`/`headers` for
HTTP.

- **Transport is inferred, with `type` as the override.** An entry with `command` is stdio; one
  with `url` is HTTP (streamable). An explicit `type` (`"stdio"`, `"http"`, `"sse"`) wins —
  that is Claude Code's key, and Cline's. LM Studio's and Claude Desktop's entries carry no
  transport key at all, which is why inference is needed rather than a required field. The
  value spellings vary between clients (check Cline's for streamable HTTP when implementing);
  accept the known ones and reject anything else with an error naming the server.
- **`disabled`** (default `false` if absent) lets the user flip a flaky server off without
  removing its entry — the equivalent of commenting out a line of config, but expressed in
  config data. It is Cline's key, and using it rather than an `enabled` of our own keeps the
  dialect count down.
- **`allowed_tools`** is ours. It restricts which of the server's tools are exposed at all,
  which is a different thing from Cline's `autoApprove` (tools that skip confirmation), so it
  is not a synonym for it.

Open: whether LM Studio tolerates keys it does not know (`disabled`, `allowed_tools`) in a
shared file. Unchecked as of 2026-10-09; one try with a dummy entry settles it.

**How `headers` reaches the SDK** (checked against `mcp` 2.3.0, 2026-10-07). The two HTTP
transports take it differently:
- `streamable_http_client(url, http_client=...)` has no `headers` parameter of its own. Headers and
  auth go on the `httpx2.AsyncClient` passed in, and `mcp.shared._httpx_utils.create_mcp_http_client(headers=..., auth=...)`
  builds one with the timeouts the transports expect.
- `sse_client(url, headers=..., auth=...)` takes both directly.

Either way the SDK merges its own MCP headers (session id, protocol version) on top of ours. It
follows redirects itself, and only within the endpoint's origin, so a key is not forwarded to
another host. A per-service API key is therefore just an entry in `headers`. For a credential
that has to be computed per request, `auth=` takes an `httpx2.Auth`, and `mcp.client.auth.oauth2`
covers the spec's OAuth flow if a server ever requires it.

Secrets such as `headers` therefore live in `mcp.json`, outside the repository, as the LLM
API key does in `api_key.txt`.

**Static at startup, not live-reloadable in v0.** Tearing down and restarting an MCP transport
(subprocess or HTTP session) is invasive enough that hot-reload isn't worth it before there's a
reason. The `disabled` flag covers the day-to-day "skip this flaky server" case without a full
config reload.

### Spawn stdio servers in a session of their own

When Librarian runs from a terminal, a stdio server it spawns can stop the whole app, as if Ctrl+Z
had been pressed. Seen in LM Studio, 2026-10-07: launched from a terminal, with the
mark3labs filesystem server or Playwright MCP enabled, LM Studio stopped at the start of every
AI turn and needed `fg`, several times over. The other two servers in the same config never did
it, and the cause inside those two was not found. Running LM Studio under `script -qc`, which
gives it a pseudo-terminal of its own, cured it.

Raven can avoid it from the spawning side: start each stdio server in a new session
(`start_new_session=True`, i.e. `setsid`), so it has no controlling terminal to stop anyone
through. Check whether the `mcp` SDK's `stdio_client` already does this on POSIX. If it does not
and offers no hook for it, wrapping the command in `setsid` is the fallback. If the symptom
appears in testing anyway — the app's shell reports `Stopped` on the first turn after an MCP
server comes up — this is what it is.

---

## 4. Tool namespacing

Multiple servers can expose the same tool name. Namespace registered tools by server label
(e.g. `label__toolname`) and reverse-map at call time, so `tool_entrypoints` keys stay unique
and the model is told unambiguous names.

**Check `mcp.client.session_group.SessionGroup` first**: per its docstring it manages several
sessions concurrently, aggregates their tools, and takes a user-provided hook for name collisions,
which may cover this section and part of §1's per-server bookkeeping. Only the docstring has been read.

### MCP tools want their own GUI toggle, separate from "Internet"

Re-homed here 2026-08-12 from the deferred item that shipped the Internet toggle, because it is the only
place this was written down and it is a decision this brief has to make.

The **Internet** toggle now gates the network-reaching built-ins (`websearch`, `webfetch`) specifically,
rather than all tools indiscriminately — `ai_turn` takes an allowed-tool-name set, and the app maps each GUI
toggle onto the relevant names. That machinery is what an MCP toggle would reuse.

**But MCP is a different trust surface and should not ride on that switch.** An MCP server is third-party
code the user chose to run; a user may reasonably want their local tools available while the model's reach
to the open internet is off, or the reverse. Folding both under one control asserts they are the same risk,
and they are not — which is the same reason the Internet toggle exists at all rather than the original
all-or-nothing `tools_enabled` bool.

Open: whether one toggle covers all configured servers or each server gets its own. Per-server matches the
config shape (`"disabled"` per entry, §3) and matches the trust argument, since two servers need not be
equally trusted; one toggle is less GUI. The `disabled` flag in config is the *static* answer to this — the
toggle is the per-session one, and they should agree on what a disabled server does to the tool list.

---

## 5. Lifecycle

- Open sessions and `list_tools()` at app start (in or alongside `setup`).
- Close cleanly at shutdown: **the MCP client exposes a `shutdown()` method; the GUI
  registers it with `gui_shutdown`** (not the other way around). Lifecycle hooks follow
  the inverse-dependency pattern: scaffold-layer code knows about its own shutdown
  semantics; GUI code knows it needs to call scaffold's shutdown when the app closes; scaffold
  imports nothing GUI-side. Alongside the recent "robustify app exit handling" bail-pattern
  work (known good on Linux, not yet tested on Mac/Tahoe). **Shutdown with timeout**: the
  async-bridge's `shutdown(timeout)` must force-terminate hung sessions after a few seconds
  rather than blocking exit indefinitely — a misbehaving MCP server must never prevent
  Librarian from quitting.
- Handle server reload/disconnect without crashing the chat: mark the server's tools
  unavailable, optionally re-list on reconnect. A dead tool server must not take down Librarian.

### Architectural constraint: scaffold doesn't import GUI

The MCP client (and the `@tool` registry from §0) live in the scaffold/inference layer,
not the GUI layer. **No imports from DearPyGUI, `chat_controller`, `gui_shutdown`, or any
other GUI module from inside scaffold-layer code.** Dependencies flow one way: GUI knows
scaffold, scaffold doesn't know GUI.

**`minichat` is the canary** — a minimal TUI chat client (terminal-based, not DPG) that
consumes scaffold without the desktop-app layer. If any phase-4 change accidentally
introduces a GUI import into scaffold-layer code, `minichat` breaks at import time.
**Verify `minichat` continues to start and function through phase-4 changes** — cheap to
run, immediate fail signal if the layering regresses. The constraint applies equally to
the §0 tool registry, §1 adapter code, this section's lifecycle hooks, and anywhere else
phase 4 touches scaffold internals.

---

## 6. Later: Librarian as an MCP *server* (scoping notes, 2026-09-29)

Not part of this brief's acceptance, and recorded here so the client work doesn't paint it into a corner.

**Use case:** other local AI agents in the lab query Librarian for what it can extract from the
document DB. Two shapes, and both are wanted:

- **Retrieval**: a tool over the HybridIR index that returns chunks. Thin.
- **An agent turn**: a RAG-backed LLM turn that returns an answer, built on `raven.librarian.agent`.
  Capacity is fine for a few parallel sessions, and the number of clients is under our control.

**Library:** the same SDK. `mcp.server.mcpserver.MCPServer` (2.x's name for what 1.x called
`FastMCP`) has a `@tool` decorator and serves stdio, SSE and streamable HTTP, and
`mcp.server.lowlevel.Server` is there if we need to supply schemas ourselves. §0's registry may be
able to feed both directions; not checked yet.

**Where it runs depends on brief 13** (*Where the database lives*). If the DB moves behind
Raven-server, the MCP server reaches it the same way every other app does. Until then, a second
process reading the index needs an answer on concurrent access. Note the SDK's server app is ASGI
(Starlette) while Raven-server is Flask (WSGI), so it doesn't simply mount into the existing server.

**Trust:** it exposes a possibly private corpus, so it binds to localhost by default. This is also
the concrete case for question 5 of `briefs/design/constellation-architecture-sketch.md`.

**When it listens beyond localhost, guard it with a per-service API key** (2026-10-07). The client
end sends `Authorization: Bearer <key>` through §3's `headers`. On the server end there are two
options, both checked against the `mcp` 2.3.0 source and neither yet run:

- **The SDK's hook: `MCPServer(token_verifier=..., auth=AuthSettings(...))`. Not this one.**
  `TokenVerifier` is a one-method protocol (`async verify_token(token) -> AccessToken | None`), and
  a static key fits it. But this path is built for OAuth. `AuthSettings.issuer_url` is mandatory,
  and the server then publishes OAuth protected-resource metadata pointing at an authorization
  server that does not exist. Our client sends the key up front and never sees the 401 that
  starts discovery. A third-party MCP client might, and would then go looking for that nonexistent
  authorization server.
- **A plain ASGI middleware around the Starlette app. This one.** It is about fifteen lines,
  publishes nothing it cannot back, and matches the client config exactly:

  ```python
  class RequireAPIKey:
      def __init__(self, app, key: str):
          self.app, self.key = app, key.encode()

      async def __call__(self, scope, receive, send):
          if scope["type"] == "http":
              supplied = dict(scope["headers"]).get(b"authorization", b"")
              if not hmac.compare_digest(supplied, b"Bearer " + self.key):
                  await send({"type": "http.response.start", "status": 401, "headers": []})
                  await send({"type": "http.response.body", "body": b""})
                  return
          await self.app(scope, receive, send)
  ```

  Each service reads its own key at startup: from an env var, or from the overrides file for a
  Raven process. The constant-time comparison stops the check from leaking the key through timing.

Two limits. stdio servers need none of this, being the client's own subprocesses with no network
endpoint. And a static key over plain HTTP is cleartext, which is acceptable on localhost or the
trusted LAN Raven-server already assumes. Beyond that it needs TLS in front, e.g. a reverse proxy,
which could then do the key check instead and keep auth out of the Python altogether.

---

## Acceptance

- A configured MCP server's tools appear in `settings.tools`; the model calls one; the
  registered closure routes it through `perform_tool_calls` → MCP session → result back to the
  model; a multi-turn tool exchange completes.
- Works over both stdio and HTTP transports.
- `llmclient`/`scaffold` remain synchronous; all async stays behind the `raven.common.async_bridge`
  facade (the sync/async impedance subsection in the adapter section).
- A dead or reloading MCP server degrades gracefully and does not crash the chat.
- Hindsight reachable as a live test target.

### A second test target: ComfyUI (maintainer, 2026-10-01)

The weather tool became a built-in on 2026-09-30 (`TODO.md`, the open-meteo item), so it is not an MCP
target any more. The replacement: **ComfyUI, so that Aria can draw pictures.** An MCP connector for ComfyUI
already exists on GitHub, and the maintainer's own workflows can feed the models. Nowhere near what one can
do in Comfy by hand (no inpainting and the like), but three modes:

- **txt2img**;
- **img2img**;
- **edit mode**: several images and a prompt saying what to do with them — the spiritual successor of
  instruct-pix2pix.

Two things this target needs that Hindsight does not (maintainer, 2026-10-01):

- **A tool result can carry image parts.** For the chat log to show the picture, and for a VLM to see it.
  `chatutil.create_chat_message` builds single-text tool messages and says multi-part content arrives "where
  a real caller needs it"; this is that caller. The output image presumably goes to a sidecar through the
  existing attachment mechanism (`imagestore` over `sidecarstore`), as an uploaded image does.
- **A tool can be sent an image** — img2img and edit mode take images as input. Wanted as a general
  mechanism: a tool argument naming **an attachment in the current chat, or an item in the document DB**.
  - The DB half is the same missing piece as `TODO_DEFERRED.md`'s "`fetch_document` inlines up to eleven
    times…", which is blocked on addressing: `sidecar:` URLs name a copy inside the datastore, and there is
    no scheme for a document in the local database. One addressing scheme would serve both, so design them
    together.

And one consequence for scheduling, a reader's inference: a ComfyUI target needs no Hindsight, so it may
loosen this brief's gate on the Hindsight playground.

---

## Appendix: postproc annotation refactor (separate task, share this file for context)

**Scope**: this appendix is documentation for a *separate refactor task*, kept in this file
so a developer assigned to it has the full §0 pattern spec in the same read. Not part of the
phase-4 MCP work itself.

### Current state

`raven.common.video.postprocessor` defines a `with_metadata(**metadata)` decorator at
`postprocessor.py:358` and applies it to ~25 effect methods on the `Postprocessor` class.
Usage shape:

```python
@with_metadata(center_x=[-1.0, 1.0],
               center_y=[-1.0, 1.0],
               strength=[0.0, 1.0],
               mode=["analog", "digital"])
def some_effect(self, center_x, center_y, *, strength=0.5, mode="analog"):
    ...
```

Metadata format: parameter name → `[min, max]` range, or enum list `[choice1, choice2, ...]`.
The decorator stores metadata as a function attribute that a downstream GUI builder
introspects to construct sliders / dropdowns / etc. for live effect parameter editing.

Predates awareness of `typing.Annotated`. Hand-rolled solution that works but is now the
*odd one out* relative to the `@tool` / `Param` pattern §0 establishes.

### Why refactor

1. **Codebase consistency.** Two parameter-metadata patterns in the same codebase is a
   maintenance tax. Future readers learning either pattern have to learn the other too, and
   the gravitational pull of "follow the pattern you saw first" produces drift in either
   direction.
2. **Colocation.** Current form has metadata on the decorator, parameters on the signature;
   the link between them is by name match, which is brittle (rename the parameter, forget
   to update the decorator → metadata silently mismatches). New form puts each parameter's
   metadata *on the parameter itself*, eliminating the by-name-link failure mode.
3. **mypy hygiene.** Current form is transparent to type checkers (parameters are just
   typed normally); new form is also transparent (`Annotated[T, ...]` reads as `T`), so
   nothing lost. But the new form actively *encodes* the type as the first arg of
   `Annotated`, which means a future "validate that the metadata range/enum is compatible
   with the declared type" checker becomes trivial to write (compare runtime metadata
   against the type extracted from the same `Annotated`). Hand-rolled current form would
   need its own type-tracking to do this.
4. **Description as data.** Current form has no first-class slot for per-parameter
   descriptions (the docstring is the unstructured fallback). New form structures it.

### Target shape

Same `Annotated[T, Metadata]` pattern as `@tool` / `Param` from §0; metadata class is
`Setting` (postproc-specific data: description + range + GUI flags, vs. tool's `Param` which
carries description + JSON Schema constraints).

```python
from dataclasses import dataclass
from typing import Annotated, Optional
from raven.common.video.postprocessor import setting, Setting

class Postprocessor:
    @setting(name="some_effect", description="Apply effect X to the frame.")
    def some_effect(self,
                    center_x: Annotated[float, Setting("Horizontal center",
                                                       range=(-1.0, 1.0))] = 0.0,
                    center_y: Annotated[float, Setting("Vertical center",
                                                       range=(-1.0, 1.0))] = 0.0,
                    *,
                    strength: Annotated[float, Setting("Effect strength",
                                                       range=(0.0, 1.0))] = 0.5,
                    mode: Annotated[str, Setting("Effect variant",
                                                 enum=["analog", "digital"])] = "analog",
                    tint_rgb: Annotated[tuple, Setting("Color tint",
                                                       widget="rgb_picker")] = (1.0, 1.0, 1.0),
                    name: Annotated[str, Setting("Effect instance name",
                                                 gui_visible=False)] = "") -> Frame:
        ...
```

Sketch of the `Setting` dataclass (fields are illustrative; the actual set lands during
implementation based on what the existing 25 effects + their GUI need):

```python
@dataclass
class Setting:
    description: str = ""
    range:        Optional[tuple[float, float]] = None    # numeric ranges
    enum:         Optional[list] = None                    # discrete choices
    widget:       Optional[str] = None                     # GUI widget hint: "slider", "rgb_picker", ...
    gui_visible:  bool = True                              # exclude internal params from the GUI
    units:        Optional[str] = None                     # future use: "px", "Hz", "dB", ...
```

Decorator renamed `@with_metadata` → `@setting` for parallelism with `@tool`. Class renamed
implicit-metadata-dict → explicit `Setting` dataclass with properly-named fields.

### Cleaning up overloaded slots in the existing metadata

The current `with_metadata` only has the `[choice1, choice2, ...]` value-list as a metadata
slot per parameter, which has led to *three different concerns being stuffed into the same
slot* via magic sentinel strings:

| Current ad-hoc form         | Concern actually being expressed       | New field on `Setting`     |
|-----------------------------|----------------------------------------|----------------------------|
| `name=["!ignore"]`          | "skip this parameter in the GUI"       | `gui_visible=False`        |
| `tint_rgb=["!RGB"]`         | "render with RGB color picker widget"  | `widget="rgb_picker"`      |
| `mode=["analog", "digital"]`| actual enum constraint                 | `enum=["analog", "digital"]` |

These got jammed into the value-list because that was the only slot available — exactly the
shape-of-the-thing problem the refactor solves. Each gets its own properly-named field on
`Setting`; readers see at a glance what each constraint means, no magic strings to memorize.
About 8 occurrences of `["!ignore"]` and 1 of `["!RGB"]` in `postprocessor.py` to clean up
(grep `["!.*"]` to find them all).

### What stays the same

- **The downstream GUI builder.** The new `@setting` decorator stores metadata in the same
  function-attribute shape the builder expects, so the builder doesn't need refactoring
  beyond reading from the new properly-named fields instead of decoding magic sentinels.
  Builder + decorator + all 25 effects get updated atomically in the same commit (everything
  lives in one module; the migration is genuinely a single-session job, no need for a
  transitional coexistence shim).
- **All effect logic.** Pure mechanical signature refactor; no behavior change.

### Migration approach — single-session atomic conversion

All 25 effects live in one module (`postprocessor.py`) and the GUI builder is our own code,
so the migration is a one-PR, one-commit job. Steps:

1. Add `@setting` + `Setting` dataclass (with all the properly-named fields) + introspection
   helper in `postprocessor.py`. ~60 lines including the field definitions.
2. Update the GUI builder to read from the new properly-named fields on `Setting` instead
   of decoding magic sentinels from the value-list.
3. Convert all 25 effects in a single pass. Each conversion is mechanical:
   - Lift the kwarg-named ranges from `@with_metadata` up into `Annotated[T, Setting(...)]`
     on the corresponding parameter.
   - **Decode the magic sentinels** at the same time: `["!ignore"]` → `gui_visible=False`,
     `["!RGB"]` → `widget="rgb_picker"`, actual enums stay as `enum=[...]`. Each parameter's
     metadata becomes self-documenting after the move.
   - Add descriptions (now first-class) by lifting from docstrings where present.
   - Preserve defaults.
4. Remove `with_metadata` and the magic-sentinel-decoding paths in the GUI builder. Both die
   in the same commit; no transitional state to reason about.
5. Verify the GUI looks/works the same across all converted effects (visual regression
   sweep, or spot-check each in turn).

Mechanical, low-risk, fits an afternoon pair session. The atomic approach is strictly
simpler than incremental for this case — coexistence shims earn their weight only when
migration is necessarily incremental (different modules, different ownership, or partial
deploys), none of which applies here.

### Sequencing relative to other work

- **Not blocking** any other work. Postproc functions correctly as-is.
- **No technical reason for either order.** The `Annotated[T, Metadata]` pattern can be
  locked in either context first; whichever ships first establishes the convention, and the
  other adopts it. The dependency is "decide once, apply consistently," not "build A before
  B." If postproc had been the active workstream when this came up, it would have been the
  natural place to introduce the pattern.
- **Phase-4 MCP first is a scheduling call.** We're actively working on Librarian, MCP is
  the immediate need, postproc isn't broken. So the convention lands in `@tool` / `Param`
  first because that's where the work is, and postproc converges on it whenever convenient.
- **Could happen anytime** after phase 4 lands. No external dependencies.

### Other natural future consumers of the same pattern

Worth flagging in the same appendix since they may come up:

- **Lorebook entry field definitions** (when configurable entry schemas land): each
  metadata field of an entry could be `Annotated[T, LoreField(...)]`.
- **Visualizer importer settings**: many configurable per-importer knobs; currently config
  dictionaries, would benefit from the same structured-metadata treatment if a refactor
  becomes warranted.
- **Memory-backend configuration** (Hindsight standup brief): per-backend config fields
  could follow the same shape, though config classes are typically static enough that the
  benefit is smaller.

Each context defines its own metadata class (`LoreField`, `ImporterSetting`,
`BackendConfigField`, etc.) — different data needs, same shape. **Shared empty marker base
class** (`AnnotationMetadata` or similar) is worth having, even though all per-context
fields live in the subclasses:

```python
class AnnotationMetadata:
    """Marker base for Raven parameter-metadata descriptors used in Annotated[T, M].

    Subclasses (Param for tools, Setting for postproc, LoreField for lorebook,
    etc.) define their own fields; the base exists solely so tooling can
    identify Raven metadata via isinstance/issubclass without maintaining a
    hand-curated registry.
    """

@dataclass
class Param(AnnotationMetadata):
    description: str = ""
    constraints: dict = field(default_factory=dict)

@dataclass
class Setting(AnnotationMetadata):
    description: str = ""
    range: Optional[tuple[float, float]] = None
    # ...etc per §appendix
```

What the marker base buys, even with zero fields:

- **Mixed `Annotated[]` payloads.** `Annotated[str, Param("..."), SomeLib.Tag("...")]` is
  legal; when walking annotations, `isinstance(item, AnnotationMetadata)` cleanly identifies
  "this is ours" without having to enumerate every Raven metadata subclass.
- **Codebase-wide introspection.** A future schema-validation or documentation tool that
  wants to enumerate "all Raven parameter-metadata declarations" has one hook
  (`issubclass(cls, AnnotationMetadata)`) rather than a hand-maintained class registry that
  drifts out of date when new metadata classes are added.
- **Honest typing.** `Param`, `Setting`, `LoreField` genuinely *are* instances of the same
  kind-of-thing ("Raven parameter-metadata descriptors attached via `Annotated[]`"); the
  category is real, encoding it as `is-a` is type honesty, not over-abstraction.

**The line worth not crossing**: putting *fields* on the base. The moment the base carries
`description: str = ""` because "every metadata kind needs description," you've committed
every subclass to that field — and the next person to add metadata reaches for the base for
"oh I should add units there too, it's universal, right?" That's where kitchen-sink drift
starts. **Empty marker yes; shared fields no.**
