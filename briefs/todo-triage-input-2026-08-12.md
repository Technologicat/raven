# TODO triage: what the 2026-08-12 session changed

Input for the triage pass that is in flight, written down because it was produced during a working session
and would otherwise live only in that session's transcript. Everything here is about `TODO.md` and
`TODO_DEFERRED.md`, which were frozen for the duration — so nothing below has been applied.

**Cited by name, not by line number**, per `briefs/README.md`: several of these items have moved already,
and `TODO_DEFERRED.md` is about to get substantially shorter.

## Delete — the work is done

- **`TODO.md`, "[Low] Add lockfile so `raven-minichat` and `raven-librarian` can't run simultaneously"**.
  Done as `raven.common.datastorelock`, which locks the file rather than recording a PID, so the OS releases
  it on a crash and there is no stale lock to detect. Verified across two real processes in both frontends.

- **`TODO_DEFERRED.md`, "Headless scaffold mode for `ai_turn` (scriptable agent layer)"**. This *is* brief 15,
  now landed as `raven.librarian.agent` and closed into `briefs/researchers-night/done/`. The 2026-08-10
  todo-sweep already marked it SUPERSEDED.

- **`TODO_DEFERRED.md`, "Lazy `api.initialize` in `llmclient` and `hybridir`"**. Done: `test_scaffold.py`'s
  `importorskip` is gone, which was the entire cost the item recorded.

- **`TODO_DEFERRED.md`, "The DPG tests we have never run in CI"**. Done, and the open question it posed is
  answered. `dearpygui` and `mistletoe` are in `.github/workflows/requirements-ci.txt`; GLFW gets a context
  on a runner with no display server, on ubuntu, macOS and Windows alike. 2090 → 2147 passing per platform,
  so 57 tests that had only ever run on a dev machine now run on every push. Tests that *map* a window are
  unaffected: they carry the `gui` marker and need `--run-gui`.

## Amend — partly overtaken

- **`TODO_DEFERRED.md`, "Make the canned AI greeting optional"**. Its first bullet — the one it flagged as
  needing "a real fix, not a length tweak" — **is fixed**. `chat_controller._get_all_greeting_node_ids` now
  identifies a greeting by role as well as position, so a user message sitting beside the greetings is no
  longer taken for one. That fix landed for its own reasons (multi-root made HEAD able to rest on a system
  prompt node), and it was this item's largest blocker.

  Still standing, and still accurate: the assert in `chat_controller` that counts system prompt + greeting +
  first user message; `minichat`'s `len(node_id_history) < 4`; `chatutil.factory_reset_datastore`; and
  `appstate._refresh_greeting`, which is where "blank means omit" has to be honoured.

  One thing worth adding to the item: with no greeting, `_refresh_greeting` must point `new_chat_HEAD` at
  the system prompt node itself. That state — HEAD resting on a root — is now a supported one rather than an
  anomaly, which is what the role check and the one-step descent in `_descend_to_latest` were for.

- **`TODO_DEFERRED.md`, "Modernize the Librarian system prompt / character card"**. Add a pointer to
  `briefs/researchers-night/done/15_headless-agent-driver-brief.md`, final section: the argument that
  `setup_interaction_style` is three different kinds of thing at once — deployment facts, conversational
  manner, and the two backend facts that have since moved to per-turn injects — and therefore cannot move
  into the system prompt as a unit. That analysis is the reason the obvious fix is wrong, and it is the only
  copy.

## New

- **Mid-run LLM backend recovery for batch tools.** `raven-pdf2bib` and `raven-importer` now stop at startup
  on both bad backend states (unreachable, and reachable-with-no-model), but neither recovers from a backend
  that goes away mid-run: every remaining document fails. Deliberately not solved, because it needs
  decisions this session did not have — how long to wait, whether to resume or restart, what to do with the
  documents already written. Recorded at both call sites so the limit is met where it applies.

- **`chat_controller` is not importable without spaCy.** It reaches the full ML stack through the avatar
  client, so `test_chat_controller.py` skips on the module under test and its pure datastore helpers cannot
  run in CI. Same anti-pattern as the `llmclient` item above, one layer up — and now that the GUI tests do
  run in CI, this is the only thing keeping the controller out.

- **Audit what else does not belong in the wheel.** The `00_workfiles` exclusion took the wheel from 107 MB
  to 14.4 MB. What remains is less obviously wrong and wants a judgement call rather than a rule:
  `raven/avatar/assets` is 8.1 MB, over half the wheel, of which three backdrop PNGs are 5.3 MB
  (`cyberspace.png` alone is 2.75 MB); and `raven/vendor/anime4k/images/6486130.png` is a 1.04 MB upstream
  sample image. The backdrops *are* runtime assets, so the question is resolution and count, not whether to
  ship them.

- **`chattree.get_all_root_nodes` is a linear scan**, and multi-root calls it from more places than the
  single-root design did — the cleanup sweep, and the controller's button gating (memoized there, and
  filtered against the live nodes because a card can now be deleted). Deliberately not optimized: we are
  orders of magnitude from where it bites. The shape of the answer, if it ever does, is an index of roots
  maintained by `create_node` / `delete_node`. Recorded here because its only other home is now an archived
  brief.

## Not Raven's list

- **An `unpythonic` skill**, for `~/.claude`. This session lost time to `flatten` returning a lazy generator
  that membership tests consumed — four button gates asking one list, the first answered correctly and the
  rest from the leftovers. "The iterable utilities are lazy wherever they can be" is the kind of thing a
  skill would have said at write time.
