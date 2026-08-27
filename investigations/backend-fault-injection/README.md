# Making a backend failure happen at a chosen moment

**Question.** What does Librarian do when the LLM backend fails *while the user is somewhere else* — on
another branch, or navigating back at the wrong instant? The interesting cases all turn on the **timing** of
the failure relative to what the user is doing, and waiting for a real backend to fail on cue is not a test.

**Instrument.** `faultproxy.py` sits between Librarian and the real backend, forwards everything verbatim,
and fails `/v1/chat/completions` on command. That makes the moment the test's choice rather than the
backend's.

## Using it

```bash
python faultproxy.py --port 8998 --upstream http://localhost:1234
raven-librarian --backend-url http://127.0.0.1:8998 --log-level INFO --log /tmp/librarian.log
```

Then, from anywhere, write a mode into the control file — it is read per request, so a running turn is
unaffected and the *next* one obeys:

| `/tmp/faultproxy.mode` | What the backend appears to do |
|---|---|
| `pass` (or absent) | nothing unusual — forwards upstream |
| `error` | answers 200 and then an SSE `event: error`, the way LM Studio reports a fault |
| `error:<secs>` | the same, after stalling — long enough to navigate away first |
| `hang:<secs>` | headers, then silence, then close: a backend that stops talking without saying so |

`error:<secs>` is the one that matters. The stall is the window in which the test does something else, so
that the failure lands while the user is elsewhere.

## What it found (2026-08-27)

**A streaming message widget was left on screen after its turn ended away from the view.** `on_done` bails
early when HEAD has moved off the turn's branch — correctly, so a finished reply does not intrude on the
chat the user has moved to — but the call that demolishes the turn's streaming widget sat *after* that
guard. So the widget stayed published with nothing left to show, its content having become a stored node,
and the view's rebuild then re-attached the empty husk whenever the user came back to that branch.

It presents as an AI message with its icon and nothing else, for a second or two. Juha saw it during a
driven run; it is too brief for a screenshot taken on a multi-second delay to catch, which is why the
instrument mattered more than the observation method.

Fixed by demolishing on the away path too, and by a backstop in the turn's `finally` for the paths that
never reach `on_done` at all.

## Notes

- **The proxy buffers.** `_forward` reads upstream in 1024-byte blocks, so a streamed reply reaches the app
  in lumps rather than token by token. Fine for fault injection, wrong for anything measuring streaming
  latency or first-token time — use the real backend for those.
- It listens on localhost only and forwards to whatever `--upstream` says. There is no authentication,
  because there is nothing here that should ever run outside a development machine.
- The control file is read per request and never written by the proxy, so `echo pass > /tmp/faultproxy.mode`
  is always enough to get back to normal.
