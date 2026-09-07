# Making a backend failure happen at a chosen moment

**Two instruments, and they answer different questions.** `faultproxy.py` makes an LLM backend *misbehave*
— answer an error, stall, go quiet mid-stream — which needs it to speak HTTP. `tcprelay.py` makes a server
*disappear*, which needs it not to: it copies bytes and knows nothing about what they mean.

The distinction is not fussiness. An HTTP proxy rebuilds each request and response from parts, so it drops
every header it was not written to copy — and Raven-server puts real data in headers (`X-Full-Size` and
`X-Crop` on avatar frames, word timestamps and phonemes on TTS audio). Pointed at Raven-server,
`faultproxy.py` gets as far as loading the avatar and then hangs in TTS warmup. Use the relay for
Raven-server, and either for an OpenAI-compatible backend.

## `tcprelay.py` — the server goes away

```bash
python tcprelay.py --port 8999 --upstream localhost:5100
raven-librarian --server-url http://127.0.0.1:8999
```

Kill it and the server has vanished; run it again and the server is back. The real Raven-server keeps
running throughout, which matters when it holds several gigabytes of models that take minutes to load.

**What it found (2026-09-07).** Librarian ran at full frame rate for the rest of the session after
Raven-server went down. `DPGAvatarRenderer.pause` told the server to stop the avatar *before* clearing
`animator_running`, and on this path the server is the one that just disappeared — so the call raised and
the flag stayed set. Librarian's idle-framerate throttle reads that flag, so it never engaged again. The
same block could skip the assignment a second way, `nonexistent_ok` leaving at the first missing widget.
Both are fixed, and `raven/client/tests/test_avatar_renderer.py` pins them.

It also gave the Raven-server status row its live test: down, up, and the linger before the row stands down.

## `faultproxy.py` — the backend misbehaves

**Question.** What does Librarian do when the LLM backend fails *while the user is somewhere else* — on
another branch, or navigating back at the wrong instant? The interesting cases all turn on the **timing** of
the failure relative to what the user is doing, and waiting for a real backend to fail on cue is not a test.

**Instrument.** `faultproxy.py` sits between Librarian and the real backend, forwards everything verbatim,
and fails `/v1/chat/completions` on command. That makes the moment the test's choice rather than the
backend's.

### Using it

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

### What it found (2026-08-27)

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
