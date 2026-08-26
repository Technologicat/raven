# Aborting an in-flight LLM request from another thread

**Question.** Librarian sometimes has a backend request in flight that it no longer wants: a speculative
context prefill when the user starts typing, or an AI turn the user has interrupted. Can that request be
abandoned *from another thread*, promptly enough to free the backend for the work the user is actually
waiting for?

**Answer.** Yes, but not by the obvious route. `requests.Response.close()` — the mechanism the deferred TODO
assumed — neither wakes the blocked reader nor returns to its caller. `socket.shutdown(SHUT_RDWR)` on the
underlying socket does both, immediately, and the backend drops the abandoned work.

Measured 2026-08-26, against a local stall server for the mechanism and LM Studio 0.4.20 (`qwen3.6-35b-a3b`,
one 24 GB card, 131072-token context) for the behaviour that matters in production.

## What the probes answer

| Script | Question |
|---|---|
| `probe_closers.py` | Which way of closing a streaming response actually aborts a blocked read — and whether the closer itself returns |
| `probe_phases.py` | On a real backend, does the long prompt-processing wait happen before or after the response headers — i.e. do we hold anything to abort? And does aborting work there? |
| `probe_backend_freed.py` | After an abort, is the backend actually free, or still chewing the abandoned prompt? |

## 1. Only `shutdown()` aborts a blocked read

`probe_closers.py`. A local HTTP server sends 200 and the SSE headers, then stalls; a reader thread blocks
in `iter_lines()`; a second thread applies one closer. Every ceiling is 15 s and the observation window is
1.5 s, so a failed approach costs seconds rather than minutes — an abort that works lands in milliseconds.

| Closer | Reader wakes? | Closer returns? |
|---|---|---|
| `response.close()` | no | **no** |
| `raw.close()` | no | **no** |
| `raw._fp.fp.close()` | no | **no** |
| `socket.close()` | no | yes |
| `_socket.socket.close()` | no | yes |
| `socket.shutdown(SHUT_RDWR)` | **yes, 0.00 s** | **yes** |

Those are Linux figures. **Windows inverts the last two rows**: `shutdown` does not wake the reader there
(Winsock cancels pending operations on `closesocket`, not on `shutdown`), and the C-level close does. See
the section below, and note that `socket.close()` is not the same call as `_socket.socket.close()`.

Two findings, and the second is the one that would have hurt.

**The `close()` family does not abort.** The blocked `recv` does not notice. `socket.close()` closes the
descriptor and the reader still does not wake, which is the classic result: the file object the reader is
blocked on holds its own reference.

**And the `close()` family blocks its *caller*, for as long as the read timeout allows.** An earlier run of
this probe with a 60 s timeout measured `response.close()` taking **59.86 s** to return. Called from a DPG
callback — which is where a Cancel button lives — that is a minute of frozen GUI. This is the reason the
implementation reaches past the public API to the socket rather than calling the method that looks right.

The aborted reader raises `requests.exceptions.ChunkedEncodingError: Response ended prematurely`.

### Why `close()` does nothing, and why that is not the reason it does not work

Two separate facts, and conflating them cost a CI cycle.

`socket.close()` does not close the socket here. urllib3 reads the response through a `makefile()` wrapper,
so the socket carries an outstanding io-ref — measured `_io_refs = 1`, `_closed = False` at the moment of
the abort — and Python's `close` then only marks the object closed, deferring the real `closesocket` until
that wrapper goes. The wrapper is the reader being interrupted, so it never goes.

That makes `close()` a no-op, and it is tempting to conclude that a *real* close would have worked.
It would not, on POSIX: `_socket.socket.close()` — the C-level close, which does reach the handle —
**still leaves the reader blocked**. Closing a descriptor does not interrupt a thread already inside `recv`
on it. Only `shutdown` does that.

Both halves matter on Windows, where `shutdown` does not cancel a pending operation and `closesocket` does:
there the deferral is what has to be bypassed, and the C-level close is the call that ends the wait.

## 2. The expensive wait is after the headers, so there is something to abort

`probe_phases.py`, run A. A cold ~90k-token prefill against LM Studio:

```
  0.125s  headers (status 200)
 18.630s  first data line
```

The response headers arrive in about a tenth of a second; the whole prompt-processing pass then happens
inside the body read. So for all but the first ~0.1 s of a request, the client holds a socket that
`shutdown()` can reach.

That 0.1 s window is real but small, and it closes on its own: an abort raised during it is remembered and
applied the moment the response exists.

**The first attempt at this measurement was wrong, and the way it was wrong is worth keeping.** Timing "the
first body byte" put the whole wait at 0.586 s and reported no long gap at all — because an OpenAI-style
stream emits a role delta immediately, long before the first token. A probe that stops at the first byte
measures the framing, not the work.

Run B aborts the same request 5 s in:

```
  0.029s  headers (status 200)
  5.029s  [aborter] shutdown() returned
  5.030s  -> ChunkedEncodingError: Response ended prematurely
```

Out at 5.03 s instead of 18.63 s, with the aborting thread returning immediately.

## 3. The backend really is freed

`probe_backend_freed.py`. Aborting the client side would be worth little if the backend kept processing the
abandoned prompt — the next real request would still queue behind it. It does not:

```
  5.056s  prefill aborted: ChunkedEncodingError
  0.990s  tiny follow-up request answered (status 200)
```

Against ~13.5 s of prompt processing still outstanding, a small request came back in under a second. Closing
the stream is how an OpenAI-compatible backend is told to stop, and that turns out to cover prompt
processing and not only token generation.

## Notes for whoever touches this next

- **The socket lives at `response.raw._fp.fp.raw._sock`**, which is four layers of private attribute across
  `requests`, `urllib3` and `http.client`. It needs a guarded accessor: a shape it does not recognize means
  no abort, never a crash. The alternative is the public `close()`, which was measured above not to work.
- **`ChunkedEncodingError` is how a successful abort presents.** `llmclient.invoke` already catches it and
  logs "Connection lost. Please check if your LLM backend is still alive", which is the right message for a
  backend that died and a misleading one for an abort we asked for. The two cases have to be told apart.
- **The context length ceiling is VRAM, not configuration.** A first attempt at a large prompt was rejected
  with `request (180010 tokens) exceeds the available context size (131072 tokens)`. Juha's note: at q4_0 KV
  and with Qwen's 3:1 DeltaNet, 24 GB does not hold much more than 128 Ki at this model size.
- Each probe takes an explicit backend URL and model, defaulting to the values above. They send real
  requests and occupy the backend for tens of seconds.

Discovered during Researchers' Night 2026 sprint work on the paired deferred items
"Librarian: in-flight AI turn bleeds into a new chat (turn-sequencing race)" and "Idle prefill fires even
when the HEAD's token count is already exact" (2026-08-26).
