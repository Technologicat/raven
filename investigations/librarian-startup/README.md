# Where Librarian's startup time goes

**The question:** Raven-librarian took around twenty seconds to reach its render loop, and starting with a
blank `chat.json` did not change that (maintainer, 2026-10-08). Where does the time go?

**The answer, measured 2026-10-09 on the work machine:** not in the chat datastore at all. Three steps, two of
them slowed by the third running beside them.

## What was measured

Configuration: the `hydrogen` RAG index (11,974 documents, 31,600 chunks; a 145 MB `data.json` and a 67 MB
`embeddings.npz`), and a local tokenizer read from `Qwen3.8-27B-UD-Q4_K_XL.gguf` (a 248,320-token vocabulary).
Raven-server running, the LLM backend on the personal machine.

From an unpatched launch's own log (`--log-level DEBUG`), first line to the render loop: 20.8 s.

| step | in the app | alone | |
|---|---|---|---|
| imports | 3.9 s | | |
| `HybridIR._load_datastore` | 8.8–9.5 s | 2.5 s | 2.5 s of the 8.8 s was the cycle collector (~2,300 collections) |
| `HybridIRFileSystemEventHandler.rescan` | 4.4 s | 0.8 s | per-file Python work over 12k paths |
| GGUF tokenizer load, on a background thread | 16.5 s | 8.6 s | overlaps both of the above |

**The tokenizer load is the cause of most of it.** During both slow steps exactly one other thread was alive,
`llmclient tokenizer load` (`boot_probe.py`). Run side by side outside the app (`contention.py`), each slowed
the other: the index load went from 2.5 s to 6.1 s and the tokenizer from 8.6 s to 11.8 s. Both are
Python-level work, so they compete for the GIL.

**And nearly all of the tokenizer's 8.6 s is in the `gguf` library** (`profile_tokenizer.py`):
`GGUFReader.__init__` parses every metadata field of the file up front, in Python — about 1.2 million element
reads through NumPy memmap views for the vocabulary and the merges — while `gguftokenizer` uses four fields.

## What was done

- **The built tokenizer is cached** (`llm_tokenizer_cache_dir`, `gguftokenizer.load(cache_dir=...)`), keyed on
  the `.gguf`'s resolved path, size and mtime. A cached tokenizer goes through the same round-trip and backend
  checks as a fresh build. Measured: the build 8.0 s, a load from the cache 0.43 s (a 20 MB file), and the two
  give identical ids on 30k characters of source.
- **The cycle collector is paused while the RAG datastore is read.** Standalone the saving is small (2.3 s to
  2.0 s, best of three), since the heap is small; in the app, collection time during the load went from 2.5 s
  to zero.
- **The rescan needed nothing.** Without the tokenizer thread beside it, it took 0.8 s.

Measured with `boot_probe.py`, which imports `hybridir` early, so its offsets omit most of the imports:

| | before | first launch (builds the cache) | later launches |
|---|---|---|---|
| render loop starts | +15.4 s | +14.6 s | +5.3 s |
| `_load_datastore` | 8.8 s | 5.9 s | 2.0 s |
| rescan | 4.4 s | 5.3 s | 0.8 s |
| tokenizer ready | +17.6 s | +17.4 s | +2.6 s |

## Scripts

- `boot_probe.py [raven-librarian options]` — runs the app with timers around `_load_datastore` and `rescan`,
  reporting wall time, time in the cycle collector and the live threads. Maps a window.
- `contention.py RAG_INDEX_DIR GGUF_FILE` — the index load alone, the tokenizer build alone, then both at once.
  Give it a **copy** of an index directory: opening ChromaDB may write.
- `profile_tokenizer.py GGUF_FILE` — `cProfile` of one tokenizer build, without the cache.
