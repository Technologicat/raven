"""Does loading a GGUF tokenizer beside the RAG index slow both down? Times each alone, then both together.

Usage: python contention.py RAG_INDEX_DIR GGUF_FILE

RAG_INDEX_DIR must be a *copy* of an index directory: constructing a `HybridIR` opens its ChromaDB store,
which may write. Raven-server must be running, as for the app. Prints one RESULT line.
"""
import pathlib
import sys
import threading
import time

from raven.client import api, config as client_config
from raven.librarian import gguftokenizer, hybridir

index_dir, gguf_file = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
api.initialize(raven_server_url=client_config.raven_server_url, raven_api_key_file=client_config.raven_api_key_file)

def build_index() -> float:
    t0 = time.perf_counter()
    hybridir.HybridIR(index_dir)
    return time.perf_counter() - t0

def load_tokenizer(out: list) -> None:
    t0 = time.perf_counter()
    loaded = gguftokenizer.load(gguf_file) is not None  # no cache_dir: always the full build
    out.append((time.perf_counter() - t0, loaded))

build_index()  # warm the page cache and the imports, so the runs below compare like with like
index_alone = build_index()
out = []
load_tokenizer(out)
tokenizer_alone, loaded = out[0]
out = []
thread = threading.Thread(target=load_tokenizer, args=(out,))
thread.start()
index_together = build_index()
thread.join()
print(f"RESULT index alone {index_alone:.2f} s; tokenizer alone {tokenizer_alone:.2f} s (loaded: {loaded}); "
      f"index with the tokenizer loading beside it {index_together:.2f} s, the tokenizer then {out[0][0]:.2f} s")
