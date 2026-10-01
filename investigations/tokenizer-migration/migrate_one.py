"""Open one RAG index, which migrates it to the current tokenizer, and report how long that took.

Usage: python migrate_one.py <index directory>

Opens the index with `HybridIR` directly, so nothing reconciles it against a documents directory. Needs a
Raven-server answering at the configured URL, which does the tokenizing. Prints one `RESULT:` line.
"""

import json
import logging
import pathlib
import sys
import time

from raven.client import api
from raven.client import config as client_config
from raven.librarian import hybridir

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")


def main() -> None:
    api.initialize(raven_server_url=client_config.raven_server_url,
                   raven_api_key_file=client_config.raven_api_key_file)
    base = pathlib.Path(sys.argv[1])
    data_file = base / "fulldocs" / "data.json"
    model = json.loads(data_file.read_text(encoding="utf-8"))["embedding_model_name"]

    t0 = time.monotonic()
    hybridir.HybridIR(datastore_base_dir=base, embedding_model_name=model, local_model_loader_fallback=False)
    dt = time.monotonic() - t0

    after = json.loads(data_file.read_text(encoding="utf-8"))
    chunks = sum(len(doc["chunks"]) for doc in after["documents"].values())
    print(f"RESULT: opened in {dt:.1f} s; version now {after.get('tokenizer_version')}; {chunks} chunks; "
          f"{1000 * dt / chunks:.1f} ms per chunk")


if __name__ == "__main__":
    main()
