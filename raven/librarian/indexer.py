"""Build or refresh Librarian's RAG document index, without starting a GUI.

Librarian ingests whatever is in its documents directory, and until this existed the only thing that could
perform that ingestion was the desktop app. That coupled a batch job to a desktop session and to the
frontend being in a runnable state — an unrelated GUI-side breakage would block an indexing run that has
nothing to do with the GUI. It also made swapping corpora a manual ritual rather than a command, which is
the shape of thing that quietly discourages measuring against a second corpus.

The indexing itself is not implemented here. `hybridir.setup` already rescans on construction, reconciling
the index against the directory — adding what is new, re-chunking what changed, dropping what is gone — and
`hybridir.HybridIR.commit` already reports per-chunk progress. This module is a front end over those, plus
the part that is genuinely missing from a library used only by long-lived apps: a way to *wait* for the
work to finish and then exit.

The configured-defaults opener that goes with it, `hybridir.open_document_store`, lives beside `setup` in
`hybridir` rather than here: every frontend needs it, and a CLI module is a strange place for the other
three to import it from.

Note what "refresh" means: this reconciles, it does not rebuild. A corrupt index is not repaired by running
this again — delete the index directory and re-run to get a clean build.
"""

__all__ = ["wait_for_indexing", "main"]

from .. import __version__

import argparse
import sys
import time
from typing import Callable, Optional

from ..client import api as client_api
from ..client import config as client_config
from . import config as librarian_config
from . import hybridir

# How often to sample indexing state while waiting. Fast enough that the progress line looks live, slow
# enough that the poll costs nothing next to the embedding work it is watching.
POLL_SECONDS = 0.5

# Consecutive quiet samples required before declaring the run finished. The rescan is dispatched to a
# background task, so there is a window at startup where nothing is indexing *yet*; requiring several
# quiet samples in a row rides over that without needing to observe the busy edge at all.
SETTLED_POLLS = 6


def wait_for_indexing(retriever: hybridir.HybridIR,
                      on_progress: Optional[Callable[[str], None]] = None) -> None:
    """Block until `retriever` has been quiet for `SETTLED_POLLS` consecutive samples.

    `on_progress`: called with the current progress string each time it changes. `None` to stay silent.

    There is no "indexing finished" event to await — the apps that use `hybridir` never need one, because
    they keep running. Hence polling, and hence `SETTLED_POLLS`: a single quiet sample can just as well
    mean the background rescan has not started yet.

    **Busy means "any pending work", not "currently committing".** `is_indexing` alone is the wrong
    predicate and fails in the worst possible direction: it reports whether the retriever is inside
    `commit()`, which is False throughout the ingest phase while documents are being read and their text
    extracted. On a corpus of 1268 PDFs that phase runs for minutes, so a waiter watching only
    `is_indexing` sees a quiet start, returns, and lets the caller exit — reporting success while
    hundreds of documents are still queued, and leaving them to die against a shut-down executor. Hence
    also `has_pending_work`, which covers the ingest queue.
    """
    last = ""
    settled = 0
    while settled < SETTLED_POLLS:
        time.sleep(POLL_SECONDS)
        busy = retriever.is_indexing() or hybridir.has_pending_work()
        report = retriever.get_indexing_progress_text()
        if on_progress is not None and report and report != last:
            on_progress(report)
            last = report
        settled = 0 if busy else settled + 1


def main() -> None:
    parser = argparse.ArgumentParser(description="""Build or refresh Raven-librarian's RAG document index. Indexes the configured documents directory unless another is given, then exits. (Configure in `raven/librarian/config.py`.)""",
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(dest="docs_dir", nargs="?", default=None, type=str, metavar="dir", help=f"Directory of documents to index. Default is the configured document store ('{librarian_config.llm_docs_dir}').")
    parser.add_argument("-d", "--db-dir", dest="db_dir", default=None, type=str, metavar="dir", help=f"Directory to write the search indices to. Default is the configured index store ('{librarian_config.llm_database_dir}'). Note this is overwritten in place, not versioned.")
    parser.add_argument("-r", "--recursive", dest="recursive", action="store_true", default=None, help="Descend into subdirectories of the documents directory.")
    parser.add_argument("-R", "--no-recursive", dest="recursive", action="store_false", help="Do not descend into subdirectories. Overrides the configured default.")
    parser.add_argument('-v', '--version', action='version', version=('%(prog)s ' + __version__))
    parser.add_argument("-q", "--quiet", dest="quiet", action="store_true", default=False, help="Print only the final summary, not per-document progress.")
    parser.add_argument("--server-url", dest="server_url", default=None, type=str, metavar="url", help=f"Raven server to talk to, overriding the configured one (default: '{client_config.raven_server_url}'). Indexing computes embeddings, so this is where that happens.")
    opts = parser.parse_args()

    raven_server_url = opts.server_url if opts.server_url is not None else client_config.raven_server_url
    client_api.initialize(raven_server_url=raven_server_url,
                          raven_api_key_file=client_config.raven_api_key_file)

    try:
        retriever, scanner = hybridir.open_document_store(docs_dir=opts.docs_dir,
                                                         db_dir=opts.db_dir,
                                                         recursive=opts.recursive)
    except Exception as exc:  # noqa: BLE001 -- the CLI's job is to report, not to add a traceback
        print(f"raven-indexer: could not open the document store: {type(exc)}: {exc}", file=sys.stderr)
        sys.exit(1)

    # Overwrite one line when a human is watching; emit a line per change when the output is a log or a
    # pipe, where a carriage return renders as a single unreadable line.
    interactive = sys.stdout.isatty()

    def report(text: str) -> None:
        if interactive:
            print(f"\r\033[K{text}", end="", flush=True)
        else:
            print(text, flush=True)

    started = time.monotonic()
    interrupted = False
    try:
        wait_for_indexing(retriever, on_progress=None if opts.quiet else report)
    except KeyboardInterrupt:
        interrupted = True
        print("\nraven-indexer: interrupted; finishing the current document and saving the partial index…",
              file=sys.stderr)
    finally:
        # Stop the watcher first, so nothing new is queued while we drain; then wait for the in-flight
        # commit to leave its per-document loop and run its partial-save tail.
        #
        # The second call is what makes an interrupted run resumable, and leaving it out is not a small
        # loss. The vector index is written *inside* the loop, per document, while the keyword index and
        # the document store are written *once*, at the tail — so a commit abandoned before that tail
        # leaves the two halves disagreeing, with the vector index holding chunks for documents the store
        # has never heard of. Re-running then re-ingests the whole batch rather than resuming, which on a
        # bulk import is an hour of embedding thrown away.
        scanner.shutdown()
        hybridir.shutdown()

    if interrupted:
        print("raven-indexer: partial index saved; re-run to continue from here.", file=sys.stderr)
        sys.exit(130)

    if interactive and not opts.quiet:
        print()
    with retriever.datastore_lock:
        n_documents = len(retriever.documents)
    plural_s = "s" if n_documents != 1 else ""
    print(f"raven-indexer: {n_documents} document{plural_s} indexed in {time.monotonic() - started:.1f}s.")


if __name__ == "__main__":
    main()
