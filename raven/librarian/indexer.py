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

`open_document_store` is the other half, and is the more reusable one: the frontends each carry their own
copy of the same six-argument `hybridir.setup` call, and a third copy is how the three drift apart. It is
public so `app` and `minichat` can adopt it.

Note what "refresh" means: this reconciles, it does not rebuild. A corrupt index is not repaired by running
this again — delete the index directory and re-run to get a clean build.
"""

__all__ = ["open_document_store", "wait_for_indexing", "main"]

from .. import __version__

import argparse
import concurrent.futures
import pathlib
import sys
import time
from typing import Callable, Optional, Tuple, Union

from ..client import api as client_api
from ..client import config as client_config
from ..common import docextract
from . import config as librarian_config
from . import hybridir

# How often to sample indexing state while waiting. Fast enough that the progress line looks live, slow
# enough that the poll costs nothing next to the embedding work it is watching.
POLL_SECONDS = 0.5

# Consecutive quiet samples required before declaring the run finished. The rescan is dispatched to a
# background task, so there is a window at startup where nothing is indexing *yet*; requiring several
# quiet samples in a row rides over that without needing to observe the busy edge at all.
SETTLED_POLLS = 6


def open_document_store(docs_dir: Union[pathlib.Path, str, None] = None,
                        db_dir: Union[pathlib.Path, str, None] = None,
                        recursive: Optional[bool] = None,
                        executor: Optional[concurrent.futures.Executor] = None) -> Tuple[hybridir.HybridIR, hybridir.HybridIRFileSystemEventHandler]:
    """Open Librarian's RAG document store, applying the configured defaults. Rescans on construction.

    Every argument defaults to the corresponding `raven.librarian.config` setting, so calling this with no
    arguments opens exactly the store the chat clients open. Pass one to point elsewhere — which is what
    an indexing run over a corpus that is not the configured one needs.

    `docs_dir`: Directory holding the documents. Defaults to `llm_docs_dir`.
    `db_dir`: Directory holding the search indices. Defaults to `llm_database_dir`.
    `recursive`: Whether to descend into subdirectories. Defaults to `llm_docs_dir_recursive`.
    `executor`: Passed to `hybridir.setup`, which see.

    Returns `(retriever, scanner)`, as `hybridir.setup` does.

    The extractor is `docextract.ALL_FORMATS` narrowed to `llm_docs_exts`, so this ingests what Librarian
    ingests. Widening it here would build an index the chat clients would then disagree with.

    `local_model_loader_fallback` is off: Librarian requires Raven-server for other reasons anyway, and a
    silent fall back to loading the embedding model in-process turns a server-down misconfiguration into a
    slow run that quietly used a different device.
    """
    docs_dir = pathlib.Path(docs_dir if docs_dir is not None else librarian_config.llm_docs_dir).expanduser().resolve()
    db_dir = pathlib.Path(db_dir if db_dir is not None else librarian_config.llm_database_dir).expanduser().resolve()
    if recursive is None:
        recursive = librarian_config.llm_docs_dir_recursive
    return hybridir.setup(docs_dir=docs_dir,
                          recursive=recursive,
                          db_dir=db_dir,
                          extractor=docextract.ALL_FORMATS.restricted_to(librarian_config.llm_docs_exts),
                          embedding_model_name=librarian_config.qa_embedding_model,
                          local_model_loader_fallback=False,
                          executor=executor)


def wait_for_indexing(retriever: hybridir.HybridIR,
                      on_progress: Optional[Callable[[str], None]] = None) -> None:
    """Block until `retriever` has been quiet for `SETTLED_POLLS` consecutive samples.

    `on_progress`: called with the current progress string each time it changes. `None` to stay silent.

    There is no "indexing finished" event to await — the apps that use `hybridir` never need one, because
    they keep running. Hence polling `is_indexing`, and hence `SETTLED_POLLS`: a single quiet sample can
    just as well mean the background rescan has not started yet.
    """
    last = ""
    settled = 0
    while settled < SETTLED_POLLS:
        time.sleep(POLL_SECONDS)
        busy = retriever.is_indexing()
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
    opts = parser.parse_args()

    client_api.initialize(raven_server_url=client_config.raven_server_url,
                          raven_api_key_file=client_config.raven_api_key_file)

    try:
        retriever, scanner = open_document_store(docs_dir=opts.docs_dir,
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
    try:
        wait_for_indexing(retriever, on_progress=None if opts.quiet else report)
    except KeyboardInterrupt:
        # The commit is per-document and the index auto-persists, so an interrupted run leaves a valid,
        # partial index rather than a corrupt one. Re-running resumes: the documents already in the index
        # are reconciled as unchanged.
        print("\nraven-indexer: interrupted; the partial index is valid, re-run to continue.", file=sys.stderr)
        sys.exit(130)
    finally:
        scanner.shutdown()

    if interactive and not opts.quiet:
        print()
    with retriever.datastore_lock:
        n_documents = len(retriever.documents)
    plural_s = "s" if n_documents != 1 else ""
    print(f"raven-indexer: {n_documents} document{plural_s} indexed in {time.monotonic() - started:.1f}s.")


if __name__ == "__main__":
    main()
