#!/usr/bin/env python
"""CLI shell for the BibTeX importer.

This is the `raven-importer` console-script entry point. It parses CLI args
and configures logging *before* importing `raven.visualizer.importer` so the
import-time records reach the configured handlers (and the optional
``--log <path>`` mirror file). The library module itself stays free of
argparse and `logsetup.configure` — the visualizer GUI imports it as a
library and configures logging on its own.

See `briefs/logsetup-fleet-wide.md` for the dual-use split rationale.
"""

__all__ = ["main"]

import argparse
import logging
import sys

from .. import __version__


def main() -> None:
    parser = argparse.ArgumentParser(description="""Convert BibTeX file(s) into a Raven-visualizer dataset file.""",
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-v', '--version', action='version', version=('%(prog)s ' + __version__))
    parser.add_argument(dest="output_filename", type=str, metavar="out", help="Output, Raven-visualizer dataset file")
    parser.add_argument(dest="input_filenames", nargs="+", default=None, type=str, metavar="bib", help="Input, BibTeX file(s) to parse")
    parser.add_argument('--log', metavar='PATH', default=None,
                        help='mirror stderr log to this file (overwritten each run)')
    parser.add_argument('--log-level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='root logger level (default: INFO)')
    opts = parser.parse_args()

    if opts.output_filename.endswith(".bib"):
        print(f"Output filename '{opts.output_filename}' looks like an input filename. Cancelling. Please check usage summary by running this program with the '-h' (or '--help') option.")
        sys.exit(1)

    from ..common import logsetup
    logsetup.configure(level=getattr(logging, opts.log_level),
                       logfile=opts.log)
    logger = logging.getLogger(__name__)

    logger.info(f"Raven-importer version {__version__} starting.")
    logger.info("Loading libraries...")
    from unpythonic import timer
    with timer() as tim:
        from ..client import api
        from ..client import config as client_config
        from . import config as visualizer_config
        from . import importer
    logger.info(f"    Done in {tim.dt:0.6g}s.")

    api.initialize(raven_server_url=client_config.raven_server_url,
                   raven_api_key_file=client_config.raven_api_key_file)

    logger.info("Settings (for LOCAL models):")
    logger.info(f"    Embedding model: {visualizer_config.embedding_model}")
    logger.info(f"        Dimension reduction method: {visualizer_config.vis_method}")
    logger.info(f"    Extract keywords: {visualizer_config.extract_keywords}")
    logger.info(f"        NLP model (spaCy): {visualizer_config.spacy_model}")
    logger.info(f"    Summarize via LLM: {visualizer_config.summarize}")

    import traceback
    try:
        with timer() as tim:
            importer.import_bibtex(None, opts.output_filename, *opts.input_filenames)
    except Exception:
        logger.warning(f"Error after {tim.dt:0.6g}s total:")
        traceback.print_exc()
    else:
        logger.info(f"All done in {tim.dt:0.6g}s total.")


if __name__ == "__main__":
    main()
