"""`raven-importer` with LLM cluster keywords, the phrase evidence switched on or off for this run only.

Usage, with the importer's own arguments after the first:

    python investigations/cluster-keywords/import_with_evidence.py on|off OUT.pickle IN.bib [...]

Sets `clusters_keyword_method = "llm"` and `clusters_llm_keyword_evidence` in-process, so the machine's
configuration and overrides file stay as they are, then runs the importer's command line.
"""

import sys

if len(sys.argv) < 2 or sys.argv[1] not in ("on", "off"):
    sys.exit(__doc__)
evidence = sys.argv.pop(1) == "on"

from raven.visualizer import config as visualizer_config  # noqa: E402 -- after the argument check

visualizer_config.clusters_keyword_method = "llm"
visualizer_config.clusters_llm_keyword_evidence = evidence

from raven.visualizer import importer_cli  # noqa: E402 -- after the configuration is set

importer_cli.main()
