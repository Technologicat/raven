"""`raven-importer` with LLM cluster keywords, the phrase evidence switched on or off for this run only.

Usage, with the importer's own arguments after the first one or three:

    python investigations/cluster-keywords/import_with_evidence.py [--method llm|hybrid] on|off OUT.pickle IN.bib [...]

Sets `clusters_keyword_method` (`"llm"` unless `--method` says otherwise) and `clusters_llm_keyword_evidence`
in-process, so the machine's configuration and overrides file stay as they are, then runs the importer's command
line.
"""

import sys

method = "llm"
if len(sys.argv) > 2 and sys.argv[1] == "--method":
    method = sys.argv[2]
    del sys.argv[1:3]
if method not in ("llm", "hybrid") or len(sys.argv) < 2 or sys.argv[1] not in ("on", "off"):
    sys.exit(__doc__)
evidence = sys.argv.pop(1) == "on"

from raven.visualizer import config as visualizer_config  # noqa: E402 -- after the argument check

visualizer_config.clusters_keyword_method = method
visualizer_config.clusters_llm_keyword_evidence = evidence

from raven.visualizer import importer_cli  # noqa: E402 -- after the configuration is set

importer_cli.main()
