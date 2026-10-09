"""Where a GGUF tokenizer build spends its time. Usage: python profile_tokenizer.py GGUF_FILE"""
import cProfile
import pathlib
import pstats
import sys

from raven.librarian import gguftokenizer

profile = cProfile.Profile()
profile.enable()
gguftokenizer.load(pathlib.Path(sys.argv[1]))  # no cache_dir: always the full build
profile.disable()
pstats.Stats(profile).sort_stats("cumulative").print_stats(18)
