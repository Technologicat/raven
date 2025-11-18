"""Dehyphenate text.

This uses character-level contextual embeddings by Flair-NLP, via the `dehyphen` package.

This can work either standalone, or with Raven-server, using its `sanitize` module.
"""

from .. import __version__

import argparse
import pathlib
import sys

from ..client import api
from ..client import config as client_config
from ..client import mayberemote
from ..common import utils as common_utils
from ..visualizer import config as visualizer_config

def main() -> None:
    parser = argparse.ArgumentParser(description="""Dehyphenate text bro-ken by hyp-he-na-tion. (You can configure the model in `raven.visualizer.config`.)""",
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(dest="filenames", nargs="*", default=None, type=str, metavar="txt", help="Text file(s) to dehyphenate. Defaults to reading input from stdin.")
    parser.add_argument("-o", "--output-suffix", dest="output_suffix", default=None, type=str, metavar="out", help="Suffix for naming output files (file.txt -> file_out.txt). Only used if at least one filename is given. Default (if this option is not given) is to concatenate all output to stdout.")
    parser.add_argument("-j", "--join-paragraphs", dest="join_paragraphs", action="store_true", default=False, help="For each input, send all paragraphs together for processing. May cause paragraphs to run together in the output, but if your input text is REALLY broken and contains newlines at arbitrary places, that's often the only way. If your input has clean paragraph breaks (double newline), you'll getter better results without this option.")
    parser.add_argument('-v', '--version', action='version', version=('%(prog)s ' + __version__))
    parser.add_argument("-V", "--verbose", dest="verbose", action="store_true", default=False, help="Print progress messages (to stderr).")
    opts = parser.parse_args()

    if not opts.filenames:
        opts.filenames = [None]  # `None` -> `maybe_open` will open stdin instead.

    api.initialize(raven_server_url=client_config.raven_server_url,
                   raven_api_key_file=client_config.raven_api_key_file,
                   tts_playback_audio_device=client_config.tts_playback_audio_device,
                   stt_capture_audio_device=client_config.stt_capture_audio_device)

    # TODO: refactor: tools shouldn't load `visualizer_config`
    dehyphenator = mayberemote.Dehyphenator(allow_local=True,
                                            model_name=visualizer_config.dehyphenation_model,
                                            device_string=visualizer_config.devices["sanitize"]["device_string"])

    for input_filename in opts.filenames:
        if input_filename is not None:
            input_filename_absolute = pathlib.Path(input_filename).expanduser().resolve()
        else:
            input_filename_absolute = None

        if opts.output_suffix:
            output_filename = f"{input_filename_absolute.stem}_{opts.output_suffix}{''.join(input_filename_absolute.suffixes)}"  # "file.txt" -> "file_out.txt";  "file.foo.txt" -> "file_out.foo.txt"
            output_filename_absolute = input_filename_absolute.parent / output_filename
        else:
            output_filename = None
            output_filename_absolute = None

        if opts.verbose:
            input_location_str = "stdin" if input_filename_absolute is None else f"'{str(input_filename_absolute)}'"
            output_location_str = "stdout" if output_filename_absolute is None else f"'{str(output_filename_absolute)}'"
            print(f"Processing {input_location_str}. Writing to {output_location_str}.", file=sys.stderr)

        with common_utils.maybe_open(input_filename_absolute, "r", fallback=sys.stdin, encoding="utf-8") as input_file:
            text = input_file.read()
            if opts.join_paragraphs:
                text = dehyphenator.dehyphenate(text)
            else:
                text = text.split("\n\n")
                text = dehyphenator.dehyphenate(text)
                text = "\n\n".join(text)
            with common_utils.maybe_open(output_filename_absolute, "w", fallback=sys.stdout, encoding="utf-8") as output_file:
                output_file.write(text)
            if output_file is sys.stdout:
                output_file.flush()

if __name__ == "__main__":
    main()
