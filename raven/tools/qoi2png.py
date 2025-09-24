"""Convert QOI image file(s) (Quite OK Image) to PNG.

This is mainly useful for re-encoding an avatar video stream recorded in QOI format,
since e.g. ImageMagick's `convert` tool does not support QOI.

The avatar video streaming uses the QOI format by default, because it is lossless,
much faster than PNG (encodes ~30x faster), and compresses almost as tightly.
"""

from .. import __version__

import argparse
import os

import PIL.Image
import qoi

import numpy as np

from unpythonic import timer, uniqify

def main() -> None:
    parser = argparse.ArgumentParser(description="""Convert QOI (Quite OK Image) image file(s) to PNG.""",
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(dest="filenames", nargs="+", default=None, type=str, metavar="myimage.qoi", help="Image file(s) to convert")
    parser.add_argument('-v', '--version', action='version', version=('%(prog)s ' + __version__))
    parser.add_argument("-V", "--verbose", dest="verbose", action="store_true", default=False, help="Print progress messages (useful with lots of files).")
    opts = parser.parse_args()

    with timer() as tim:
        input_filenames = list(uniqify(opts.filenames))
        for input_filename in input_filenames:
            dirname = os.path.dirname(input_filename)
            basename_without_extension = os.path.splitext(os.path.basename(input_filename))[0]
            output_filename = os.path.join(dirname, f"{basename_without_extension}.png")
            if opts.verbose:
                print(f"{input_filename} -> {output_filename}")

            # Read and decode input file
            with open(input_filename, "rb") as image_file:
                image_data = image_file.read()
            image_rgba = qoi.decode(image_data)  # -> uint8 array of shape (h, w, c)

            # Load the RGBA array to PIL
            if np.shape(image_rgba)[2] == 4:  # RGBA?
                pil_image = PIL.Image.fromarray(image_rgba, mode="RGBA")
            else:  # RGB
                pil_image = PIL.Image.fromarray(image_rgba)

            # Save output file
            pil_image.save(output_filename)
    if opts.verbose:
        plural_s = "s" if len(input_filenames) != 1 else ""
        average_time_str = f" (average {tim.dt / len(input_filenames):0.6g}s per image)" if len(input_filenames) > 1 else ""
        print(f"Converted {len(input_filenames)} image{plural_s} in {tim.dt:0.6g}s{average_time_str}.")

if __name__ == "__main__":
    main()
