"""Render a pose with THA3 for the pose editor, and turn any of the poser's outputs into an image to show.

The pose editor's output index chooses among the poser's outputs: the final image, and its intermediates. Those come
in several sizes, and only the four-channel ones are images; the rest are data, such as masks and flow fields.
`render` returns any of them as a displayable image of one size.

This module is part of the pose editor, and is licensed under the GNU AGPL, see `LICENSE`.
"""

__all__ = ["OUTPUTS", "output_labels",
           "to_display_image", "render"]

import numpy as np
import torch

from ...common.image import utils as imageutils
from ...common.video import compositor
from ...vendor.tha3.util import torch_linear_to_srgb


#: What THA3's poser outputs are, in order: `(network, output)`. The poser returns the outputs of its five networks
#: concatenated in this order (`Branch.all_outputs` in `tha3/poser/modes/*.py`, the same in all four modes), and each
#: network names its own outputs by `*_INDEX` constants (in `tha3/nn/`). The names here are those, lowercased.
OUTPUTS = (("editor", "color changed image"),  # the final image, output 0
           ("editor", "color change alpha"),
           ("editor", "color change image"),
           ("editor", "warped image"),
           ("editor", "grid change"),
           ("rotator", "direct image"),
           ("rotator", "warped image"),
           ("rotator", "grid change"),
           ("face morpher", "output image"),
           ("face morpher", "eye alpha"),
           ("face morpher", "eye color change"),
           ("face morpher", "iris mouth image 1"),
           ("face morpher", "iris mouth alpha"),
           ("face morpher", "iris mouth color change"),
           ("face morpher", "iris mouth image 0"),
           ("eyebrow combiner", "eyebrow image"),
           ("eyebrow combiner", "combine alpha"),
           ("eyebrow combiner", "eyebrow image no combine alpha"),
           ("eyebrow combiner", "morphed eyebrow layer"),
           ("eyebrow combiner", "morphed eyebrow layer alpha"),
           ("eyebrow combiner", "morphed eyebrow layer color change"),
           ("eyebrow combiner", "warped eyebrow layer"),
           ("eyebrow combiner", "morphed eyebrow layer grid change"),
           ("eyebrow decomposer", "eyebrow layer"),
           ("eyebrow decomposer", "eyebrow layer alpha"),
           ("eyebrow decomposer", "eyebrow layer color change"),
           ("eyebrow decomposer", "background layer"),
           ("eyebrow decomposer", "background layer alpha"),
           ("eyebrow decomposer", "background layer color change"))


def output_labels(n: int) -> list[str]:
    """Labels for the poser's `n` outputs, as `"9: face morpher, eye alpha"`, or bare numbers if `n` is not the count
    `OUTPUTS` describes (a model these names were not read from). `int(label.split(":")[0])` is the index either way.
    """
    if n != len(OUTPUTS):
        return [str(i) for i in range(n)]
    return [f"{i}: {network}, {name}" for i, (network, name) in enumerate(OUTPUTS)]


def to_display_image(output: torch.Tensor, size: int) -> torch.Tensor:
    """One of the poser's outputs, `[c, h, w]`, as an RGBA image of `size` × `size` pixels in [0, 1], sRGB.

    A four-channel output is an image in [-1, 1], linear intensity, and is converted as one. Any other is data, shown
    through `raven.common.image.utils.channels_to_rgba`. Either is then scaled to `size`, nearest-neighbour, so that
    a low-resolution intermediate shows its pixels as they are.

    Never modifies `output`.
    """
    # Out of place, because the poser's outputs are not ours to modify. The THA3 modes cache the eyebrow decomposer's
    # outputs while the input image stays the same, and some of those are among the outputs shown, so converting one
    # in place would change what every later render composes the face from.
    if output.shape[0] == 4:
        image = (output + 1.0) * 0.5  # [-1, 1] -> [0, 1]
        image = torch.cat([torch_linear_to_srgb(image[:3, :, :]), image[3:, :, :]])  # gamma correction, alpha as is
    else:
        image = imageutils.channels_to_rgba(output)
    if image.shape[1:] != (size, size):
        image = torch.nn.functional.interpolate(image.unsqueeze(0), size=(size, size), mode="nearest")[0]
    return image


def render(poser, base_image: torch.Tensor, cels: dict, celstack: list, pose: list, output_index: int, *,
           device: torch.device, dtype: torch.dtype, size: int) -> np.ndarray:
    """Blend `celstack` onto `base_image`, pose it, and return output `output_index` as a displayable image.

    `poser`: a THA3 poser.
    `base_image`: the character, `[c, h, w]`, in [0, 1], linear intensity, on `device`.
    `cels`: the character's cels, as `raven.common.video.compositor.render_celstack` takes them.
    `celstack`: which cels to blend, and how strongly.
    `pose`: the pose parameters.
    `output_index`: which of the poser's outputs to return; 0 is the final image.
    `device`, `dtype`: where and in which precision the poser runs.
    `size`: the side of the square image returned, in pixels.

    Returns a float NumPy array of shape `[size, size, 4]`, in [0, 1], sRGB, on the CPU.
    """
    posetensor = torch.tensor(pose, device=device, dtype=dtype)
    with torch.inference_mode():
        source_image = compositor.render_celstack(base_image, celstack, cels)  # a fresh tensor, ours to modify
        source_image.mul_(2.0)  # [0, 1] -> [-1, 1], the poser's range
        source_image.sub_(1.0)
        output = poser.pose(source_image, posetensor, output_index)[0].float()
        image = to_display_image(output, size)
        return image.permute(1, 2, 0).contiguous().cpu().numpy()  # [c, h, w] -> [h, w, c]
