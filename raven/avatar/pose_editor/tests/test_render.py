"""Tests for `raven.avatar.pose_editor.render`: any of the poser's outputs as an image of one size."""

import numpy as np
import pytest
import torch

from raven.avatar.pose_editor import render

SIZE = 512


class FakePoser:
    """Returns outputs of the shapes THA3's do (measured on the pose editor, 2026-11-13), the same tensors every call,
    as THA3's modes do for the eyebrow decomposer's outputs while the input image is unchanged."""

    def __init__(self):
        generator = torch.Generator().manual_seed(0)
        shapes = [(4, 512, 512), (2, 512, 512), (4, 256, 256), (2, 256, 256), (4, 192, 192), (1, 192, 192)]
        self.outputs = [torch.rand(1, *shape, generator=generator) * 2.0 - 1.0 for shape in shapes]
        self.seen_inputs = []

    def pose(self, image, pose, output_index):
        self.seen_inputs.append(image.clone())
        return self.outputs[output_index]


@pytest.fixture
def poser():
    return FakePoser()


def run(poser, output_index):
    base = torch.rand(4, SIZE, SIZE, generator=torch.Generator().manual_seed(1))
    return render.render(poser, base, cels={}, celstack=[], pose=[0.0] * 3, output_index=output_index,
                         device=torch.device("cpu"), dtype=torch.float32, size=SIZE)


class TestEveryOutputIsAnImageOfOneSize:
    @pytest.mark.parametrize("output_index", range(6))
    def test_each_output_comes_back_as_rgba_of_the_texture_size(self, poser, output_index):
        image = run(poser, output_index)
        assert image.shape == (SIZE, SIZE, 4)
        assert image.dtype == np.float32
        assert image.min() >= 0.0 and image.max() <= 1.0

    def test_data_outputs_are_opaque(self, poser):
        for output_index in (1, 3, 5):  # two- and one-channel
            assert np.all(run(poser, output_index)[:, :, 3] == 1.0)

    def test_a_small_output_is_enlarged_pixel_for_pixel(self, poser):
        image = run(poser, 5)  # 192 -> 512 is not a whole multiple, so check runs of equal pixels instead of blocks
        row = image[100, :, 0]
        assert len(np.unique(row)) <= 192, "more distinct values than the source had pixels: interpolated, not nearest"


class TestThePosersOutputsAreLeftAlone:
    def test_showing_an_output_does_not_change_it(self, poser):
        # THA3 hands back the same cached tensors for an unchanged input image, so converting one in place would
        # change what the next render composes the face from.
        for output_index in range(6):
            before = poser.outputs[output_index].clone()
            run(poser, output_index)
            assert torch.equal(poser.outputs[output_index], before), f"output {output_index} was modified"

    def test_the_poser_gets_the_character_in_its_own_range(self, poser):
        run(poser, 0)
        image = poser.seen_inputs[0]
        assert image.min() >= -1.0 and image.max() <= 1.0
        assert image.min() < 0.0, "still in [0, 1]: the mapping to the poser's [-1, 1] did not happen"
