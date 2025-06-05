"""Utilities for the avatar subapp."""

__all__ = ["posedict_keys", "posedict_key_to_index",
           "load_emotion_presets",
           "posedict_to_pose", "pose_to_posedict",
           "maybe_install_models",
           "convert_linear_to_srgb", "convert_float_to_uint8", "to_talkinghead_image"]

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import json
import os
from typing import Dict, List, Tuple

import PIL

import numpy as np

import torch

# from .vendor.tha3.util import rgba_to_numpy_image, rgb_to_numpy_image, grid_change_to_numpy_image, torch_linear_to_srgb
from ..vendor.tha3.util import torch_linear_to_srgb


# The keys for a pose in the emotion JSON files.
posedict_keys = ["eyebrow_troubled_left_index", "eyebrow_troubled_right_index",
                 "eyebrow_angry_left_index", "eyebrow_angry_right_index",
                 "eyebrow_lowered_left_index", "eyebrow_lowered_right_index",
                 "eyebrow_raised_left_index", "eyebrow_raised_right_index",
                 "eyebrow_happy_left_index", "eyebrow_happy_right_index",
                 "eyebrow_serious_left_index", "eyebrow_serious_right_index",
                 "eye_wink_left_index", "eye_wink_right_index",
                 "eye_happy_wink_left_index", "eye_happy_wink_right_index",
                 "eye_surprised_left_index", "eye_surprised_right_index",
                 "eye_relaxed_left_index", "eye_relaxed_right_index",
                 "eye_unimpressed_left_index", "eye_unimpressed_right_index",
                 "eye_raised_lower_eyelid_left_index", "eye_raised_lower_eyelid_right_index",
                 "iris_small_left_index", "iris_small_right_index",
                 "mouth_aaa_index",
                 "mouth_iii_index",
                 "mouth_uuu_index",
                 "mouth_eee_index",
                 "mouth_ooo_index",
                 "mouth_delta",
                 "mouth_lowered_corner_left_index", "mouth_lowered_corner_right_index",
                 "mouth_raised_corner_left_index", "mouth_raised_corner_right_index",
                 "mouth_smirk",
                 "iris_rotation_x_index", "iris_rotation_y_index",
                 "head_x_index", "head_y_index",
                 "neck_z_index",
                 "body_y_index", "body_z_index",
                 "breathing_index"]
assert len(posedict_keys) == 45

# posedict_keys gives us index->key; make an inverse mapping.
posedict_key_to_index = {key: idx for idx, key in enumerate(posedict_keys)}


def load_emotion_presets(directory: str) -> Tuple[Dict[str, Dict[str, float]], List[str]]:
    """Load emotion presets from disk.

    Returns the tuple `(emotions, emotion_names)`, where::

        emotions = {emotion0_name: posedict0, ...}
        emotion_names = [emotion0_name, emotion1_name, ...]

    The dict contains the actual pose data. The list is a sorted list of emotion names
    that can be used to map a linear index (e.g. the choice index in a GUI dropdown)
    to the corresponding key of `emotions`.

    The directory "talkinghead/emotions" must also contain a "_defaults.json" file,
    containing factory defaults (as a fallback) for the 28 standard emotions
    (as recognized by distilbert), as well as a hidden "zero" preset that represents
    a neutral pose. (This is separate from the "neutral" emotion, which is allowed
    to be "non-zero".)
    """
    emotion_names = set()
    for root, dirs, files in os.walk(directory, topdown=True):
        for filename in files:
            if filename == "_defaults.json":  # skip the repository containing the default fallbacks
                continue
            if filename.lower().endswith(".json"):
                emotion_names.add(filename[:-5])  # drop the ".json"

    # Load the factory-default emotions as a fallback
    with open(os.path.join(directory, "_defaults.json"), "r") as json_file:
        factory_default_emotions = json.load(json_file)
    for key in factory_default_emotions:  # get keys from here too, in case some emotion files are missing
        if key != "zero":  # not an actual emotion, but a "reset character" feature
            emotion_names.add(key)

    emotion_names = list(emotion_names)
    emotion_names.sort()  # the 28 actual emotions

    def load_emotion_with_fallback(emotion_name: str) -> Dict[str, float]:
        try:
            with open(os.path.join(directory, f"{emotion_name}.json"), "r") as json_file:
                emotions_from_json = json.load(json_file)  # A single json file may contain presets for multiple emotions.
            posedict = emotions_from_json[emotion_name]
        except (FileNotFoundError, KeyError):  # If no separate json exists for the specified emotion, load the factory default (all 28 emotions have a default).
            posedict = factory_default_emotions[emotion_name]
        # If still not found, it's an error, so fail-fast: let the app exit with an informative exception message.
        return posedict

    # Dict keeps its keys in insertion order, so define some special states before inserting the actual emotions.
    emotions = {"[custom]": {},  # custom = the user has changed at least one value manually after last loading a preset
                "[reset]": load_emotion_with_fallback("zero")}  # reset = a preset with all sliders in their default positions. Found in "_defaults.json".
    for emotion_name in emotion_names:
        emotions[emotion_name] = load_emotion_with_fallback(emotion_name)

    emotion_names = list(emotions.keys())
    return emotions, emotion_names


def posedict_to_pose(posedict: Dict[str, float]) -> List[float]:
    """Convert a posedict (from an emotion JSON) into a list of morph values (in the order the models expect them)."""
    # sanity check
    unrecognized_keys = set(posedict.keys()) - set(posedict_keys)
    if unrecognized_keys:
        logger.warning(f"posedict_to_pose: ignoring unrecognized keys in posedict: {unrecognized_keys}")
    # Missing keys are fine - keys for zero values can simply be omitted.

    pose = [0.0 for i in range(len(posedict_keys))]
    for idx, key in enumerate(posedict_keys):
        pose[idx] = posedict.get(key, 0.0)
    return pose


def pose_to_posedict(pose: List[float]) -> Dict[str, float]:
    """Convert `pose` into a posedict for saving into an emotion JSON."""
    return dict(zip(posedict_keys, pose))

# --------------------------------------------------------------------------------

def maybe_install_models(hf_reponame: str, modelsdir: str) -> None:
    """Download and install the posing engine (THA3) models into `modelsdir` if the directory does not exist yet. Else do nothing.

    For maximal OS compatibility, symlinks are not used.

    `hf_reponame`: HuggingFace repository to download from, e.g. "OktayAlpk/talking-head-anime-3".
    `modelsdir`: Local path (absolute or relative) to install in.
    """
    logger.info(f"maybe_install_models: Checking for THA3 models at '{modelsdir}'.")
    if os.path.exists(modelsdir):
        logger.info("maybe_install_models: THA3 models directory exists. We're good to go!")
    else:
        logger.info(f"maybe_install_models: THA3 models not yet installed. Installing from '{hf_reponame}' into '{modelsdir}'. (Don't worry, this will happen only once.)")

        # API:
        #   https://huggingface.co/docs/huggingface_hub/en/guides/download
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise ImportError(
                "You need to install huggingface_hub to install talkinghead models automatically. "
                "See https://pypi.org/project/huggingface-hub/ for installation."
            )
        os.makedirs(modelsdir, exist_ok=True)
        # Installing with symlinks would be generally better, but MS Windows support for symlinks is not optimal,
        # so for maximal compatibility we avoid them. The drawback of installing directly as plain files is that
        # if multiple programs need to download THA3, they will do so separately. But THA3 is rather rare, so in
        # practice this is unlikely to be an issue.
        snapshot_download(repo_id=hf_reponame, local_dir=modelsdir, local_dir_use_symlinks=False)

# --------------------------------------------------------------------------------

def convert_linear_to_srgb(image: torch.Tensor) -> torch.Tensor:
    """RGBA (linear) -> RGBA (SRGB), preserving the alpha channel."""
    rgb_image = torch_linear_to_srgb(image[0:3, :, :])
    return torch.cat([rgb_image, image[3:4, :, :]], dim=0)

def convert_float_to_uint8(image: np.array) -> np.array:
    """Convert the given `image` (a float array of shape [h, w, c]) into uint8, for file saving."""
    uint8_image = image * 255.0
    uint8_image = np.array(uint8_image, dtype=np.uint8)
    return uint8_image

# # I have no idea what half of these modes are doing. Fortunately this function is no longer needed.
# # See also `vendor.tha3.util.convert_output_image_from_torch_to_numpy`, which this is based on, fortunately also unused.
# def torch_image_to_numpy(image: torch.tensor) -> np.array:
#     if image.shape[2] == 2:
#         h, w, c = image.shape
#         numpy_image = torch.transpose(image.reshape(h * w, c), 0, 1).reshape(c, h, w)
#     elif image.shape[0] == 4:
#         numpy_image = rgba_to_numpy_image(image)
#     elif image.shape[0] == 3:
#         numpy_image = rgb_to_numpy_image(image)
#     elif image.shape[0] == 1:
#         c, h, w = image.shape
#         alpha_image = torch.cat([image.repeat(3, 1, 1) * 2.0 - 1.0, torch.ones(1, h, w)], dim=0)
#         numpy_image = rgba_to_numpy_image(alpha_image)
#     elif image.shape[0] == 2:
#         numpy_image = grid_change_to_numpy_image(image, num_channels=4)
#     else:
#         msg = f"torch_image_to_numpy: unsupported # image channels: {image.shape[0]}"
#         logger.error(msg)
#         raise RuntimeError(msg)
#     numpy_image = np.uint8(np.rint(numpy_image * 255.0))
#     return numpy_image

def to_talkinghead_image(image: PIL.Image, new_size: Tuple[int] = (512, 512)) -> PIL.Image:
    """Resize image to `new_size`, add alpha channel, and center.

    With default `new_size`:

      - Step 1: Resize (Lanczos) the image to maintain the aspect ratio with the larger dimension being 512 pixels.
      - Step 2: Create a new image of size 512x512 with transparency.
      - Step 3: Paste the resized image into the new image, centered.
    """
    image.thumbnail(new_size, PIL.Image.LANCZOS)
    new_image = PIL.Image.new("RGBA", new_size)
    new_image.paste(image, ((new_size[0] - image.size[0]) // 2,
                            (new_size[1] - image.size[1]) // 2))
    return new_image
