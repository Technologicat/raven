"""Utilities related to the AI avatar system, shared between `raven.server`, `raven.avatar.pose_editor`, and `raven.client.api`.

This module is licensed under the 2-clause BSD license.
"""

__all__ = ["posedict_keys", "posedict_key_to_index",
           "load_emotion_presets",
           "posedict_to_pose", "pose_to_posedict",
           "torch_load_rgba_image", "torch_image_to_numpy",
           "supported_cels", "scan_addon_cels", "render_celstack", "get_cel_index_in_stack"]

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import json
import os
from typing import Dict, List, Tuple

import numpy as np

import torch

from ...vendor.tha3.util import torch_linear_to_srgb, numpy_srgb_to_linear, resize_PIL_image, extract_PIL_image_from_filelike


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

        emotions = {emotion0_name: {"pose": posedict0, "cels": celstack0}, ...}
        emotion_names = [emotion0_name, emotion1_name, ...]

    The posedict contains the actual pose data. The list is a sorted list of emotion names
    that can be used to map a linear index (e.g. the choice index in a GUI dropdown)
    to the corresponding key of `emotions`.

    The celstack format is [(celname0, strength0), ...]. It controls the cel blending step
    that happens before posing.

    The directory "raven/avatar/assets/emotions/" must also contain a "_defaults.json" file,
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
            emotion = emotions_from_json[emotion_name]
        except (FileNotFoundError, KeyError):  # If no separate json exists for the specified emotion, load the factory default (all 28 emotions have a default).
            emotion = factory_default_emotions[emotion_name]
            # If still not found, it's an error, so fail-fast: let the exception propagate.
        posedict = emotion["pose"]
        celstack = list(emotion["cels"].items())
        return {"pose": posedict, "cels": celstack}

    # Dict keeps its keys in insertion order, so define some special states before inserting the actual emotions.
    emotions = {"[custom]": {"pose": {}, "cels": []},  # custom = in `raven.avatar.pose_editor.app`, the user has changed at least one value manually after last loading a preset
                "[reset]": load_emotion_with_fallback("zero")}  # reset = a preset with all pose sliders in their default positions. Found in "_defaults.json".
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

def _preprocess_poser_image(image: np.array) -> np.array:
    """Do some things THA3 needs:

      - Take all pixels that have zero alpha (and any color), and replace them with [0, 0, 0, 0],
        so that they won't affect the color when we run the image through the poser.

      - In the RGB channels, convert SRGB to linear.

      - Convert to the Torch layout [c, h, w].

    Input is shape [h, w, c], float32, range [0, 1] (SRGB, i.e. with gamma correction applied).

    Output is [c, h, w], float32, range [0, 1] (linear RGB, i.e. no gamma).

    This is needed only upon image loading. We do this on the CPU.
    """
    h, w, c = image.shape
    if c == 4:  # alpha channel present?
        # search for transparent pixels(alpha==0) and change them to [0 0 0 0] to avoid the color influence to the model
        mask = np.where(image[:, :, 3] == 0, 0, 1)
        mask = np.expand_dims(mask, 2)  # unsqueeze
        image = image * mask
    else:
        image = image.copy()
    image[:, :, 0:3] = numpy_srgb_to_linear(image[:, :, 0:3])
    image = image.reshape(h * w, c).transpose().reshape(c, h, w)
    return image

def torch_load_rgba_image(filename: str, target_w: int, target_h: int, device: str, dtype: torch.dtype) -> torch.tensor:
    """Load an RGBA image from disk, Lanczos-rescale it to `(target_w, target_h)`, load that into a Torch tensor, and send to `device` with the given float `dtype`.

    If the aspect ratio of the image file does not match the target's, fit a box with the target's aspect ratio onto the image area, and crop the excess.
    Rescale the part inside the box to the target size. See `raven.vendor.tha3.util.resize_PIL_image` for details.

    The image file is assumed to be SRGB encoded (i.e. with gamma correction applied).
    It may have either 3 channels (RGB) or 4 channels (RGBA).

    Return value is a Torch tensor, shape [c, h, w], specified float `dtype`, range [0, 1], linear RGB (i.e. no gamma correction), loaded onto `device`.

    Raises `ValueError` if the input file has no alpha channel.
    """
    pil_image = resize_PIL_image(extract_PIL_image_from_filelike(filename),
                                 (target_w, target_h))
    w, h = pil_image.size
    if pil_image.mode != "RGBA":  # input image must have an alpha channel
        raise ValueError("Incompatible input image (no alpha channel)")
    arr = np.asarray(pil_image.convert("RGBA"))  # [h, w, c], uint8, SRGB (gamma-corrected)
    arr = np.array(arr, dtype=np.float32) / 255  # uint8 -> [0, 1]
    numpy_image = _preprocess_poser_image(arr)  # -> [c, h, w], SRGB to linear, zero out transparent pixels
    torch_image = torch.from_numpy(numpy_image).to(device).to(dtype)
    return torch_image

def torch_image_to_numpy(image: torch.tensor) -> np.array:
    """Convert Torch image tensor (on any device) to NumPy image array (on CPU).

    Input is a Torch tensor on any device, of shape [c, h, w], any float dtype, range [0, 1], linear RGB.
    The input may have either 3 channels (RGB) or 4 channels (RGBA).

    Output is a NumPy array on CPU, of shape [h, w, c], dtype float32, range [0, 1], SRGB (display-ready).
    You can `.ravel()` this for sending into a DPG texture.
    """
    with torch.no_grad():
        image = torch.clone(image.detach())
        image[:3, :, :] = torch_linear_to_srgb(image[:3, :, :])
        c, h, w = image.shape
        image = torch.transpose(image.reshape(c, h * w), 0, 1).reshape(h, w, c)  # -> [h, w, c]
        numpy_image = image.float().detach().cpu().numpy()
    return numpy_image

# --------------------------------------------------------------------------------
# Cel blending

# List of cels understood by the character loaders (in `pose_editor` and `animator`).
# This also defines the canonical render order for the cels (bottommost first).
#
# This is a fixed list for two reasons:
#  - Having a fixed set of supported cels makes emotion JSON files compatible between different characters.
#  - If a particular character does not have some of the cels, simply skipping those blend keys
#    degrades the look gracefully (instead of completely breaking how the character looks,
#    e.g. if their hair or clothing was a custom cel).
supported_cels = [
    # Effects that go on the character itself.
    # These can be set up in the pose editor as part of an emotion.
    # Applied before posing.
    "blush1", "blush2", "blush3",
    "shadow1",
    "sweat1", "sweat2", "sweat3",
    "tears1", "tears2", "tears3",
    "waver1", "waver2",  # These two are special, for the "intense emotion" eye-wavering effect.
    # Anime-style effects that go *around* the character.
    # These are automatically applied by the live animator when the relevant emotion state is entered.
    # Applied after posing.
    "fx_angervein1", "fx_angervein2",  # the standalone vein, often red, anger (cycle, then fade out)
    "fx_sweatdrop1",  # huge sweatdrop (fade out)
    "fx_blackcloud1", "fx_blackcloud2",  # frustration (cycle, then fade out)
    "fx_shock1",  # shock lines, fear (fade out)
    "fx_notice1", "fx_notice2",  # notice lines, realization (quick flash cel1, cel2, then off)
    "fx_beaming1", "fx_beaming2",  # happy lines, joy (quick flash cel1, cel2, then off)
    "fx_question1", "fx_question2", "fx_question3",  # question mark(s), confusion (quick flash cel1, cel2, cel3, then off)
    "fx_exclaim1", "fx_exclaim2", "fx_exclaim3",  # exclamation mark(s), realization (quick flash cel1, cel2, cel3, then off)
]

def scan_addon_cels(image_file_name: str) -> Dict[str, str]:
    """Given `image_file_name`, scan for its associated add-on cels on disk.

    The cels should have the same basename, followed by an underscore and then the cel name.
    E.g. "example.png" may have a cel "example_blush.png".

    Returns a dict `{celname0: absolute_filename0, ...}`.

    The result may be empty if there are no add-on cels for `image_file_name`.
    """
    logger.info(f"scan_addon_cels: Scanning cels for '{image_file_name}'.")
    basename = os.path.basename(image_file_name)  # e.g. "/foo/bar/example.png" -> "example.png"
    stem, ext = os.path.splitext(basename)  # -> "example", ".png"
    cels_filenames = {}
    for root, dirs, files in os.walk(os.path.dirname(image_file_name), topdown=True):
        dirs.clear()  # don't recurse
        for filename in files:
            if filename.startswith(f"{stem}_") and filename.endswith(ext):
                base, _ = os.path.splitext(filename)  # "example_blush.png" -> "example_blush"
                _, celname = base.split("_", maxsplit=1)  # # "example_blush" -> "blush"
                if celname in supported_cels:
                    cels_filenames[celname] = os.path.join(root, filename)
                else:
                    logger.warning(f"scan_addon_cels: Ignoring unsupported cel '{celname}' for '{image_file_name}'. Supported cels are: {supported_cels}")

    # Sort the results.
    def index_in_supported_cels(item):
        celname, filename = item
        return supported_cels.index(celname)
    cels_filenames = dict(sorted(cels_filenames.items(), key=index_in_supported_cels))

    if cels_filenames:
        logger.info(f"scan_addon_cels: Found add-on-cels for '{image_file_name}': {list(cels_filenames.keys())}.")
    else:
        logger.info(f"scan_addon_cels: No add-on cels found for '{image_file_name}'.")

    return cels_filenames

def render_celstack(base_image: torch.tensor, celstack: List[Tuple[str, float]], torch_cels: Dict[str, torch.tensor]) -> torch.tensor:
    """Given a base RGBA image and a stack of RGBA cel images, blend the final image.

    `base_image`: shape [c, h, w], range [0, 1], linear RGB. 4 channels (RGBA).

    `celstack`: The add-on cels to blend in, `[(name0, strength0), ...]`. Each strength is in [0, 1],
                where 0 means completely off, and 1 means full strength.

                The cells should be listed in a bottom-to-top order (as if they were actual physical cels
                stacked on top of the base image).

                Cels with zero strength are automatically skipped.

                Cels not present in `torch_cels` (see below) are automatically skipped. This is a
                convenience feature, as the add-on cels are optional and need to be made separately
                for each character.

    `torch_cels`: The actual image data for the cels, `{name0: tensor0, ...}`. Each tensor in same format as `base_image`.

    The return value is a tensor of shape [c, h, w], containing the final blended image.
    """
    if not celstack:
        logger.debug("compose_cels: Celstack is empty, returning base image as-is.")
        return base_image.clone()  # always cloned, because the caller may directly modify the result.

    logger.debug(f"compose_cels: Composing {celstack}.")

    def over(a: torch.tensor, b: torch.tensor) -> torch.tensor:
        """Alpha-blending operator. "a over b", i.e. "a" sits on top of "b".

        https://en.wikipedia.org/wiki/Alpha_compositing
        """
        RGBa = a[:3, :, :]
        RGBb = b[:3, :, :]
        alpa = a[3, :, :].unsqueeze(0)  # [1, h, w]
        alpb = b[3, :, :].unsqueeze(0)
        alpo = (alpa + alpb * (1 - alpa))
        RGBo = (RGBa * alpa + RGBb * alpb * (1 - alpa)) / (alpo + 1e-5)
        return torch.cat([RGBo, alpo], dim=0)

    out = base_image.clone()
    for celname, strength in celstack:
        if celname not in torch_cels:  # Ignore any cels that are not loaded (e.g. the character didn't have them).
            continue
        if strength == 0.0:
            continue
        elif strength == 1.0:
            cel = torch_cels[celname]
        else:
            cel = torch_cels[celname].clone()
            cel[3, :, :].mul_(strength)
        out = over(cel, out)

    return out

def get_cel_index_in_stack(celname: str, celstack: List[Tuple[str, float]]) -> int:
    """Given `celname`, return its position in `celstack`, or -1 if not found."""
    for idx, (name, strength) in enumerate(celstack):
        if name == celname:
            return idx
    return -1
