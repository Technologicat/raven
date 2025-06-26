"""Utilities for cel animation, with a functional flavor.

- The animations work by modifying the parameters of a *cel stack*, functionally
  (i.e. by returning a new, modified copy of the cel stack).

- The cel stack contains cel names and blend strengths only. The actual cels (images, as Torch tensors)
  are stored separately, to be looked up by name.

- The animation drivers are pure (stateless) functions. E.g. for a cycle of cels, the caller
  must record the epoch (start time of the first cycle), and pass it in. Similarly,
  for a fadeout animation, the caller must record the start time, and pass it in.

- The cel stack (after any parameter processing) can then be rendered by `render_celstack`.
"""

__all__ = ["render_celstack", "get_cel_index_in_stack",
           "animate_cel_cycle", "animate_cel_sequence", "animate_cel_fadeout",
           "animate_cel_cycle_with_fadeout", "animate_cel_sequence_with_fadeout"]

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import copy
import time
from typing import Dict, List, Tuple

import torch

def render_celstack(base_image: torch.tensor, celstack: List[Tuple[str, float]], torch_cels: Dict[str, torch.tensor]) -> torch.tensor:
    """Given a base RGBA image and a stack of RGBA cel images, blend the final image.

    `base_image`: shape [c, h, w], range [0, 1], linear RGB. 4 channels (RGBA). On any device.

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
    """Given `celname`, return its (zero-based) index in `celstack`, or -1 if not found."""
    for idx, (name, strength) in enumerate(celstack):
        if name == celname:
            return idx
    return -1

def animate_cel_cycle(cycle_duration: float,
                      epoch: float,
                      strength: float,
                      cels: List[str],
                      celstack: List[Tuple[str, float]]) -> Tuple[float, List[Tuple[str, float]]]:
    """Generic looping cel animation driver (e.g. "intense emotion" eye-waver effect).

    `cycle_duration` (seconds) is the duration of one cycle through `cels`. The special value 0.0 disables the animation.

    `epoch` anchors the cycle start time (as given by `time.time_ns()`). This is parameterized to keep this function stateless.

    `strength` is the cel opacity, range [0, 1].

    `cels` is the list of (one or more) cel names to cycle through. If the list is empty, this function does nothing.

    Returns `new_epoch, new_celstack`.

    Be sure to update your stored epoch; the epoch resets after each full cycle to avoid rounding issues during a long session.
    """
    new_celstack = copy.copy(celstack)
    if cycle_duration == 0.0 or not cels:  # convenience feature: zero cycle duration or no cels = effect disabled
        return new_celstack

    time_now = time.time_ns()
    t = (time_now - epoch) / 10**9
    cycle_pos = t / cycle_duration
    if cycle_pos > 1.0:
        epoch = time_now  # note `epoch` will be returned to caller
    cycle_pos = cycle_pos - float(int(cycle_pos))

    # NOTE: For the best look, this animation needs all of the `cels` to be present in `celstack`.
    # Hence, they all need to be present in the emotion templates, because we populate our cel stack
    # from those templates.
    #
    # During any missing cels, the animation will not show any cel.

    active_cel_number = int(len(cels) * cycle_pos)
    active_celname = cels[active_cel_number]

    # Set all inactive cels to zero strength
    for celname in cels:
        if celname != active_celname:
            inactive_idx = get_cel_index_in_stack(celname, new_celstack)
            if inactive_idx != -1:  # found?
                new_celstack[inactive_idx] = (celname, 0.0)

    active_idx = get_cel_index_in_stack(active_celname, new_celstack)
    if active_idx != -1:  # found?
        new_celstack[active_idx] = (active_celname, strength)

    return epoch, new_celstack

def animate_cel_sequence(t0: float,
                         duration: float,
                         strength: float,
                         cels: List[str],
                         celstack: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    """Generic play-once cel animation driver (e.g. an exclamation mark when the character realizes something).

    `t0` is the effect start time (as given by `time.time_ns()`). This is parameterized to keep this function stateless.

    `duration` (seconds) is divided evenly to the cels. The special value 0.0 disables the animation.

    `strength` is the cel opacity, range [0, 1].

    `cels` is the list of (one or more) cel names to show in sequence. If the list is empty, this function does nothing.

    Returns the modified celstack.
    """
    new_celstack = copy.copy(celstack)
    if duration == 0.0 or not cels:  # convenience feature: zero duration or no cels = effect disabled
        return new_celstack

    time_now = time.time_ns()
    t = (time_now - t0) / 10**9
    animation_pos = t / duration
    if animation_pos >= 1.0:  # effect ended?
        return new_celstack

    cel_number = int(len(cels) * animation_pos)
    celname = cels[cel_number]
    idx = get_cel_index_in_stack(celname, new_celstack)
    if idx != -1:  # found?
        new_celstack[idx] = (celname, strength)
    return new_celstack

def animate_cel_fadeout(t0: float,
                        duration: float,
                        cels: List[str],
                        celstack: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    """Generic fadeout cel animation driver (e.g. a huge sweatdrop that turns translucent and vanishes).

    `t0` is the effect start time (as given by `time.time_ns()`). This is parameterized to keep this function stateless.

    `duration` (seconds) is the fadeout duration. The special value 0.0 disables the animation.

    `cels` is the list of (one or more) cel names affected by the fadeout. Their strength will fade from its current value toward zero.
           If the list is empty, this function does nothing.
    """
    new_celstack = copy.copy(celstack)
    if duration == 0.0 or not cels:  # convenience feature: zero duration or no cels = effect disabled
        return new_celstack

    time_now = time.time_ns()
    t = (time_now - t0) / 10**9
    animation_pos = t / duration
    if animation_pos >= 1.0:  # effect ended?
        r = 0.0
    else:
        r = 1.0 - animation_pos  # linear fade; could modify this for other profiles

    for celname in cels:
        idx = get_cel_index_in_stack(celname, new_celstack)
        if idx != -1:  # found?
            _, strength = new_celstack[idx]
            new_celstack[idx] = (celname, r * strength)

    return new_celstack

def animate_cel_cycle_with_fadeout(cycle_duration: float,
                                   epoch: float,
                                   strength: float,
                                   fadeout_t0: float,
                                   fadeout_duration: float,
                                   cels: List[str],
                                   celstack: List[Tuple[str, float]]) -> Tuple[float, List[Tuple[str, float]]]:
    """Generic cel animation driver combining `animate_cel_cycle` and `animate_cel_fadeout`, which see.

    Returns `new_epoch, new_celstack`.
    """
    # Compute base strengths for the cels
    epoch, new_celstack = animate_cel_cycle(cycle_duration=cycle_duration,
                                            epoch=epoch,
                                            strength=strength,
                                            cels=cels,
                                            celstack=celstack)
    # Make the cels fade out
    new_celstack = animate_cel_fadeout(t0=fadeout_t0,
                                       duration=fadeout_duration,
                                       cels=cels,
                                       celstack=new_celstack)
    return epoch, new_celstack

def animate_cel_sequence_with_fadeout(t0: float,
                                      duration: float,
                                      strength: float,
                                      cels: List[str],
                                      celstack: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    """Generic cel animation driver combining `animate_cel_sequence` and `animate_cel_fadeout`, which see."""
    new_celstack = animate_cel_sequence(t0=t0,
                                        duration=duration,
                                        strength=strength,
                                        cels=cels,
                                        celstack=celstack)
    new_celstack = animate_cel_fadeout(t0=t0,
                                       duration=duration,
                                       cels=cels,
                                       celstack=new_celstack)
    return new_celstack
