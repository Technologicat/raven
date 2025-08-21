"""List available audio devices.

Useful for setting up the TTS (speech synthesize) audio device, if you want to use a non-default device.

You can use one of these as `tts_playback_audio_device` in `raven.client.config`.
"""

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from typing import Tuple

from mcpyrate import colorizer

import pygame
import pygame._sdl2.audio as sdl2_audio

from .. import __version__

logger.info(f"Raven-check-audio-devices version {__version__}")

def get_devices(capture_devices: bool = False) -> Tuple[str, ...]:
    init_by_me = not pygame.mixer.get_init()
    if init_by_me:
        pygame.mixer.init()
    devices = tuple(sdl2_audio.get_audio_device_names(capture_devices))
    if init_by_me:
        pygame.mixer.quit()
    return devices

def main():
    print(f"{colorizer.colorize('Playback', colorizer.Style.BRIGHT)} devices detected:")
    for device_name in get_devices():
        print(f"    {device_name}")

    print(f"{colorizer.colorize('Capture', colorizer.Style.BRIGHT)} devices detected:")
    for device_name in get_devices(capture_devices=True):
        print(f"    {device_name}")

if __name__ == "__main__":
    main()
