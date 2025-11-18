"""List available audio devices.

Useful for setting up the TTS (speech synthesize) audio device, if you want to use a non-default device.

You can use one of these as `tts_playback_audio_device` in `raven.client.config`.
"""

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from typing import Tuple

from mcpyrate import colorizer

import pvrecorder
import pygame
import pygame._sdl2.audio as sdl2_audio

from .. import __version__

logger.info(f"Raven-check-audio-devices version {__version__}")

def get_devices(capture_devices: bool = False) -> Tuple[str, ...]:
    if not capture_devices:
        should_init = not pygame.mixer.get_init()
        if should_init:
            pygame.mixer.init()
        devices = tuple(sdl2_audio.get_audio_device_names(capture_devices))
        if should_init:  # should also teardown, then
            pygame.mixer.quit()
    else:
        # `pygame` doesn't support recording (although it can *list*
        # which capture devices it sees), so Raven uses `pvrecorder`
        # for recording audio for its STT (speech to text) features.
        #
        # To get a guaranteed-correct list of devices, query with
        # the same library that will be used for recording.
        #
        # There is at least one important difference in practice:
        # `pygame` doesn't see monitoring devices (i.e. capture devices
        # that record the audio that is going to an audio output),
        # while `pvrecorder` does.
        devices = tuple(pvrecorder.PvRecorder.get_available_devices())
    return devices

def main():
    print(f"{colorizer.colorize('▶ Playback', colorizer.Style.BRIGHT, colorizer.Fore.GREEN)} devices detected:")
    for device_name in get_devices():
        print(f"    {device_name}")

    print(f"{colorizer.colorize('⏺ Capture', colorizer.Style.BRIGHT, colorizer.Fore.RED)} devices detected:")
    for device_name in get_devices(capture_devices=True):
        print(f"    {device_name}")

if __name__ == "__main__":
    main()
