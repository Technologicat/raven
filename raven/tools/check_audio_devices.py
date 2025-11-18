"""List available audio devices.

Useful for setting up the TTS (speech synthesize) audio device, if you want to use a non-default device.

You can use one of these as `tts_playback_audio_device` in `raven.client.config`.
"""

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from typing import List

from mcpyrate import colorizer

from .. import __version__

from ..common.audio import player as audio_player
from ..common.audio import recorder as audio_recorder

logger.info(f"Raven-check-audio-devices version {__version__}")

def get_available_devices(role: str) -> List[str]:
    if role == "playback":
        devices = audio_player.get_available_devices()
    else:  # role == "capture":
        devices = audio_recorder.get_available_devices()
    return devices

def main():
    print(f"{colorizer.colorize('▶ Playback', colorizer.Style.BRIGHT, colorizer.Fore.GREEN)} devices detected:")
    for device_name in get_available_devices(role="playback"):
        print(f"    {device_name}")

    print(f"{colorizer.colorize('⏺ Capture', colorizer.Style.BRIGHT, colorizer.Fore.RED)} devices detected:")
    for device_name in get_available_devices(role="capture"):
        print(f"    {device_name}")

if __name__ == "__main__":
    main()
