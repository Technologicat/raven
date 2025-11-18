"""A simple audio player for TTS (text to speech).

This is effectively a wrapper over `pygame`, so that we can easily switch audio playback backends later.
"""

__all__ = ["get_available_devices",
           "Player"]

from typing import BinaryIO, List, Optional

from unpythonic import Singleton

import pygame
import pygame._sdl2.audio as sdl2_audio

# `pygame` doesn't support recording (although it can *list*
# which capture devices it sees), so Raven uses `pvrecorder`
# for recording audio for its STT (speech to text) features.
#
# To get a guaranteed-correct list of devices for each role (playback, capture),
# we query with the same library that is used for that role.
#
# There is at least one important difference in practice:
# `pygame` doesn't see monitoring devices (i.e. capture devices
# that record the audio that is going to an audio output),
# while `pvrecorder` does.
#
def get_available_devices() -> List[str]:
    """Return a list of the names of available audio playback devices."""
    should_init = not pygame.mixer.get_init()
    if should_init:
        pygame.mixer.init()
    devices = list(sdl2_audio.get_audio_device_names(False))  # flag: `False` means "playback devices", `True` means "capture devices"
    if should_init:  # should also teardown, then
        pygame.mixer.quit()
    return devices

class Player(Singleton):
    def __init__(self,
                 frequency: int,
                 channels: int,
                 buffer_size: int,
                 device_name: Optional[str] = None):
        """A simple audio player for TTS (text to speech).

        Playback sample format is s16 (signed 16-bit).

        `frequency`: Playback sample rate.

        `channels`: 1 for mono, 2 for stereo (note stereo playback can also play mono files)

        `buffer_size`: Length of playback buffer, in samples at `frequency`.

                       Higher values avoid xruns under heavy system load, but add more latency.

                       If unsure, try e.g. 512.

        `device_name`: Name of playback audio device. `None` means "use the system's default device".

        See the command-line utility `raven-check-audio-devices` to list audio devices on your system.
        """
        # https://www.pygame.org/docs/ref/mixer.html
        pygame.mixer.init(frequency=frequency,
                          size=-16,  # minus: signed values will be used
                          channels=channels,
                          buffer=buffer_size,  # There seems to be no way to *get* the buffer size from `pygame.mixer`, so we must *set* it to know it.
                          devicename=device_name)  # `None` means "use the system's default playback device".
        self.channels = channels
        self.frequency = frequency
        self.buffer_size = buffer_size
        self.device_name = device_name

        self.latency = self.buffer_size / self.frequency  # seconds

    def __del__(self):
        pygame.mixer.quit()

    def load(self, stream: BinaryIO) -> None:
        """Load an audio file for playback.

        `stream`: a filelike (e.g. an `io.BytesIO` object).
        """
        try:
            pygame.mixer.music.load(stream)
        except pygame.error as exc:
            raise RuntimeError("Player.load: failed to load audio stream into player, reason {type(exc)}: {exc}") from exc

    def start(self) -> None:
        """Start playback of the loaded audio file."""
        pygame.mixer.music.play()

    def stop(self) -> None:
        """Stop playback."""
        pygame.mixer.music.stop()

    def is_playing(self) -> bool:
        """Return whether the audio player is playing."""
        return pygame.mixer.music.get_busy()

    def get_position(self) -> float:
        """Return playback position in seconds, in the loaded audio file.

        This auto-compensates for `buffer_size`, but not for unknown delays elsewhere in the system.
        """
        return (pygame.mixer.music.get_pos() / 1000) - self.latency
