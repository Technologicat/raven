"""A simple audio player for TTS (text to speech).

This is effectively a wrapper over `pygame`, so that we can easily switch audio playback backends later.
"""

__all__ = ["get_available_devices",
           "validate_playback_device",
           "Player",

           "DEFAULT_FREQUENCY",
           "DEFAULT_CHANNELS",
           "DEFAULT_BUFFER_SIZE",
           "instance",
           "initialize",
           "require"]

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from typing import BinaryIO, List, Optional

from unpythonic import memoize, Singleton

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
@memoize  # prevent spamming mixer init/quit
def get_available_devices() -> List[str]:
    """Return a list of the names of available audio playback devices."""
    should_init = not pygame.mixer.get_init()
    if should_init:
        pygame.mixer.init()
    devices = list(sdl2_audio.get_audio_device_names(False))  # flag: `False` means "playback devices", `True` means "capture devices"
    if should_init:  # should also teardown, then
        pygame.mixer.quit()
    return devices

def validate_playback_device(device_name: Optional[str]) -> str:
    """Validate `device_name` against list of audio playback devices detected on the system.

    The return value is always the name of a valid playback capture device on which the `Player`
    can be instantiated.

    If `device_name` is given, return `device_name` if OK; raise `ValueError` if the specified
    device was not found on the system.

    If `device_name is None`, return the name of the first available audio playback device.

    See the command-line utility `raven-check-audio-devices` to list audio devices on your system.
    """
    if device_name == "system-default":  # magic value to tell `Player` to let the backend choose the default device
        logger.debug("validate_playback_device: Using the system's current default audio playback device (will be handled by backend).")
        return device_name

    device_names = get_available_devices()
    if device_name is not None:  # User-specified device name
        try:
            device_names.index(device_name)  # we just want to check if it's there
        except IndexError:
            error_msg = f"validate_playback_device: No such audio playback device '{device_name}'."
            logger.error(error_msg)
            raise ValueError(error_msg)
        logger.debug(f"validate_playback_device: Using audio playback device '{device_name}'.")
    else:  # First available device
        if not device_names:
            error_msg = "validate_playback_device: No audio playback device found on this system."
            logger.error(error_msg)
            raise ValueError(error_msg)
        device_name = device_names[0]
        logger.debug(f"validate_playback_device: Using first available audio playback device '{device_name}'.")
    return device_name

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

        `device_name`: Name of playback audio device. `None` means "use first available device".

        See the command-line utility `raven-check-audio-devices` to list audio devices on your system.
        """
        logger.info("Player.__init__: Initializing audio player.")

        device_name = validate_playback_device(device_name)  # autodetect if `None`, and sanity check in any case
        if device_name != "system-default":
            pygame_device_name = device_name
            device_names = get_available_devices()
            assert device_name in device_names  # we only get here if the validation succeeded
        else:
            pygame_device_name = None
        self.device_name = device_name  # for information only
        plural_s = "s" if channels != 1 else ""
        logger.info(f"Player.__init__: Audio playback device '{device_name}', frequency {frequency}, {channels} channel{plural_s}, buffer size {buffer_size} samples.")

        # https://www.pygame.org/docs/ref/mixer.html
        pygame.mixer.init(frequency=frequency,
                          size=-16,  # minus: signed values will be used
                          channels=channels,
                          buffer=buffer_size,  # There seems to be no way to *get* the buffer size from `pygame.mixer`, so we must *set* it to know it.
                          devicename=pygame_device_name)  # `None` here means "use the system's default playback device".
        self.channels = channels
        self.frequency = frequency
        self.buffer_size = buffer_size

        self.latency = self.buffer_size / self.frequency  # seconds

        # Capture the teardown callable now, while the module is fully live.
        # At interpreter shutdown Python clears module globals (so `pygame` becomes `None`)
        # before finalizers run, and reaching `pygame.mixer.quit` through the module
        # would raise `AttributeError: 'NoneType' object has no attribute 'mixer'`.
        self._mixer_quit = pygame.mixer.quit

        logger.info("Player.__init__: Initialization complete.")

    def __del__(self):
        try:
            self._mixer_quit()
        except Exception:  # pygame internals may also be partially torn down
            pass

    def load(self, stream: BinaryIO) -> None:
        """Load an audio file for playback.

        `stream`: a filelike (e.g. an `io.BytesIO` object).
        """
        logger.info("Player.load: Loading audio stream to player.")
        try:
            pygame.mixer.music.load(stream)
        except pygame.error as exc:
            raise RuntimeError(f"Player.load: failed to load audio stream into player, reason {type(exc)}: {exc}") from exc
        logger.info("Player.load: Successfully loaded.")

    def start(self) -> None:
        """Start playback of the loaded audio file."""
        logger.info("Player.start: Starting audio playback.")
        pygame.mixer.music.play()
        logger.info("Player.start: Done.")

    def stop(self) -> None:
        """Stop playback."""
        logger.info("Player.stop: Stopping audio playback.")
        pygame.mixer.music.stop()
        logger.info("Player.stop: Done.")

    def is_playing(self) -> bool:
        """Return whether the audio player is playing."""
        return pygame.mixer.music.get_busy()

    def get_position(self) -> float:
        """Return playback position in seconds, in the loaded audio file.

        This auto-compensates for `buffer_size`, but not for unknown delays elsewhere in the system.
        """
        return (pygame.mixer.music.get_pos() / 1000) - self.latency


# Default audio playback settings. `DEFAULT_BUFFER_SIZE` is slightly larger than pygame's
# default 512 to prevent xruns while the AI translator/subtitler is running simultaneously.
DEFAULT_FREQUENCY = 44100
DEFAULT_CHANNELS = 2
DEFAULT_BUFFER_SIZE = 2048

# The default (singleton) `Player` instance. `None` until `initialize` is called.
#
# Pre-populated to `None` so that apps can read the attribute and decide whether to
# initialize (or re-use) the player. Apps that don't need audio playback don't need
# to call `initialize`; they just leave this as `None`.
#
# Access via `raven.common.audio.player.instance` (read-only by convention).
instance: Optional["Player"] = None

def initialize(frequency: int = DEFAULT_FREQUENCY,
               channels: int = DEFAULT_CHANNELS,
               buffer_size: int = DEFAULT_BUFFER_SIZE,
               device_name: Optional[str] = None) -> "Player":
    """Initialize the default audio player singleton.

    Constructs a `Player` with the given parameters and assigns it to the module-level
    `instance`. Idempotent: subsequent calls return the existing instance without
    rebuilding it (since `Player` is a singleton anyway).

    `device_name`: One of the Playback device names listed by `raven-check-audio-devices`,
                   `"system-default"` to let the backend pick, or `None` for the first
                   available device.

    Returns the `Player` instance.
    """
    global instance
    if instance is not None:
        logger.info("initialize: audio player already initialized. Using existing instance.")
        return instance

    if device_name == "system-default":
        logger.info("initialize: Using system's current default audio playback device. If you want to use another device, see `raven.client.config`, and run `raven-check-audio-devices` to get available choices.")
    elif device_name is not None:
        logger.info(f"initialize: Validating audio playback device '{device_name}'.")
    else:
        logger.info("initialize: Using first available audio playback device. If you want to use another device, see `raven.client.config`, and run `raven-check-audio-devices` to get available choices.")

    instance = Player(frequency=frequency,
                      channels=channels,
                      buffer_size=buffer_size,
                      device_name=device_name)
    return instance

def require() -> "Player":
    """Return the player, raising `RuntimeError` if not initialized.

    Use this at the entry point of any code that needs audio playback: it fails fast
    with a clear message, instead of letting an `AttributeError: 'NoneType'` surface
    deep inside a playback call.
    """
    if instance is None:
        raise RuntimeError("raven.common.audio.player.require: no player initialized. Call `raven.common.audio.initialize(...)` or `raven.common.audio.player.initialize(...)` first.")
    return instance
