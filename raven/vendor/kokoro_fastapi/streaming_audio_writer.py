"""Audio conversion service with proper streaming support.

This file is part of Kokoro-FastAPI:
    https://github.com/remsky/Kokoro-FastAPI

Original path was `api/src/services/streaming_audio_writer.py`.
This version is from commit e54ea702ab611fd9a77c898c50d3c6653e26608d, dated 17 July, 2026.

Used under the Apache License 2.0. **This file has been modified from the original**, which section 4(b) of
that licence requires stating. The modifications are confined to this docstring, an `__all__`, dropping two
imports the original does not use (`struct`, `soundfile`), and swapping `loguru` for the standard library's
logging so that vendoring this file pulls in no logging framework of its own. Nothing here changes behaviour.

Keep the commit hash above current when re-vendoring. It is what makes the divergence recoverable: without
it, "modified from upstream" cannot be separated into our changes and their progress. That mattered - the
previous pin sat at April 2025 while upstream fixed a lost final page on OGG/Opus and a chunk-end click on
WAV, and the pin is how those were found.
"""

__all__ = ["StreamingAudioWriter"]

import logging
from io import BytesIO
from typing import Optional

import av
import numpy as np

logger = logging.getLogger(__name__)


class StreamingAudioWriter:
    """Handles streaming audio format conversions"""

    def __init__(self, format: str, sample_rate: int, channels: int = 1):
        self.format = format.lower()
        self.sample_rate = sample_rate
        self.channels = channels
        self.bytes_written = 0
        self.pts = 0

        codec_map = {
            "wav": "pcm_s16le",
            "mp3": "mp3",
            "opus": "libopus",
            "flac": "flac",
            "aac": "aac",
        }
        # Format-specific setup
        if self.format in ["wav", "flac", "mp3", "pcm", "aac", "opus"]:
            if self.format != "pcm":
                self.output_buffer = BytesIO()
                container_options = {}
                # Try disabling Xing VBR header for MP3 to fix iOS timeline reading issues
                if self.format == "mp3":
                    # Disable Xing VBR header
                    container_options = {"write_xing": "0"}
                    logger.debug("Disabling Xing VBR header for MP3 encoding.")

                self.container = av.open(
                    self.output_buffer,
                    mode="w",
                    format=self.format if self.format != "aac" else "adts",
                    options=container_options,  # Pass options here
                )
                self.stream = self.container.add_stream(
                    codec_map[self.format],
                    rate=self.sample_rate,
                    layout="mono" if self.channels == 1 else "stereo",
                )
                # Set bit_rate only for codecs where it's applicable and useful
                if self.format in ["mp3", "aac", "opus"]:
                    self.stream.bit_rate = 128000
        else:
            raise ValueError(
                f"Unsupported format: {self.format}"
            )  # Use self.format here

    def close(self):
        if hasattr(self, "container"):
            self.container.close()

        if hasattr(self, "output_buffer"):
            self.output_buffer.close()

    def write_chunk(
        self, audio_data: Optional[np.ndarray] = None, finalize: bool = False
    ) -> bytes:
        """Write a chunk of audio data and return bytes in the target format.

        Args:
            audio_data: Audio data to write, or None if finalizing
            finalize: Whether this is the final write to close the stream
        """

        if finalize:
            if self.format != "pcm":
                # Flush stream encoder
                packets = self.stream.encode(None)
                for packet in packets:
                    self.container.mux(packet)

                # Whether the buffer may be read before or after `close()` depends on what the format does
                # at close, and the two families want opposite orders.
                #
                #   - WAV and FLAC *seek back to the start* to patch a header they could not fill in
                #     until the length was known (RIFF sizes; FLAC STREAMINFO). Every chunk truncated
                #     this buffer, so position 0 is no longer the header - it is the flushed audio we
                #     just muxed, and the patch overwrites it. Read first, then close.
                #   - OGG/Opus write their final page *during* close, so their last audio only exists
                #     afterwards. Close first, then read.
                #
                # Getting this wrong is silent: the file still decodes, just short. Upstream reordered
                # to fix OGG/Opus truncation, hit the resulting corruption on WAV, and special-cased WAV
                # alone by returning b"" - which leaves FLAC broken, dropping the tail of every encode.
                # Measured here at 2304 of 3200 samples surviving a 0.2 s round trip.
                if self.format in ("wav", "flac"):
                    data = self.output_buffer.getvalue()
                    self.container.close()
                else:
                    self.container.close()
                    data = self.output_buffer.getvalue()
                logger.debug(f"write_chunk: finalized {self.format}, {len(data)} bytes of trailing data")

                self.output_buffer.close()
                return data

        if audio_data is None or len(audio_data) == 0:
            return b""

        if self.format == "pcm":
            # Write raw bytes
            return audio_data.tobytes()
        else:
            frame = av.AudioFrame.from_ndarray(
                audio_data.reshape(1, -1),
                format="s16",
                layout="mono" if self.channels == 1 else "stereo",
            )
            frame.sample_rate = self.sample_rate

            frame.pts = self.pts
            self.pts += frame.samples

            packets = self.stream.encode(frame)
            for packet in packets:
                self.container.mux(packet)

            data = self.output_buffer.getvalue()
            self.output_buffer.seek(0)
            self.output_buffer.truncate(0)
            return data
