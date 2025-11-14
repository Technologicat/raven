"""Audio handling utilities."""

__all__ = ["encode_audio", "decode_audio"]

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import io
from typing import BinaryIO, Generator, List, Optional, Tuple, Union

import numpy as np

import av

from .numutils import si_prefix

from ..vendor.kokoro_fastapi.streaming_audio_writer import StreamingAudioWriter

def encode_audio(audio_data: Union[np.array, List[np.array]],
                 format: str,
                 sample_rate: int,
                 stream: bool = False) -> Union[bytes, Generator[bytes, None, None]]:
    """Encode audio data from an in-memory array into an audio file format.

    `audio_data`: Audio data in s16 format (signed 16-bit).

                  One chunk:

                  rank-1 `np.array` of shape `[n]` for mono,
                  rank-2 `np.array` of shape `[n, 2]` for stereo.

                  Several chunks:

                  List of rank-1 arrays, each may have different length.
                  List of rank-2 arrays, each may have different length,
                  as long as each has two channels (last dimension is length 2).

    `format`: Name of output format. Any audio format supported by PyAV, e.g. "mp3".

    `sample_rate`: Sample rate used for interpreting the audio data stored in the array.

    `stream`: If `False` (default), return a `bytes` object containing the encoded audio file.

              If `True`, return a generator, which yields a separate `bytes` object for
              the part of the encoded audio file that corresponds to one input array.
              Finally, it yields one more `bytes` object, resulting from finalization.
    """
    if not isinstance(audio_data, list):
        audio_data = [audio_data]

    # `StreamingAudioWriter` will reshape the input data array, but it needs to be of a compatible size, and something that reshapes sensibly.
    dims = np.shape(audio_data[0])
    if len(dims) == 1:
        channels = 1
    elif len(dims) == 2 and dims[-1] == 1:
        channels = 1
    elif len(dims) == 2 and dims[-1] == 2:
        channels = 2
    else:
        error_msg = f"encode_audio only supports mono (rank-1 [n] or rank-2 [n, 1]) or stereo (rank-2 [n, 2]) arrays; got `audio_data` of shape {dims}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # TODO: Doesn't really need to be streaming here. We could just use PyAV directly, like `decode_audio` does.
    audio_encoder = StreamingAudioWriter(format=format,
                                         sample_rate=sample_rate,
                                         channels=channels)
    if stream:
        def streamer():
            for audio_chunk in audio_data:
                yield audio_encoder.write_chunk(audio_chunk)
            yield audio_encoder.write_chunk(finalize=True)
        return streamer
    else:
        audio_buffer = io.BytesIO()
        for audio_chunk in audio_data:
            audio_buffer.write(audio_encoder.write_chunk(audio_chunk))
        audio_buffer.write(audio_encoder.write_chunk(finalize=True))
        audio_bytes = audio_buffer.getvalue()
        return audio_bytes

def decode_audio(stream: BinaryIO,
                 target_sample_format: Optional[str] = None,
                 target_sample_rate: Optional[int] = None,
                 target_layout: Optional[str] = None) -> Tuple[dict, np.array]:
    """Decode an audio stream into an in-memory array, with optional resampling.

    `stream`: Filelike (e.g. `io.BytesIO`) in any audio file format supported by PyAV
              (e.g. mp3 is fine).

    `target_sample_format`: Desired sample format for resampling.

                            If not specified, the output will be in the same sample format as `stream`.

                            NOTE: **sample** format, NOT file format. Some common sample formats are:

                              - "s16": signed 16-bit,
                              - "fltp": floating point.

                            https://pyav.org/docs/stable/api/audio.html

    `target_sample_rate`: Desired sample rate for resampling.

                          If not specified, the output will be at the same sample rate as `stream`.

                          E.g. some AI models have been trained at a specific sample rate,
                          and will only accept audio input at that sample rate.

    `target_layout`: Desired channel layout, e.g. "mono" or "stereo".

                     If not specified, the output will have the same layout as `stream`.

                     https://pyav.org/docs/stable/api/audio.html#module-av.audio.resampler

    Returns the tuple `(metadata, audio_data)`, where:

        - `metadata` is a dictionary with the following items:
            - `input_sample_format` is the original sample format of the input (e.g. "fltp").
            - `input_sample_rate` is the original sample rate of the input.
            - `input_layout` is the original layout of the input (e.g. "mono" or "stereo").

            These are useful mainly when you don't specify the corresponding `target_*` parameters,
            and want to get this information from the audio file instead.

        - `audio_data` is a rank-1 `np.array` containing PCM audio data with the desired sample format,
          sample rate, and layout.
    """
    logger.info("decode_audio: Decoding received audio file.")

    audio_data = io.BytesIO()
    audio_data.write(stream.read())  # copy input into our own in-memory buffer, just in case
    audio_data.seek(0)

    # https://stackoverflow.com/questions/73198826/trying-to-get-audio-raw-data-and-print-it-using-pyav
    # https://pyav.org/docs/stable/api/audio.html
    # https://pyav.org/docs/stable/api/_globals.html?highlight=open#av.open
    # https://pyav.org/docs/stable/overview/caveats.html#garbage-collection
    data = None
    resampler = None
    input_sample_format = None
    input_sample_rate = None
    input_layout = None
    with av.open(audio_data) as container:
        # According to documentation, `av.time_base` is in fractional seconds:
        #     https://pyav.org/docs/stable/api/time.html
        # The name isn't uppercase; correct name (lowercase) mentioned here:
        #     https://pyav.org/docs/stable/api/container.html?highlight=time_base#av.container.InputContainer.seek
        # print(av.time_base)  # 1000000
        logger.info(f"decode_audio: Detected container type '{container.format.name}', bitrate {si_prefix(container.bit_rate)}bps, duration {container.duration / av.time_base:0.6g}s, size {si_prefix(container.size)}B.")
        for packet in container.demux():
            for frame in packet.decode():
                if isinstance(frame, av.audio.frame.AudioFrame):
                    # # DEBUG
                    # print(frame,
                    #       frame.format,  # The audio sample format.
                    #       frame.sample_rate,  # Sample rate of the audio data, in samples per second.
                    #       frame.layout,  # The audio channel layout.
                    #       frame.layout.channels,  # A tuple of AudioChannel objects.
                    #       frame.samples)  # Number of audio samples (per channel).
                    if input_sample_rate is None:  # first audio frame in the stream?
                        # Detect sample format et al.
                        input_sample_format = frame.format.name
                        input_sample_rate = frame.sample_rate
                        input_layout = frame.layout.name

                        plural_s = "s" if frame.samples != 1 else ""
                        logger.info(f"decode_audio: Detected input sample format '{frame.format.name}', sample rate {frame.sample_rate}, audio frame size {frame.samples} sample{plural_s} per channel, layout '{frame.layout.name}', channels: { {channel.name: channel.description for channel in frame.layout.channels} }.")

                        target_sample_format_str = f"'{target_sample_format}'" if (target_sample_format is not None) else "same as input"
                        target_sample_rate_str = f"{target_sample_rate} Hz" if (target_sample_rate is not None) else "same as input"
                        target_layout_str = f"'{target_layout}'" if (target_layout is not None) else "same as input"
                        logger.info(f"decode_audio: Decoding to output sample format {target_sample_format_str}, sample rate {target_sample_rate_str}, layout {target_layout_str}.")

                    # Initialize optional parameters
                    # TODO: Add support for specifying these as `av.audio.format.AudioFormat` and `av.audio.layout.AudioLayout`, for more fine-grained control. The question is how to detect if they match the input.
                    if target_sample_format is None:
                        target_sample_format = input_sample_format
                    if target_sample_rate is None:  # `None`: keep input sample rate
                        target_sample_rate = input_sample_rate
                    if target_layout is None:
                        target_layout = input_layout

                    # Write to output array, with resampling if needed
                    if (target_sample_format != input_sample_format) or (target_sample_rate != input_sample_rate) or (target_layout != input_layout):
                        if not resampler:  # create resampler at first frame (we need the parameters, which we may autodetect from the stream)
                            resampler = av.audio.resampler.AudioResampler(format=target_sample_format,
                                                                          layout=target_layout,
                                                                          rate=target_sample_rate)
                        resampled_frames = resampler.resample(frame)
                        for resampled_frame in resampled_frames:
                            array = resampled_frame.to_ndarray()[0]
                            if data is not None:
                                data = np.concatenate([data, array])
                            else:
                                data = array
                    else:
                        array = frame.to_ndarray()[0]
                        if data is not None:
                            data = np.concatenate([data, array])
                        else:
                            data = array

    logger.info(f"decode_audio: Decoded {len(data) / target_sample_rate:0.6g}s of audio.")
    metadata = {"input_sample_format": input_sample_format,
                "input_sample_rate": input_sample_rate,
                "input_layout": input_layout}
    return metadata, data
