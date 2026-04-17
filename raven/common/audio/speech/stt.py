"""Speech-to-text engine (Whisper), as an in-process library.

Transport concerns (HTTP upload handling, audio container decoding, tqdm
progress reporting) live in the server wrapper at `raven.server.modules.stt`.
Client-side remote/local dispatch lives in `raven.client.mayberemote.STT`.
"""

__all__ = ["STTModel", "load_stt_model", "transcribe"]

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np

import torch
import transformers

from ...hfutil import maybe_install_models
from .. import resample as audio_resample


@dataclass
class STTModel:
    """Loaded Whisper model + its processor + the device and dtype it lives on.

    `sample_rate` is a convenience mirror of `processor.feature_extractor.sampling_rate`;
    callers need it to know what rate to resample to before calling `transcribe`.
    """
    model: transformers.AutoModelForSpeechSeq2Seq
    processor: transformers.AutoProcessor
    device: torch.device
    dtype: torch.dtype
    sample_rate: int


# Cache, keyed by (model_name, device_string, str(dtype)).
# Same pattern as `nlptools._spacy_pipelines`.
_stt_models: dict[tuple[str, str, str], STTModel] = {}


def load_stt_model(model_name: str,
                   device_string: str,
                   dtype: Union[str, torch.dtype]) -> STTModel:
    """Load (and cache) a Whisper model from HuggingFace.

    `model_name`: e.g. `"openai/whisper-base"`, `"openai/whisper-large-v3-turbo"`.
                  Auto-downloaded via `maybe_install_models` if not present.

    `device_string`: e.g. `"cpu"`, `"cuda:0"`. Passed to `torch.device`.

    `dtype`: e.g. `torch.float32`, `torch.float16`. Accepts the string name too
             (e.g. `"float32"`) for ergonomic use from config files.

    Repeat calls with the same `(model_name, device_string, str(dtype))` triple
    return the cached `STTModel` — the model is loaded at most once per process.
    """
    cache_key = (model_name, device_string, str(dtype))
    if (cached := _stt_models.get(cache_key)) is not None:
        logger.info(f"load_stt_model: returning cached model for {cache_key}")
        return cached

    logger.info(f"load_stt_model: ensuring model '{model_name}' is installed.")
    maybe_install_models(model_name)

    logger.info(f"load_stt_model: loading '{model_name}' on '{device_string}' with dtype {dtype}.")
    device = torch.device(device_string)
    processor = transformers.AutoProcessor.from_pretrained(model_name)
    model = transformers.AutoModelForSpeechSeq2Seq.from_pretrained(model_name,
                                                                   dtype=dtype,
                                                                   use_safetensors=True).to(device)

    stt_model = STTModel(model=model,
                         processor=processor,
                         device=device,
                         dtype=dtype if isinstance(dtype, torch.dtype) else getattr(torch, str(dtype)),
                         sample_rate=processor.feature_extractor.sampling_rate)

    _stt_models[cache_key] = stt_model
    return stt_model


def transcribe(stt_model: STTModel,
               audio: np.ndarray,
               sample_rate: int,
               prompt: Optional[str] = None,
               language: Optional[str] = None,
               progress_callback: Optional[Callable[[int, int], None]] = None) -> str:
    """Transcribe mono audio to text.

    `audio`: rank-1 float numpy array, mono. Any sample rate — this function
             resamples to the model's native rate (Whisper is 16 kHz) via
             `raven.common.audio.resample.resample` if they don't match.

    `sample_rate`: the sample rate of `audio`. Must match the actual rate of the
                   samples — a raw numpy array carries no rate metadata, so this
                   parameter is how the function knows what the bytes mean. Passing
                   the wrong value produces speed-shifted and resampled audio, which
                   Whisper then transcribes as gibberish.

    `prompt`: optional conditioning text. See `raven.server.app.api_stt_transcribe`
              for Whisper prompting guidance (list rare proper names, set context,
              or nudge the transcription style). Whisper uses only the last 224 tokens.

    `language`: optional ISO-639-1 code (e.g. `"en"`, `"fi"`). `None` → autodetect.

    `progress_callback`: optional `(current_frame, total_frames) -> None` callback,
                         invoked during decoding. Typical use: a tqdm progress bar
                         (the server wrapper supplies one; library callers can pass
                         whatever UI they like, or omit for silent operation).

    Returns the transcribed text, with per-segment outputs joined into a single
    whitespace-stripped string.
    """
    if sample_rate != stt_model.sample_rate:
        logger.info(f"transcribe: resampling audio from {sample_rate} Hz → {stt_model.sample_rate} Hz before inference.")
        audio = audio_resample.resample(audio, from_rate=sample_rate, to_rate=stt_model.sample_rate)
        sample_rate = stt_model.sample_rate

    prompt_log = f"'{prompt}'" if prompt is not None else None
    language_log = f"'{language}'" if language is not None else None
    logger.info(f"transcribe: request received with prompt = {prompt_log}, language = {language_log}.")

    # See:
    # https://huggingface.co/docs/transformers/en/model_doc/whisper
    # https://huggingface.co/openai/whisper-large-v3-turbo
    # Parameters of `transformers.models.whisper.generation_whisper.WhisperGenerationMixin.generate`
    # Parameters of `transformers.models.whisper.modeling_whisper.WhisperForConditionalGeneration.forward` (some are passed here by `generate`)
    kwargs = {  # "max_new_tokens": 448,  # Whisper maximum - better to just use the default, since we don't know the number of "special start tokens", or the number of overlap tokens in long-form transcription.
              "return_timestamps": True,  # Needed for long-form transcription. To actually return them, may also need `return_dict=True`? (see the `forward` method mentioned above)
              "compression_ratio_threshold": 1.35,  # zlib compression ratio threshold (in token space)
              "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
              "logprob_threshold": -1.0,
              # "no_speech_threshold": 0.6,  # Doesn't work, causes crash inside `transformers` (disregard repo name, actually this bug has nothing to do with PEFT): https://github.com/huggingface/peft/issues/1988
              "condition_on_prev_tokens": True}

    if language is not None:
        kwargs["language"] = language

    if prompt is not None:
        # IMPORTANT and not well documented: send the prompt tensor to the correct device, it will be used as-is.
        prompt_ids = stt_model.processor.get_prompt_ids(prompt, return_tensors="pt").to(stt_model.device)
        kwargs["prompt_ids"] = prompt_ids
        kwargs["prompt_condition_type"] = "all-segments"

    logger.info("transcribe: preprocessing audio features.")
    processor = stt_model.processor
    inputs = processor(audio=audio,
                       sampling_rate=stt_model.sample_rate,
                       return_tensors="pt",
                       truncation=False,
                       padding="longest",
                       return_attention_mask=True).to(stt_model.device, dtype=stt_model.dtype)
    # https://github.com/huggingface/transformers/issues/30740
    if inputs.input_features.shape[-1] < 3000:
        logger.info(f"transcribe: feature stream length {inputs.input_features.shape[-1]} — short-form path.")
        # We in-fact have short-form -> pre-process accordingly.
        inputs = processor(audio=audio,
                           sampling_rate=stt_model.sample_rate,
                           return_tensors="pt",
                           return_attention_mask=True).to(stt_model.device, dtype=stt_model.dtype)
    else:
        logger.info(f"transcribe: feature stream length {inputs.input_features.shape[-1]} — long-form path.")

    # Adapt simple `(current, total)` callback to HF's batch-tensor convention.
    # p_batch is shape (n, 2) for batch size n. Column 0 is current frame, column 1 is total frames.
    # We pick the batch item furthest from done, to mirror tqdm's "worst-case" feel.
    if progress_callback is not None:
        def hf_monitor_progress(p_batch):
            i = torch.argmax(p_batch[:, 1])
            p = p_batch[i].detach().cpu()
            progress_callback(int(p[0]), int(p[1]))
        kwargs["monitor_progress"] = hf_monitor_progress

    logger.info("transcribe: generating tokens.")
    pred_ids = stt_model.model.generate(**inputs, **kwargs)

    if progress_callback is not None:
        progress_callback(1, 1)  # signal completion (current == total)

    logger.info("transcribe: decoding tokens.")
    pred_text = processor.batch_decode(pred_ids, skip_special_tokens=True, decode_with_timestamps=False)
    pred_text = " ".join(item.strip() for item in pred_text)

    logger.info("transcribe: done.")
    return pred_text
