"""Speech to text."""

__all__ = ["init_module", "is_available", "speech_to_text"]

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import importlib
import traceback
import transformers
from typing import Optional, List, Union

from colorama import Fore, Style

import torch

from tqdm import tqdm

from ...common import audioutils
from ...common import hfutil

server_config = None
model = None
processor = None
# pipe = None
loaded_model_device = None
loaded_model_dtype = None

def init_module(config_module_name: str, device_string: str, dtype: Union[str, torch.dtype]) -> None:
    global server_config
    global model
    global processor
    # global pipe
    global loaded_model_device
    global loaded_model_dtype
    print(f"Initializing {Fore.GREEN}{Style.BRIGHT}stt{Style.RESET_ALL} on device '{Fore.GREEN}{Style.BRIGHT}{device_string}{Style.RESET_ALL}'...")
    try:
        server_config = importlib.import_module(config_module_name)
        model_name = server_config.speech_recognition_model

        logger.info(f"init_module: Ensuring model '{Fore.GREEN}{Style.BRIGHT}{model_name}{Style.RESET_ALL}' is installed...")
        hfutil.maybe_install_models(server_config.speech_recognition_model)

        logger.info(f"init_module: Loading model '{Fore.GREEN}{Style.BRIGHT}{model_name}{Style.RESET_ALL}' on device '{Fore.GREEN}{Style.BRIGHT}{device_string}{Style.RESET_ALL}'...")
        loaded_model_device = torch.device(device_string)
        loaded_model_dtype = dtype
        processor = transformers.AutoProcessor.from_pretrained(model_name)
        model = transformers.AutoModelForSpeechSeq2Seq.from_pretrained(model_name,
                                                                       dtype=dtype,
                                                                       use_safetensors=True).to(loaded_model_device)
    #     pipe = transformers.pipeline("automatic-speech-recognition",
    #                                  model=model,
    #                                  tokenizer=processor.tokenizer,
    #                                  feature_extractor=processor.feature_extractor,
    #                                  chunk_length_s=30,
    #                                  batch_size=16,  # batch size for inference - set based on your device
    #                                  dtype=dtype,
    #                                  device=loaded_model_device)
    except Exception as exc:
        print(f"{Fore.RED}{Style.BRIGHT}Internal server error during init of module 'embeddings'.{Style.RESET_ALL} Details follow.")
        traceback.print_exc()
        logger.error(f"init_module: failed: {type(exc)}: {exc}")
        server_config = None
        model = None
        processor = None
        # pipe = None
        loaded_model_device = None
        loaded_model_dtype = None

def is_available() -> bool:
    """Return whether this module is up and running."""
    return (server_config is not None)

# TODO: the input is a flask.request.file.stream; what's the type of that?
def speech_to_text(stream,
                   prompt: Optional[str],
                   language: Optional[str]) -> List[str]:
    prompt_log_msg_str = f"'{prompt}'" if prompt is not None else None
    language_log_msg_str = f"'{language}'" if language is not None else None
    logger.info(f"speech_to_text: Request received with prompt = {prompt_log_msg_str}, language = {language_log_msg_str}.")

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
              "condition_on_prev_tokens": True,
             }

    if language is not None:  # input audio language (optional)
        kwargs["language"] = language

    if prompt is not None:  # input prompt (optional), e.g. to spell rare proper names correctly
        prompt_ids = processor.get_prompt_ids(prompt, return_tensors="pt").to(loaded_model_device)  # IMPORTANT and not well documented: send the prompt tensor to the correct device, it will be used as-is.
        kwargs["prompt_ids"] = prompt_ids
        kwargs["prompt_condition_type"] = "all-segments"
        # kwargs["max_new_tokens"] -= (prompt_ids.shape[0] + 3)  # also the prompt tokens and "special start tokens" (TODO: whatever those are?) count toward the output limit

    unused_metadata, numpy_audio_data = audioutils.decode_audio(stream,
                                                                target_sample_format="fltp",  # float
                                                                target_sample_rate=processor.feature_extractor.sampling_rate,
                                                                target_layout="mono")

    # # With high-level pipeline API
    # # TODO: Even the high-level API doesn't seem to return anything but the "text" field, though `return_timestamps=True`.
    # result = pipe(numpy_audio_data,
    #               generate_kwargs=kwargs)
    # pred_text = result["text"].strip()

    # With low-level API -->
    logger.info("speech_to_text: Preprocessing audio for STT.")
    inputs = processor(audio=numpy_audio_data,
                       sampling_rate=processor.feature_extractor.sampling_rate,
                       return_tensors="pt",
                       truncation=False,
                       padding="longest",
                       return_attention_mask=True).to(loaded_model_device, dtype=loaded_model_dtype)
    # https://github.com/huggingface/transformers/issues/30740
    if inputs.input_features.shape[-1] < 3000:
        logger.info(f"speech_to_text: Length of feature stream is {inputs.input_features.shape[-1]}. Using short-form transcription.")
        # We in-fact have short-form -> pre-process accordingly.
        inputs = processor(audio=numpy_audio_data,
                           sampling_rate=processor.feature_extractor.sampling_rate,
                           return_tensors="pt",
                           return_attention_mask=True).to(loaded_model_device, dtype=loaded_model_dtype)
    else:
        logger.info(f"speech_to_text: Length of feature stream is {inputs.input_features.shape[-1]}. Using long-form transcription.")

    logger.info("speech_to_text: Transcribing.")
    # Transcription progress callback.
    #
    # NOTE: As of 11/2025, requires a recent `transformers` (4.57.1 has this).
    #
    # The function takes a tensor argument p of shape (n, 2), where n is the batch size.
    # p[i, 0] contains the index of the audio frame that is currently being transcribed
    # for batch item i. p[i, 1] contains the total number of frames for batch item i.
    # No return value is expected.
    with tqdm(desc="Transcribing", leave=True) as pbar:
        def monitor_progress(p_batch):
            i = torch.argmax(p_batch[:, 1])
            p = p_batch[i].detach().cpu()
            pbar.total = int(p[1])
            pbar.n = int(p[0])
            pbar.refresh()
        def finish_progress():
            pbar.n = pbar.total
            pbar.refresh()
        pred_ids = model.generate(**inputs, **kwargs, monitor_progress=monitor_progress)
        finish_progress()

    logger.info("speech_to_text: Decoding transcribed tokens into text.")
    pred_text = processor.batch_decode(pred_ids, skip_special_tokens=True, decode_with_timestamps=False)
    pred_text = ' '.join([item.strip() for item in pred_text])
    # <-- end with low-level API

    logger.info("speech_to_text: All done.")
    return pred_text
