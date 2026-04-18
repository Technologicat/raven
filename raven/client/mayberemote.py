"""Transparent Raven-server support for some NLP components, with local (client-side) fallback.

NOTE: Before using this module, you must `raven.client.api.initialize` first.
"""

# TODO: This could be extended to cover all applicable server modules, but YAGNI. As of v0.2.4, we only need some specific modules to have this capability, for Raven-visualizer's importer.

__all__ = ["MaybeRemoteService",
           "Dehyphenator",
           "Embedder",
           "NLP",
           "STT",
           "TTS"]

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import io
from dataclasses import replace
from typing import List, Optional, Union

import numpy as np

import torch

from ..client import api
from ..client import config as client_config

from ..common import nlptools
from ..common.audio import codec as audio_codec
from ..common.audio.speech import stt as speech_stt
from ..common.audio.speech import tts as speech_tts


class MaybeRemoteService:
    def __init__(self,
                 allow_local: bool):
        """Transparent Raven-server support for some NLP models.

        Base class. Specific services derive from this.

        When a `MaybeRemoteService` is instantiated, it checks whether Raven-server
        can be reached at the URL set in `raven.client.config`.

        If the server is reached, the service enters remote mode. Any calls to the service
        call into the corresponding module on the server.

        If the server cannot be reached, the service enters local mode, and loads the model
        locally in the client process. Any calls to the service use the local model.

        In both cases, the API for calling the service is the same. Details depend
        on the service type; see derived classes.

        This is convenient for some apps, such as Raven-visualizer, which only use
        the server for some NLP work, so that they can run also without the server
        if those features are available locally. This makes using those apps easier,
        avoiding a need to start the server.

        But if the server is running, using its services will save VRAM, because
        the client avoids loading another copy of the same model that is already
        loaded on the server.

        `allow_local`: Whether to allow the local mode.

                       If the app can work without Raven-server, then it is recommended
                       to set this to `True`.

                       If `False`, and the server cannot be reached, raises `RuntimeError`.

                       The purpose of this flag is to provide a unified API for apps that need the
                       server for other purposes anyway. For such apps, loading the model locally
                       obviously won't help the client to run without the server.

                       The server may be running on another machine, so that if the server is
                       unreachable, causing the service to enter local mode, this would in turn
                       cause the client app to locally download and install a model that it doesn't need.

                       Obviously, the only solution that actually helps an app that needs the
                       server is to make the server reachable (e.g. start it if it was not running).

                       Avoiding extra model downloads is important, as these models can be multiple GB each.
        """
        self.server_available = api.raven_server_available()
        if self.server_available:
            server_modules = api.modules()
            logger.info(f"MaybeRemoteService.__init__: Connected to Raven-server at '{client_config.raven_server_url}'; available modules = {server_modules}.")
        else:
            server_modules = []
            if allow_local:
                logger.info(f"MaybeRemoteService.__init__: Could not connect to Raven-server at '{client_config.raven_server_url}'; model will be loaded locally.")
            else:
                logger.error(f"MaybeRemoteService.__init__: Could not connect to Raven-server at '{client_config.raven_server_url}', and `allow_local` is disabled. Cannot proceed.")
                raise RuntimeError(f"MaybeRemoteService.__init__: Could not connect to Raven-server at '{client_config.raven_server_url}', and `allow_local` is disabled. Cannot proceed.")
        self.server_modules = server_modules
        self._local_model = None

    def is_local(self) -> bool:
        """Return whether this service is in local mode."""
        return self._local_model is not None  # In local mode, each derived class loads the relevant local model.


class Dehyphenator(MaybeRemoteService):
    def __init__(self,
                 allow_local: bool,
                 model_name: Optional[str],
                 device_string: Optional[str]):
        """Dehyphenate broken text (e.g. as extracted from a PDF), via perplexity analysis using a character-level AI model for NLP.

        `allow_local`: See `MaybeRemoteService`.

        `model_name`: Required if `allow_local=True`. If local mode triggers, spaCy model name to load, e.g. "en_web_core_sm".

        `device_string`: Required if `allow_local=True`. If local mode triggers, passed to `raven.common.nlptools`.
        """
        super().__init__(allow_local)
        self.model_name = model_name
        self.device_string = device_string

        if "sanitize" in self.server_modules:
            logger.info(f"Dehyphenator.__init__: Using `sanitize` module on Raven-server at '{client_config.raven_server_url}'.")
        else:
            if self.server_available:
                logger.info(f"Dehyphenator.__init__: No `sanitize` module loaded on Raven-server at '{client_config.raven_server_url}', loading dehyphenator model locally.")
            self._local_model = nlptools.load_dehyphenator(model_name,
                                                           device_string)

    def dehyphenate(self,
                    text: Union[str, List[str]]) -> Union[str, List[str]]:
        """Dehyphenate `text`.

        `text`: one or more texts to dehyphenate.

        Returns `str` (one input) or `list` of `str` (more inputs).
        """
        if self._local_model is None:
            dehyphenated_text = api.sanitize_dehyphenate(text)
        else:
            dehyphenated_text = nlptools.dehyphenate(self._local_model,
                                                     text)
        return dehyphenated_text


class Embedder(MaybeRemoteService):
    def __init__(self,
                 allow_local: bool,
                 model_name: str,
                 device_string: Optional[str],
                 dtype: Optional[Union[str, torch.dtype]]):
        """Semantic embeddings (for e.g. vector storage).

        `allow_local`: See `MaybeRemoteService`.

        `model_name`: HuggingFace model name of the embedding model to load.

                      In remote mode, this can also be an embeddings role in `raven.server.config` (e.g. "default", "qa").

        `device_string`: Required if `allow_local=True`. If local mode triggers, passed to `raven.common.nlptools`.

        `dtype`: Required if `allow_local=True`. If local mode triggers, passed to `raven.common.nlptools`.
        """
        super().__init__(allow_local)
        self.model_name = model_name
        self.device_string = device_string
        self.dtype = dtype

        if "embeddings" in self.server_modules:
            logger.info(f"Embedder.__init__: Using `embeddings` module on Raven-server at '{client_config.raven_server_url}'.")
        else:
            if self.server_available:
                logger.info(f"Embedder.__init__: No `embeddings` module loaded on Raven-server at '{client_config.raven_server_url}', loading semantic embedding model locally.")
            self._local_model = nlptools.load_embedder(model_name,
                                                       device_string,
                                                       dtype)
        # Fail-fast; especially important with remote model. If the requested embeddings model isn't loaded on the server, any attempt to encode will HTTP 400.
        self.encode(["The quick brown fox jumps over the lazy dog"])

    def encode(self, text: Union[str, List[str]]) -> np.array:
        """Embed `text` (containing one or more texts).

        Each text produces a vector.
        """
        if self._local_model is None:
            vectors = api.embeddings_compute(text=text,
                                             model=self.model_name)  # -> np.array
        else:
            vectors = nlptools.embed_sentences(embedder=self._local_model,
                                               text=text)  # -> list, or list of lists (for easy JSONability)
            vectors = np.array(vectors)  # ...so we must convert to `np.array` ourselves
        return vectors


class NLP(MaybeRemoteService):
    def __init__(self,
                 allow_local: bool,
                 model_name: Optional[str],
                 device_string: Optional[str]):
        """spaCy NLP pipeline.

        `allow_local`: See `MaybeRemoteService`.

        `model_name`: Required if `allow_local=True`. If local mode triggers, spaCy model name to load, e.g. "en_web_core_sm".

        `device_string`: Required if `allow_local=True`. If local mode triggers, passed to `raven.common.nlptools`.
        """
        super().__init__(allow_local)
        self.model_name = model_name
        self.device_string = device_string

        if "natlang" in self.server_modules:
            logger.info(f"NLP.__init__: Using `natlang` module on Raven-server at '{client_config.raven_server_url}'.")
        else:
            if self.server_available:
                logger.info(f"NLP.__init__: No `natlang` module loaded on Raven-server at '{client_config.raven_server_url}', loading spaCy NLP model locally.")
            self._local_model = nlptools.load_spacy_pipeline(model_name,
                                                             device_string)

    def analyze(self,
                text: Union[str, List[str]],
                pipes: Optional[List[str]] = None) -> List[List["spacy.tokens.token.Token"]]:  # noqa: F821 -- type annotation only, avoid importing spaCy here
        """Perform NLP analysis on `text`.

        `pipes`: If provided, enable only the listed pipes. Which ones exist depend on the loaded spaCy model.
                 If not provided, use the model's default pipes.

        Returns a `list` of spaCy documents (even if just one `text`).

        In remote mode, the pipeline at the client side is blank, but the tokens have all the usual data.
        """
        if self._local_model is None:
            docs = api.natlang_analyze(text,
                                       pipes)
        else:
            docs = nlptools.spacy_analyze(self._local_model,
                                          text,
                                          pipes)
        return docs


class STT(MaybeRemoteService):
    def __init__(self,
                 allow_local: bool,
                 model_name: Optional[str] = None,
                 device_string: Optional[str] = None,
                 dtype: Optional[Union[str, torch.dtype]] = None):
        """Speech-to-text (Whisper).

        `allow_local`: See `MaybeRemoteService`. Note that Whisper models are heavy
                       (74 MB for whisper-base, 800 MB for whisper-large-v3-turbo);
                       apps that only occasionally need STT should prefer
                       `allow_local=False` and fail clearly when the server is down
                       rather than trigger a multi-hundred-megabyte download.

        `model_name`: Required if `allow_local=True`. HuggingFace model name, e.g.
                      `"openai/whisper-base"`, `"openai/whisper-large-v3-turbo"`.

        `device_string`: Required if `allow_local=True`. E.g. `"cpu"`, `"cuda:0"`.

        `dtype`: Required if `allow_local=True`. E.g. `torch.float32`, `torch.float16`,
                 or a string like `"float32"`.
        """
        super().__init__(allow_local)
        self.model_name = model_name
        self.device_string = device_string
        self.dtype = dtype

        # Whisper's canonical input sample rate. In local mode we read this off the
        # loaded model; in remote mode we assume the canonical value so `sample_rate`
        # is populated before the first call. If upstream ever changes this, both
        # modes fail loud (local: validation in `speech_stt.transcribe`; remote: the
        # server mirrors the same value).
        self.sample_rate = 16000

        if "stt" in self.server_modules:
            logger.info(f"STT.__init__: Using `stt` module on Raven-server at '{client_config.raven_server_url}'.")
        else:
            if self.server_available:
                logger.info(f"STT.__init__: No `stt` module loaded on Raven-server at '{client_config.raven_server_url}', loading Whisper model locally.")
            self._local_model = speech_stt.load_stt_model(model_name=model_name,
                                                          device_string=device_string,
                                                          dtype=dtype)
            self.sample_rate = self._local_model.sample_rate

    def transcribe(self,
                   audio: np.ndarray,
                   sample_rate: int,
                   prompt: Optional[str] = None,
                   language: Optional[str] = None) -> str:
        """Transcribe mono `audio` to text.

        `audio`: rank-1 float numpy array, mono, samples in [-1, 1] (standard float
                 audio convention). The remote path's s16 cast happens in
                 `api.stt_transcribe_array` at the wire-format boundary; the local
                 path feeds Whisper's processor directly.

        `sample_rate`: sample rate of `audio`, any rate. Rate conversion to Whisper's
                       native 16 kHz happens inside the engine (locally via
                       `raven.common.audio.speech.stt.transcribe`, or remotely via
                       the server's `audio_codec.decode` call on the uploaded audio).

        `prompt`, `language`: see `raven.common.audio.speech.stt.transcribe`.
        """
        if self._local_model is None:
            # Remote: `api.stt_transcribe_array` accepts float or s16 and handles the cast internally.
            # The server resamples during its container-decode pass, so any sample rate works.
            return api.stt_transcribe_array(audio, sample_rate=sample_rate, prompt=prompt, language=language)

        return speech_stt.transcribe(self._local_model,
                                     audio=audio,
                                     sample_rate=sample_rate,
                                     prompt=prompt,
                                     language=language)


class TTS(MaybeRemoteService):
    def __init__(self,
                 allow_local: bool,
                 repo_id: Optional[str] = None,
                 device_string: Optional[str] = None,
                 lang_code: str = "a"):
        """Text-to-speech (Kokoro).

        `allow_local`: See `MaybeRemoteService`. Kokoro-82M weighs ~360 MB;
                       apps that only occasionally need TTS should prefer
                       `allow_local=False`.

        `repo_id`: Required if `allow_local=True`. HuggingFace repo, e.g.
                   `"hexgrad/Kokoro-82M"`.

        `device_string`: Required if `allow_local=True`. E.g. `"cpu"`, `"cuda:0"`.

        `lang_code`: phonemizer language code. Word-level metadata currently
                     supports English only (`"a"` or `"b"`). See
                     `raven.common.audio.speech.tts.load_tts_pipeline` for the
                     full list.
        """
        super().__init__(allow_local)
        self.repo_id = repo_id
        self.device_string = device_string
        self.lang_code = lang_code

        self.sample_rate = speech_tts.SAMPLE_RATE

        if "tts" in self.server_modules:
            logger.info(f"TTS.__init__: Using `tts` module on Raven-server at '{client_config.raven_server_url}'.")
        else:
            if self.server_available:
                logger.info(f"TTS.__init__: No `tts` module loaded on Raven-server at '{client_config.raven_server_url}', loading Kokoro pipeline locally.")
            self._local_model = speech_tts.load_tts_pipeline(repo_id=repo_id,
                                                             device_string=device_string,
                                                             lang_code=lang_code)

    def list_voices(self) -> List[str]:
        """List installed voices. Remote uses the server's /api/tts/list_voices; local scans the modelsdir."""
        if self._local_model is None:
            return api.tts_list_voices()
        return speech_tts.get_voices(self._local_model)

    def synthesize(self,
                   voice: str,
                   text: str,
                   speed: float = 1.0,
                   get_metadata: bool = True) -> speech_tts.TTSResult:
        """Synthesize `text` to a `TTSResult` with float32 mono audio and optional word timings.

        Uniform return type across modes:
          - `audio`: float32 in [-1, 1], at `self.sample_rate` (24 kHz).
          - `word_metadata`: list of `WordTiming` post-processed for lipsync —
            `speech_tts.clean_timestamps(for_lipsync=True)` +
            `speech_tts.expand_phoneme_diphthongs` applied in both modes.

        Remote mode: delegates to `api.tts_prepare`, which handles the HTTP call,
        timestamp-JSON decoding, and the post-processing pipeline. We decode the
        returned MP3 back to float here (note: lossy codec round-trip — sample
        values differ slightly from the server's internal float output, inaudible
        but not bit-identical).

        Local mode: delegates to `speech_tts.synthesize`, then applies the same
        post-processing the remote `tts_prepare` path does, to keep behaviour
        uniform.
        """
        if self._local_model is None:
            prep = api.tts_prepare(voice=voice, text=text, speed=speed, get_metadata=get_metadata)
            if prep is None:
                # Blank input, or metadata requested but no phonemes generated.
                # Mirror the common-layer `synthesize` empty-result shape.
                return speech_tts.TTSResult(audio=np.zeros(0, dtype=np.float32),
                                            sample_rate=self.sample_rate,
                                            duration=0.0,
                                            word_metadata=[] if get_metadata else None)

            # MP3 → float32 mono at Kokoro's native 24 kHz.
            _metadata, audio = audio_codec.decode(io.BytesIO(prep["audio_bytes"]),
                                                  target_sample_format="fltp",
                                                  target_sample_rate=self.sample_rate,
                                                  target_layout="mono")
            audio = audio.astype(np.float32, copy=False)

            # `tts_prepare` already applied clean_timestamps + expand_phoneme_diphthongs.
            word_metadata = prep.get("timestamps") if get_metadata else None

            return speech_tts.TTSResult(audio=audio,
                                        sample_rate=self.sample_rate,
                                        duration=len(audio) / self.sample_rate,
                                        word_metadata=word_metadata)

        result = speech_tts.synthesize(self._local_model,
                                       voice=voice,
                                       text=text,
                                       speed=speed,
                                       get_metadata=get_metadata)
        if result.word_metadata is not None:
            cleaned = speech_tts.clean_timestamps(result.word_metadata, for_lipsync=True)
            cleaned = speech_tts.expand_phoneme_diphthongs(cleaned)
            result = replace(result, word_metadata=cleaned)
        return result
