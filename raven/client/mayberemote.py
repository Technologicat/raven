"""Transparent Raven-server support for some NLP and imagefx components, with local (client-side) fallback.

NOTE: Before using this module, you must `raven.client.api.initialize` first.
"""

__all__ = ["MaybeRemoteService",
           "Classifier",
           "Dehyphenator",
           "Embedder",
           "NLP",
           "Postprocessor",
           "STT",
           "Translator",
           "TTS",
           "Upscaler"]

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import threading
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

import torch

from unpythonic.env import env as _envcls

from ..client import api
from ..client import config as client_config
from ..client import tts as client_tts  # for `play_encoded_with_avatar_lipsync` (used by `TTS.speak_lipsynced` local mode)
from ..client import util as client_util  # for the singleton audio_player / task_manager

from ..common import nlptools
from ..common.audio.speech import datatypes as speech_datatypes
from ..common.audio.speech import playback as speech_playback
from ..common.audio.speech import stt as speech_stt
from ..common.audio.speech import tts as speech_tts
from ..common.video.postprocessor import Postprocessor as _LocalPostprocessor
from ..common.video.upscaler import Upscaler as _LocalUpscaler


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
                msg = f"MaybeRemoteService.__init__: Could not connect to Raven-server at '{client_config.raven_server_url}', and `allow_local` is disabled. Cannot proceed."
                logger.error(msg)
                raise RuntimeError(msg)
        self.server_modules = server_modules
        self._local_model = None

    def is_local(self) -> bool:
        """Return whether this service is in local mode."""
        return self._local_model is not None  # In local mode, each derived class loads the relevant local model.


class Classifier(MaybeRemoteService):
    def __init__(self,
                 allow_local: bool,
                 model_name: Optional[str] = None,
                 device_string: Optional[str] = None,
                 dtype: Optional[Union[str, torch.dtype]] = None):
        """Text sentiment classification (distilBERT-family, 28-emotion by default).

        `allow_local`: See `MaybeRemoteService`.

        `model_name`: Required if `allow_local=True`. HuggingFace model name, e.g.
                      `"joeddav/distilbert-base-uncased-go-emotions-student"`. In remote
                      mode the server decides which model is loaded; the argument is
                      ignored.

        `device_string`: Required if `allow_local=True`. E.g. `"cpu"`, `"cuda:0"`.

        `dtype`: Required if `allow_local=True`. E.g. `torch.float32`, or a string `"float32"`.
        """
        super().__init__(allow_local)
        self.model_name = model_name
        self.device_string = device_string
        self.dtype = dtype

        if "classify" in self.server_modules:
            logger.info(f"Classifier.__init__: Using `classify` module on Raven-server at '{client_config.raven_server_url}'.")
        else:
            if self.server_available:
                logger.info(f"Classifier.__init__: No `classify` module loaded on Raven-server at '{client_config.raven_server_url}', loading classifier model locally.")
            self._local_model = nlptools.load_classifier(model_name,
                                                         device_string,
                                                         dtype)

    def classify(self, text: str) -> Dict[str, float]:
        """Classify the sentiment of `text`.

        Returns `{label: score, ...}`, sorted by score descending (the iteration order
        of regular `dict` preserves insertion order). Same shape in both local and
        remote modes — the local path normalizes nlptools' list-of-dicts form to match
        the wire format.
        """
        if not self.is_local():
            return api.classify(text)
        sorted_records = nlptools.classify(self._local_model, text)  # [{"label": ..., "score": ...}, ...]
        return {record["label"]: record["score"] for record in sorted_records}

    def labels(self) -> List[str]:
        """List the emotion labels the classifier can assign.

        The set is fixed by the underlying model (e.g. 28 emotions for the default
        go-emotions model). Sorted alphabetically.
        """
        if not self.is_local():
            return api.classify_labels()
        return sorted(self._local_model.model.config.id2label.values())


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
        if not self.is_local():
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
        if not self.is_local():
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
                pipes: Optional[List[str]] = None,
                with_vectors: bool = False) -> List["spacy.tokens.Doc"]:  # noqa: F821 -- type annotation only, avoid importing spaCy here
        """Perform NLP analysis on `text`.

        `pipes`: If provided, enable only the listed pipes. Which ones exist depend on the loaded spaCy model.
                 If not provided, use the model's default pipes.

        `with_vectors`: If `True`, ensure `token.vector` is available on the returned docs — a wire-format
                        concern only. In remote mode, requests `doc.tensor` from the server (bigger payload).
                        In local mode, ignored: the in-process spaCy pipeline produces a fully-featured doc
                        regardless, so vectors are always accessible. The flag gives callers identical
                        feature-parity semantics across modes: "if I ask for vectors, I get vectors."

        Returns a `list` of spaCy documents (even if just one `text`).

        In remote mode, the pipeline at the client side is blank, but the tokens have all the usual data.
        """
        if not self.is_local():
            docs = api.natlang_analyze(text,
                                       pipes,
                                       with_vectors=with_vectors)
        else:
            docs = nlptools.spacy_analyze(self._local_model,
                                          text,
                                          pipes)
        return docs


class Translator(MaybeRemoteService):
    def __init__(self,
                 allow_local: bool,
                 source_lang: str,
                 target_lang: str,
                 model_name: Optional[str] = None,
                 device_string: Optional[str] = None,
                 dtype: Optional[Union[str, torch.dtype]] = None,
                 spacy_model_name: Optional[str] = None):
        """Machine translation between natural languages (Helsinki-NLP / OPUS-MT family).

        `allow_local`: See `MaybeRemoteService`.

        `source_lang`, `target_lang`: Language codes, e.g. `"en"` → `"fi"`. In remote
                                      mode the server must have a translator loaded for
                                      this pair (see `raven.server.config.translation_models`);
                                      in local mode, the pair just identifies the model.

        `model_name`: Required if `allow_local=True`. HuggingFace model name, e.g.
                      `"Helsinki-NLP/opus-mt-tc-big-en-fi"`.

        `device_string`: Required if `allow_local=True`. E.g. `"cpu"`, `"cuda:0"`.

        `dtype`: Required if `allow_local=True`. E.g. `torch.float32`.

        `spacy_model_name`: Required if `allow_local=True`. spaCy pipeline for
                            sentence-boundary detection during chunked translation
                            (long inputs). Usually `"en_core_web_sm"`. Cached via
                            `nlptools.load_spacy_pipeline`, so sharing the same model
                            with a `MaybeRemote.NLP` instance costs nothing extra.
        """
        super().__init__(allow_local)
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.model_name = model_name
        self.device_string = device_string
        self.dtype = dtype
        self.spacy_model_name = spacy_model_name

        if "translate" in self.server_modules:
            logger.info(f"Translator.__init__: Using `translate` module on Raven-server at '{client_config.raven_server_url}'.")
        else:
            if self.server_available:
                logger.info(f"Translator.__init__: No `translate` module loaded on Raven-server at '{client_config.raven_server_url}', loading translation model locally.")
            self._local_model = nlptools.load_translator(model_name,
                                                         device_string,
                                                         dtype,
                                                         source_lang=source_lang,
                                                         target_lang=target_lang)
            # Sentence splitter for chunked translation. Reuses any cached spaCy pipeline.
            self._local_nlp = nlptools.load_spacy_pipeline(spacy_model_name, device_string)

    def translate(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
        """Translate `text` from `source_lang` to `target_lang`.

        Returns `str` for a single input, `list[str]` for a list of inputs.
        """
        if not self.is_local():
            return api.translate_translate(text,
                                           source_lang=self.source_lang,
                                           target_lang=self.target_lang)
        return nlptools.translate(self._local_model, self._local_nlp, text)


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

        if "stt" in self.server_modules:
            logger.info(f"STT.__init__: Using `stt` module on Raven-server at '{client_config.raven_server_url}'.")
            self.sample_rate = api.stt_info()["sample_rate"]
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
        if not self.is_local():
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
                 model_name: Optional[str] = None,
                 device_string: Optional[str] = None,
                 lang_code: str = "a"):
        """Text-to-speech (Kokoro).

        `allow_local`: See `MaybeRemoteService`. Kokoro-82M weighs ~360 MB;
                       apps that only occasionally need TTS should prefer
                       `allow_local=False`.

        `model_name`: Required if `allow_local=True`. HuggingFace repo
                      identifier, e.g. `"hexgrad/Kokoro-82M"`.

        `device_string`: Required if `allow_local=True`. E.g. `"cpu"`, `"cuda:0"`.

        `lang_code`: phonemizer language code. Word-level metadata currently
                     supports English only (`"a"` or `"b"`). See
                     `raven.common.audio.speech.tts.load_tts_pipeline` for the
                     full list.
        """
        super().__init__(allow_local)
        self.model_name = model_name
        self.device_string = device_string
        self.lang_code = lang_code

        if "tts" in self.server_modules:
            logger.info(f"TTS.__init__: Using `tts` module on Raven-server at '{client_config.raven_server_url}'.")
            self.sample_rate = api.tts_info()["sample_rate"]
        else:
            if self.server_available:
                logger.info(f"TTS.__init__: No `tts` module loaded on Raven-server at '{client_config.raven_server_url}', loading Kokoro pipeline locally.")
            self._local_model = speech_tts.load_tts_pipeline(model_name=model_name,
                                                             device_string=device_string,
                                                             lang_code=lang_code)
            self.sample_rate = self._local_model.sample_rate

    def list_voices(self) -> List[str]:
        """List installed voices. Remote uses the server's /api/tts/list_voices; local scans the modelsdir."""
        if not self.is_local():
            return api.tts_list_voices()
        return speech_tts.get_voices(self._local_model)

    def synthesize(self,
                   voice: str,
                   text: str,
                   speed: float = 1.0,
                   get_metadata: bool = True,
                   format: Optional[str] = None) -> Union[speech_datatypes.TTSResult, speech_datatypes.EncodedTTSResult]:
        """Synthesize `text`, with audio in the shape the caller asks for.

        `format=None` (default): return a `TTSResult` with raw float32 mono audio
        in [-1, 1] at `self.sample_rate` (24 kHz). Use for analysis paths
        (embedding, STT round-trip, lipsync driver).

        `format="mp3"` / `"flac"` / …: return an `EncodedTTSResult` with the audio
        encoded in the requested format. Use for playback paths that want encoded
        bytes ready to hand to an audio player (pygame mixer etc.).

        `word_metadata` is populated identically in both shapes when
        `get_metadata=True` — a list of `WordTiming` post-processed for lipsync
        via `speech_tts.finalize_metadata`.

        Pure 2×2 dispatch; caching lives in the bottom layers. Local mode
        calls `speech_tts.prepare_cached` or `speech_tts.prepare_encoded_cached`
        in the common layer; remote mode calls `api.tts_prepare_decoded_cached`
        or `api.tts_prepare_cached`. A remote `format=None` call therefore still
        pays the encoded-wire cost once (default FLAC) but caches the decoded
        `TTSResult`, not just the encoded wire bytes — so repeat calls are free
        on both sides.
        """
        if not self.is_local():
            if format is None:
                return api.tts_prepare_decoded_cached(text=text, voice=voice, speed=speed, get_metadata=get_metadata)
            return api.tts_prepare_cached(text=text, voice=voice, speed=speed, get_metadata=get_metadata, format=format)
        if format is None:
            return speech_tts.prepare_cached(self._local_model, voice=voice, text=text, speed=speed, get_metadata=get_metadata)
        return speech_tts.prepare_encoded_cached(self._local_model, voice=voice, text=text, speed=speed, get_metadata=get_metadata, format=format)

    def speak(self,
              voice: str,
              text: str,
              speed: float = 1.0,
              on_audio_ready: Optional[Callable] = None,
              on_start: Optional[Callable] = None,
              on_stop: Optional[Callable] = None,
              prep: Optional[Union[speech_datatypes.TTSResult, speech_datatypes.EncodedTTSResult]] = None) -> None:
        """Synthesize and speak `text`, fire-and-forget. Non-lipsynced.

        Signature mirrors `raven.client.api.tts_speak`, with one generalization:
        `prep` accepts either `TTSResult` (local-mode native shape) or `EncodedTTSResult`
        (remote-mode native shape, or an already-encoded local result). The wrapper
        encodes to FLAC internally when needed, before handing bytes to the audio player.

        In remote mode, delegates to `raven.client.api.tts_speak` (which handles its
        own task_manager submission and playback). In local mode, synthesizes via
        `self.synthesize(format="flac")` and submits
        `raven.common.audio.speech.playback.play_encoded` to the client task_manager
        for fire-and-forget playback.

        See `raven.client.api.tts_speak` for callback semantics.
        """
        if not self.is_local():
            encoded = _to_encoded(prep)
            return api.tts_speak(text=text, voice=voice, speed=speed,
                                 on_audio_ready=on_audio_ready, on_start=on_start, on_stop=on_stop,
                                 prep=encoded)
        # Local mode: synthesize (if needed), then submit playback to the task manager.
        def _speak(task_env):
            if prep is None:
                final = speech_tts.prepare_encoded_cached(self._local_model, voice=voice, text=text, speed=speed, get_metadata=False, format="flac")
            else:
                final = _to_encoded(prep)
            if not final.audio_bytes:
                logger.info(f"TTS.speak: instance {task_env.task_name}: no audio produced. Cancelled.")
                return
            speech_playback.play_encoded(final.audio_bytes,
                                         player=client_util.api_config.audio_player,
                                         on_audio_ready=on_audio_ready,
                                         on_start=on_start,
                                         on_stop=on_stop)
        client_util.api_config.task_manager.submit(_speak, _envcls())

    def speak_lipsynced(self,
                        instance_id: str,
                        voice: str,
                        text: str,
                        speed: float = 1.0,
                        video_offset: float = 0.0,
                        on_audio_ready: Optional[Callable] = None,
                        on_start: Optional[Callable] = None,
                        on_stop: Optional[Callable] = None,
                        prep: Optional[Union[speech_datatypes.TTSResult, speech_datatypes.EncodedTTSResult]] = None) -> None:
        """Synthesize and speak `text` with avatar lipsync, fire-and-forget.

        Signature mirrors `raven.client.api.tts_speak_lipsynced`. As with `speak`,
        `prep` accepts either TTS result shape; encoding to FLAC happens internally
        as needed.

        In remote mode, delegates to `raven.client.api.tts_speak_lipsynced`. In local
        mode, synthesizes via `self.synthesize(format="flac", get_metadata=True)` and
        submits `raven.client.tts.play_encoded_with_avatar_lipsync` to the client
        task_manager — same Raven-avatar driver as the remote path, just with the
        synthesis step running in-process.

        Note on the hybrid local-TTS + remote-avatar path: the lipsync driver calls
        `api.avatar_modify_overrides` per phoneme tick, which is a round-trip to the
        server. Until the client-local avatar animator lands (see `TODO_DEFERRED.md`),
        the avatar remains server-side regardless of where TTS runs. LAN latency is
        usually fine for this; if it isn't, fall back to remote mode.

        See `raven.client.api.tts_speak_lipsynced` for callback semantics.
        """
        if not self.is_local():
            encoded = _to_encoded(prep)
            return api.tts_speak_lipsynced(instance_id=instance_id,
                                           text=text, voice=voice, speed=speed, video_offset=video_offset,
                                           on_audio_ready=on_audio_ready, on_start=on_start, on_stop=on_stop,
                                           prep=encoded)
        # Local mode: synthesize with metadata, then submit lipsynced playback.
        def _speak(task_env):
            if prep is None:
                final = speech_tts.prepare_encoded_cached(self._local_model, voice=voice, text=text, speed=speed, get_metadata=True, format="flac")
            elif isinstance(prep, speech_datatypes.TTSResult):
                final = speech_tts.encode(prep, format="flac")
            else:
                final = prep
            if not final.audio_bytes:
                logger.info(f"TTS.speak_lipsynced: instance {task_env.task_name}: no audio produced. Cancelled.")
                return
            client_tts.play_encoded_with_avatar_lipsync(final.audio_bytes,
                                                        timestamps=final.word_metadata,
                                                        instance_id=instance_id,
                                                        video_offset=video_offset,
                                                        on_audio_ready=on_audio_ready,
                                                        on_start=on_start,
                                                        on_stop=on_stop)
        client_util.api_config.task_manager.submit(_speak, _envcls())

    def stop(self) -> None:
        """Stop TTS playback immediately. No effect if nothing is currently playing.

        Mode-independent: the client's audio player is always local (audio hardware
        is on the user's machine regardless of synthesis mode). Provided as a method
        on `TTS` for the symmetry with `.speak` / `.speak_lipsynced`.
        """
        client_util.api_config.audio_player.stop()

    def is_speaking(self) -> bool:
        """Return whether the TTS is currently producing audio.

        Mode-independent for the same reason as `stop`: queries the client's audio
        player, which is the one doing the playing in both local and remote modes.
        """
        return client_util.api_config.audio_player.is_playing()


def _to_encoded(prep: Optional[Union[speech_datatypes.TTSResult, speech_datatypes.EncodedTTSResult]],
                format: str = "flac") -> Optional[speech_datatypes.EncodedTTSResult]:
    """Normalize a `prep` argument to `EncodedTTSResult` (or pass `None` through).

    Accepts either `TTSResult` (encoded on the fly) or `EncodedTTSResult` (returned as-is).
    Used by `TTS.speak` / `TTS.speak_lipsynced` — callers from either mode may have
    whichever shape on hand; we accept both and encode where needed.
    """
    if prep is None:
        return None
    if isinstance(prep, speech_datatypes.EncodedTTSResult):
        return prep
    return speech_tts.encode(prep, format=format)


class Postprocessor(MaybeRemoteService):
    def __init__(self,
                 allow_local: bool,
                 device_string: Optional[str] = None,
                 dtype: Optional[Union[str, torch.dtype]] = None):
        """Image postprocessor — applies a filter chain (see `raven.common.video.postprocessor`).

        Unlike the other `MaybeRemote` services, this one wraps a server module (`imagefx`)
        that is effectively stateless across requests: the filter chain travels with each call.
        The local side is stateful (a `Postprocessor` instance holds torch tensors on a device),
        so in local mode we reuse one `_LocalPostprocessor`, swapping its `.chain` per call
        under a lock (same pattern as the server module uses internally).

        `allow_local`: See `MaybeRemoteService`. The postprocessor is pure torch — no heavy ML
                       models are downloaded — so `allow_local=True` is usually safe.

        `device_string`: Required if `allow_local=True`. E.g. `"cpu"`, `"cuda:0"`.

        `dtype`: Required if `allow_local=True`. E.g. `torch.float32`, `torch.float16`.
        """
        super().__init__(allow_local)
        self.device_string = device_string
        self.dtype = dtype

        if "imagefx" in self.server_modules:
            logger.info(f"Postprocessor.__init__: Using `imagefx` module on Raven-server at '{client_config.raven_server_url}'.")
        else:
            if self.server_available:
                logger.info(f"Postprocessor.__init__: No `imagefx` module loaded on Raven-server at '{client_config.raven_server_url}', instantiating local postprocessor.")
            # Start with an empty chain; callers supply the real chain per `process` call.
            self._local_model = _LocalPostprocessor(device_string, dtype, chain=[])
            # Guards the `.chain` swap + `render_into` pair across concurrent callers.
            self._local_lock = threading.Lock()

    def process(self,
                image: np.ndarray,
                filters: List[Dict[str, Any]]) -> np.ndarray:
        """Apply the postprocessor `filters` chain to `image`.

        `image`: float32 `np.ndarray` in [0, 1], shape `(h, w, c)` with 3 or 4 channels.
                 Three-channel input is normalized to RGBA internally.

        `filters`: Filter chain, formatted as in `raven.server.config.postprocessor_defaults`.

        Returns the processed image as float32 `np.ndarray` in [0, 1], shape `(h, w, 4)`.
        """
        if not self.is_local():
            return api.imagefx_process_array(image, filters=filters)

        # Local path: HWC float → CHW tensor → postprocess in-place → HWC float.
        from ..common.image import utils as imageutils  # deferred: heavy to import when not needed
        image = imageutils.ensure_rgba(image)
        tensor = torch.from_numpy(image).permute(2, 0, 1).to(dtype=self._local_model.dtype,
                                                              device=self._local_model.device)
        with self._local_lock:
            self._local_model.chain = filters
            self._local_model.render_into(tensor)
        return tensor.clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy().astype(np.float32)


class Upscaler(MaybeRemoteService):
    def __init__(self,
                 allow_local: bool,
                 device_string: Optional[str] = None,
                 dtype: Optional[Union[str, torch.dtype]] = None):
        """Image upscaler (Anime4K-PyTorch, plus bilinear / bicubic fast paths).

        Mirrors the `imagefx` upscale endpoint: target resolution, preset, and quality
        travel with each call. In local mode, `_LocalUpscaler` instances are cached by
        `(width, height, preset, quality)` since model choice depends on preset/quality;
        repeat calls with the same config reuse the cached pipeline.

        `allow_local`: See `MaybeRemoteService`. Anime4K is small and loads fast; the
                       `bilinear` / `bicubic` quality settings skip Anime4K entirely.

        `device_string`, `dtype`: Required if `allow_local=True`. Same semantics as `Postprocessor`.
        """
        super().__init__(allow_local)
        self.device_string = device_string
        self.dtype = dtype

        if "imagefx" in self.server_modules:
            logger.info(f"Upscaler.__init__: Using `imagefx` module on Raven-server at '{client_config.raven_server_url}'.")
        else:
            if self.server_available:
                logger.info(f"Upscaler.__init__: No `imagefx` module loaded on Raven-server at '{client_config.raven_server_url}', local upscaler instances will be created on demand.")
            # `_local_model` holds the cache of `_LocalUpscaler` instances, keyed by
            # config tuple `(width, height, preset, quality)`. Being a (non-None) dict
            # also makes `is_local()` return True per the base class's convention.
            # Instances are constructed lazily on the first `.upscale` call with a
            # given config; different configs trigger new constructions (each of which
            # loads the underlying Anime4K models, hence the cache).
            self._local_model: Dict[tuple, _LocalUpscaler] = {}
            self._local_lock = threading.Lock()

    def upscale(self,
                image: np.ndarray,
                upscaled_width: int = 1920,
                upscaled_height: int = 1080,
                preset: str = "C",
                quality: str = "high") -> np.ndarray:
        """Upscale `image` to `(upscaled_width, upscaled_height)`.

        `image`: float32 `np.ndarray` in [0, 1], shape `(h, w, c)` with 3 or 4 channels.
                 Three-channel input is normalized to RGBA internally.

        `preset`: one of `"A"`, `"B"`, `"C"` (Anime4K-style pipeline selection).

        `quality`: one of `"low"`, `"high"` (Anime4K model sizes), or `"bilinear"` /
                   `"bicubic"` (fast bypass — no Anime4K).

        Returns the upscaled image as float32 `np.ndarray` in [0, 1], shape `(upscaled_height, upscaled_width, 4)`.
        """
        if not self.is_local():
            return api.imagefx_upscale_array(image,
                                             upscaled_width=upscaled_width,
                                             upscaled_height=upscaled_height,
                                             preset=preset,
                                             quality=quality)

        # Local path: ensure RGBA, HWC float → CHW tensor → upscale → HWC float.
        from ..common.image import utils as imageutils  # deferred: heavy to import when not needed
        image = imageutils.ensure_rgba(image)
        tensor = torch.from_numpy(image).permute(2, 0, 1).to(dtype=self.dtype,
                                                              device=self.device_string)
        # Fetch-or-construct the appropriate `_LocalUpscaler` for this config.
        # Serialized so concurrent callers with the same novel config don't race
        # to construct parallel copies.
        key = (upscaled_width, upscaled_height, preset, quality)
        with self._local_lock:
            if key not in self._local_model:
                logger.info(f"Upscaler.upscale: constructing new _LocalUpscaler for {key}")
                self._local_model[key] = _LocalUpscaler(device=self.device_string,
                                                         dtype=self.dtype,
                                                         upscaled_width=upscaled_width,
                                                         upscaled_height=upscaled_height,
                                                         preset=preset,
                                                         quality=quality)
            upscaler = self._local_model[key]
        upscaled = upscaler.upscale(tensor)
        return upscaled.clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy().astype(np.float32)
