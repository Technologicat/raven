"""Tests for `raven.client.mayberemote`'s choice of mode: which of local and remote a service may use.

Needs no server and loads no model: the base class decides the mode, and asking the server is the one thing
it does that these tests watch.
"""

import pytest

from raven.client import api, mayberemote


@pytest.fixture
def probe(monkeypatch):
    """Stands in for asking whether Raven-server answers, counting the times it was asked."""
    asked = []

    def raven_server_available():
        asked.append(True)
        return True
    monkeypatch.setattr(api, "raven_server_available", raven_server_available)
    monkeypatch.setattr(api, "modules", lambda: ["natlang"])
    return asked


class TestModes:
    def test_without_remote_the_server_is_never_asked(self, probe):
        service = mayberemote.MaybeRemoteService(allow_local=True, allow_remote=False)
        assert probe == [] and service.server_modules == [] and not service.server_available

    def test_by_default_the_server_is_asked(self, probe):
        service = mayberemote.MaybeRemoteService(allow_local=True)
        assert probe == [True] and service.server_modules == ["natlang"], \
            "the control: with remote allowed, the probe above is the one the service uses"

    def test_neither_mode_allowed_is_refused(self, probe):
        with pytest.raises(ValueError, match="cannot both be `False`"):
            mayberemote.MaybeRemoteService(allow_local=False, allow_remote=False)
        assert probe == []


INFO = {"classify": {"model": "some/classifier"},
        "sanitize": {"model": "multi"},
        "translate": {"models": {"fi": {"en": "en-fi-model"}}, "spacy": {"model": "en_core_web_sm", "version": "3.8.0"}},
        "natlang": {"model": "en_core_web_sm", "version": "3.8.0"},
        "embeddings": {"models": {"default": {"model": "big/embedder", "dimension": 1024},
                                  "qa": {"model": "qa/embedder", "dimension": 768}}},
        "stt": {"model": "openai/whisper-base", "sample_rate": 16000},
        "tts": {"model": "hexgrad/Kokoro-82M", "sample_rate": 24000}}


@pytest.fixture
def server(monkeypatch):
    """A server running every module, answering each `<module>_info` from `INFO`. No model loads in remote mode."""
    monkeypatch.setattr(api, "raven_server_available", lambda: True)
    monkeypatch.setattr(api, "modules", lambda: list(INFO))
    for module, info in INFO.items():
        monkeypatch.setattr(api, f"{module}_info", lambda info=info: info)
    monkeypatch.setattr(api, "embeddings_compute", lambda text, model: [[0.0]])


class TestModelInfoInRemoteMode:
    """Each service says which of the server's models it uses, whatever it asked for."""

    def test_single_model_services_report_the_servers_model(self, server):
        assert mayberemote.Classifier(allow_local=False, model_name="asked/for").model_info == {"model": "some/classifier"}
        assert mayberemote.Dehyphenator(allow_local=False, model_name="asked", device_string=None).model_info == \
               {"model": "multi"}
        assert mayberemote.NLP(allow_local=False, model_name="asked", device_string=None).model_info == INFO["natlang"]
        assert mayberemote.STT(allow_local=False, model_name="asked").model_info == {"model": "openai/whisper-base"}
        assert mayberemote.TTS(allow_local=False, model_name="asked").model_info == {"model": "hexgrad/Kokoro-82M"}

    def test_a_translator_reports_its_own_pair_and_its_sentence_splitter(self, server):
        translator = mayberemote.Translator(allow_local=False, source_lang="en", target_lang="fi", model_name=None,
                                            device_string=None, dtype=None, spacy_model_name=None)
        assert translator.model_info == {"model": "en-fi-model", "spacy": INFO["translate"]["spacy"]}

    @pytest.mark.parametrize("asked, expected", [("qa", "qa/embedder"), ("big/embedder", "big/embedder")])
    def test_an_embedder_finds_its_model_by_role_or_by_name(self, server, asked, expected):
        embedder = mayberemote.Embedder(allow_local=False, model_name=asked, device_string=None, dtype=None)
        assert embedder.model_info["model"] == expected
