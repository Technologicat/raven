"""What each model-holding server module reports about the models it loaded: `get_info`, across the modules.

Each test stands in for a loaded model by setting the module's state directly, so no model is loaded. The
modules import the ML stack all the same, hence the `ml` marker and a guard per module.
"""

import types

import pytest

pytestmark = pytest.mark.ml


def module(name):
    return pytest.importorskip(f"raven.server.modules.{name}", reason="the server's modules need the ML stack")


class TestBeforeLoading:
    @pytest.mark.parametrize("name", ["classify", "sanitize", "translate", "natlang", "avatar"])
    def test_a_module_that_loaded_nothing_says_so_rather_than_answering(self, name):
        with pytest.raises(RuntimeError, match="not initialized"):
            module(name).get_info()


class TestAfterLoading:
    def test_classify_names_its_model(self, monkeypatch):
        classify = module("classify")
        monkeypatch.setattr(classify, "classifier", object())
        monkeypatch.setattr(classify, "_model_name", "some/classifier")
        assert classify.get_info() == {"model": "some/classifier"}

    def test_sanitize_names_its_model(self, monkeypatch):
        sanitize = module("sanitize")
        monkeypatch.setattr(sanitize, "dehyphenator", object())
        monkeypatch.setattr(sanitize, "_model_name", "multi")
        assert sanitize.get_info() == {"model": "multi"}

    def test_translate_names_a_model_per_pair_and_its_sentence_splitter(self, monkeypatch):
        translate = module("translate")
        config = types.SimpleNamespace(translation_models={"fi": {"en": "en-fi-model", "sv": "sv-fi-model"}})
        pipeline = types.SimpleNamespace(meta={"lang": "en", "name": "core_web_sm", "version": "3.8.0"})
        monkeypatch.setattr(translate, "server_config", config)
        monkeypatch.setattr(translate, "nlp_pipe", pipeline)
        assert translate.get_info() == {"models": {"fi": {"en": "en-fi-model", "sv": "sv-fi-model"}},
                                        "spacy": {"model": "en_core_web_sm", "version": "3.8.0"}}

    def test_natlang_names_its_pipeline_and_version(self, monkeypatch):
        natlang = module("natlang")
        monkeypatch.setattr(natlang, "nlp_pipeline",
                            types.SimpleNamespace(meta={"lang": "en", "name": "core_web_sm", "version": "3.8.0"}))
        assert natlang.get_info() == {"model": "en_core_web_sm", "version": "3.8.0"}

    def test_avatar_names_its_variant_and_where_it_came_from(self, monkeypatch):
        avatar = module("avatar")
        monkeypatch.setattr(avatar, "_poser", object())
        monkeypatch.setattr(avatar, "_model", "separable_half")
        monkeypatch.setattr(avatar, "_server_config", types.SimpleNamespace(talkinghead_models="some/repo"))
        assert avatar.get_info() == {"model": "separable_half", "repository": "some/repo"}
