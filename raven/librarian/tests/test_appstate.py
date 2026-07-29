"""Unit tests for raven.librarian.appstate (load / save)."""

import json
import pathlib

import pytest

from raven.librarian import appstate, chattree, chatutil


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load(tmp_path, llm_settings, *,
          datastore_name="data.json",
          state_name="state.json",
          state_contents=None,
          autosave=False,
          persist=True):
    """Drive `appstate.load` on files in `tmp_path`.

    Writes `state_contents` (if given) as JSON to the state file before loading.
    Returns `(datastore, state, datastore_path, state_path)`.

    With `persist=True` (the default), the datastore and state are explicitly saved
    to disk after loading — simulating the end-of-run persistence that `autosave=True`
    would do via `atexit`. This makes "load → app ran → exited → load again" tests
    straightforward without actually registering `atexit` hooks.
    """
    datastore_path = tmp_path / datastore_name
    state_path = tmp_path / state_name
    if state_contents is not None:
        state_path.write_text(json.dumps(state_contents), encoding="utf-8")
    datastore, state = appstate.load(llm_settings=llm_settings,
                                     datastore_file=datastore_path,
                                     state_file=state_path,
                                     autosave=autosave)
    if persist:
        datastore.save()
        appstate.save(state_path, state)
    return datastore, state, datastore_path, state_path


# ---------------------------------------------------------------------------
# Legacy (string-content) datastore migration on load
# ---------------------------------------------------------------------------

class TestLegacyContentMigration:
    """Regression: `load` runs the content-parts migration BEFORE `_refresh_greeting`, which compares stored
    greeting content via `content_to_text` (now assumes a parts list). A legacy datastore stores `content` as a
    bare string; if the migration ran after the greeting refresh, that read crashed on the un-migrated string."""

    def test_legacy_string_content_loads_and_migrates(self, tmp_path, llm_settings):
        datastore_path = tmp_path / "data.json"
        seed = chattree.PersistentForest(datastore_path, autosave=False)
        system_id = seed.create_node(
            payload={"message": {"role": "system", "content": "old system prompt", "tool_calls": []},
                     "general_metadata": {"persona": None}},
            parent_id=None)
        seed.create_node(  # greeting matching the configured one (persona-prefixed string, the legacy form)
            payload={"message": {"role": "assistant", "content": "Aria: How can I help you today?", "tool_calls": []},
                     "general_metadata": {"persona": "Aria"}},
            parent_id=system_id)
        seed.save()

        # Must not raise — the ordering bug crashed here (TypeError from content_to_text iterating a str).
        datastore, state, _, _ = _load(tmp_path, llm_settings)

        # Every message's content, across all revisions of all nodes, is now a content-parts list.
        for node in datastore.nodes.values():
            for payload in node["data"].values():
                assert isinstance(payload["message"]["content"], list)

        # The seeded greeting matched the configured one, so it was reused (HEAD points to it), not duplicated.
        head_content = chatutil.content_to_text(datastore.get_payload(state["new_chat_HEAD"])["message"]["content"])
        assert "How can I help you today?" in head_content


# ---------------------------------------------------------------------------
# Fresh start: empty datastore, missing state file
# ---------------------------------------------------------------------------

class TestLoadEmpty:
    def test_empty_datastore_triggers_factory_reset(self, tmp_path, llm_settings):
        datastore, state, _, _ = _load(tmp_path, llm_settings)

        # Factory reset created exactly two nodes: system prompt (root) + greeting (child).
        assert len(datastore.nodes) == 2
        roots = datastore.get_all_root_nodes()
        assert len(roots) == 1
        root = roots[0]
        assert datastore.get_children(root) != []

    def test_state_dict_has_all_required_keys(self, tmp_path, llm_settings):
        _, state, _, _ = _load(tmp_path, llm_settings)
        for key in ("HEAD", "new_chat_HEAD", "system_prompt_node_id",
                    "tools_enabled", "docs_enabled", "speculate_enabled",
                    "avatar_speech_enabled", "avatar_subtitles_enabled"):
            assert key in state, f"state missing key {key!r}"

    def test_default_flag_values(self, tmp_path, llm_settings):
        _, state, _, _ = _load(tmp_path, llm_settings)
        for key, default in appstate._DEFAULT_FLAGS.items():
            assert state[key] is default, f"flag {key!r} did not get its default"

    def test_head_points_to_greeting_node(self, tmp_path, llm_settings):
        datastore, state, _, _ = _load(tmp_path, llm_settings)
        assert state["HEAD"] == state["new_chat_HEAD"]
        greeting_payload = datastore.get_payload(state["HEAD"])
        assert greeting_payload["message"]["role"] == "assistant"
        # The greeting content is prefixed by the character name (`Aria: ...`).
        assert llm_settings.greeting in chatutil.content_to_text(greeting_payload["message"]["content"])

    def test_system_prompt_node_id_points_to_root(self, tmp_path, llm_settings):
        datastore, state, _, _ = _load(tmp_path, llm_settings)
        roots = datastore.get_all_root_nodes()
        assert state["system_prompt_node_id"] == roots[0]


# ---------------------------------------------------------------------------
# Existing datastore, state file variations
# ---------------------------------------------------------------------------

class TestLoadWithState:
    def test_preserved_head_is_respected(self, tmp_path, llm_settings):
        # First, create the datastore + state.
        _, state1, datastore_path, state_path = _load(tmp_path, llm_settings)
        original_head = state1["HEAD"]
        # Write the state file explicitly (autosave=False means we must).
        appstate.save(state_path, state1)

        # Second load should preserve the stored HEAD.
        _, state2, _, _ = _load(tmp_path, llm_settings)
        assert state2["HEAD"] == original_head

    def test_dangling_head_resets_to_new_chat_head(self, tmp_path, llm_settings):
        # Bootstrap a real datastore.
        _, state, _, state_path = _load(tmp_path, llm_settings)
        # Corrupt: point HEAD at a node that doesn't exist.
        state["HEAD"] = "gensym#forest-node:does-not-exist"
        appstate.save(state_path, state)

        _, state2, _, _ = _load(tmp_path, llm_settings)
        assert state2["HEAD"] == state2["new_chat_HEAD"]

    def test_missing_head_key_set_to_new_chat_head(self, tmp_path, llm_settings):
        # Bootstrap a valid datastore on disk.
        _load(tmp_path, llm_settings)
        # Overwrite the state file with one missing the HEAD key (everything else present).
        state_path = tmp_path / "state.json"
        bootstrap_state = {"new_chat_HEAD": "placeholder",
                           "tools_enabled": True,
                           "docs_enabled": True,
                           "speculate_enabled": False,
                           "avatar_speech_enabled": True,
                           "avatar_subtitles_enabled": True}
        state_path.write_text(json.dumps(bootstrap_state), encoding="utf-8")
        # Now load: new_chat_HEAD gets recomputed (always is), HEAD gets set to it.
        _, state2, _, _ = _load(tmp_path, llm_settings)
        assert "HEAD" in state2
        assert state2["HEAD"] == state2["new_chat_HEAD"]

    @pytest.mark.parametrize("missing_key,default", list(appstate._DEFAULT_FLAGS.items()))
    def test_missing_individual_flag_gets_default(self, tmp_path, llm_settings, missing_key, default):
        # Bootstrap a valid state file.
        _, state, _, state_path = _load(tmp_path, llm_settings)
        appstate.save(state_path, state)
        # Remove the flag from the on-disk file.
        stored = json.loads(state_path.read_text(encoding="utf-8"))
        del stored[missing_key]
        state_path.write_text(json.dumps(stored), encoding="utf-8")
        # Reload: the flag should be restored to its default.
        _, state2, _, _ = _load(tmp_path, llm_settings)
        assert state2[missing_key] is default


# ---------------------------------------------------------------------------
# System prompt refresh
# ---------------------------------------------------------------------------

class TestSystemPromptRefresh:
    def test_system_prompt_content_matches_current_settings(self, tmp_path, llm_settings):
        datastore, state, _, _ = _load(tmp_path, llm_settings)
        payload = datastore.get_payload(state["system_prompt_node_id"])
        content = chatutil.content_to_text(payload["message"]["content"])
        # The system-prompt payload weaves system_prompt + character_card together.
        assert llm_settings.system_prompt in content
        assert llm_settings.character_card in content

    def test_refresh_leaves_exactly_one_revision(self, tmp_path, llm_settings):
        # The refresh adds a new revision and deletes the old one, so the node should
        # end up with exactly one revision even after bootstrap + refresh.
        datastore, state, _, _ = _load(tmp_path, llm_settings)
        system_prompt_node_id = state["system_prompt_node_id"]
        revisions = datastore.get_revisions(system_prompt_node_id)
        assert len(revisions) == 1

    def test_refresh_picks_up_updated_prompt(self, tmp_path, llm_settings):
        # First load with original settings.
        _load(tmp_path, llm_settings)
        # Mutate llm_settings and reload.
        llm_settings.system_prompt = "You are an updated assistant."
        datastore, state, _, _ = _load(tmp_path, llm_settings)
        payload = datastore.get_payload(state["system_prompt_node_id"])
        assert "updated assistant" in chatutil.content_to_text(payload["message"]["content"])


# ---------------------------------------------------------------------------
# Greeting refresh
# ---------------------------------------------------------------------------

class TestGreetingRefresh:
    def test_existing_matching_greeting_is_reused(self, tmp_path, llm_settings):
        _, state1, _, _ = _load(tmp_path, llm_settings)
        first_greeting = state1["new_chat_HEAD"]
        # Second load with identical settings should reuse the same greeting node.
        _, state2, _, _ = _load(tmp_path, llm_settings)
        assert state2["new_chat_HEAD"] == first_greeting

    def test_changed_greeting_creates_new_node(self, tmp_path, llm_settings):
        _, state1, _, _ = _load(tmp_path, llm_settings)
        first_greeting = state1["new_chat_HEAD"]
        # Change the greeting text and reload: a new greeting node should be created.
        llm_settings.greeting = "Hello, I'm here."
        datastore, state2, _, _ = _load(tmp_path, llm_settings)
        assert state2["new_chat_HEAD"] != first_greeting
        # Both greeting nodes should exist under the system prompt node.
        system_prompt_node_id = state2["system_prompt_node_id"]
        children = datastore.get_children(system_prompt_node_id)
        assert first_greeting in children
        assert state2["new_chat_HEAD"] in children


# ---------------------------------------------------------------------------
# save()
# ---------------------------------------------------------------------------

class TestSave:
    def test_save_then_load_roundtrips(self, tmp_path, llm_settings):
        _, state, _, state_path = _load(tmp_path, llm_settings)
        state["tools_enabled"] = False
        state["docs_enabled"] = False
        appstate.save(state_path, state)

        stored = json.loads(state_path.read_text(encoding="utf-8"))
        assert stored["tools_enabled"] is False
        assert stored["docs_enabled"] is False
        assert stored["HEAD"] == state["HEAD"]

    def test_save_missing_required_key_raises(self, tmp_path, llm_settings):
        _, state, _, state_path = _load(tmp_path, llm_settings)
        del state["tools_enabled"]
        with pytest.raises(KeyError):
            appstate.save(state_path, state)


# ---------------------------------------------------------------------------
# autosave behaviour
# ---------------------------------------------------------------------------

class TestAutosave:
    # Note: `atexit.register` is a module-global, so patching either
    # `raven.librarian.chattree.atexit.register` or `raven.librarian.appstate.atexit.register`
    # replaces it for both call sites (same underlying module object). A single
    # `monkeypatch.setattr("atexit.register", ...)` is equivalent and clearer.

    def test_autosave_false_skips_atexit_registration(self, monkeypatch, tmp_path, llm_settings):
        """With `autosave=False`, neither the PersistentForest save nor the appstate save
        should be registered with `atexit`."""
        registered = []
        monkeypatch.setattr("atexit.register",
                            lambda fn, *a, **kw: registered.append(fn))

        _load(tmp_path, llm_settings, autosave=False)
        assert registered == []

    def test_autosave_true_registers_two_hooks(self, monkeypatch, tmp_path, llm_settings):
        """With `autosave=True`, `PersistentForest.__init__` and `appstate.load` each
        register one `atexit` hook (two total)."""
        registered = []
        monkeypatch.setattr("atexit.register",
                            lambda fn, *a, **kw: registered.append(fn))

        _load(tmp_path, llm_settings, autosave=True)
        assert len(registered) == 2

    def test_persistentforest_autosave_property_is_readonly(self, tmp_path):
        """The `autosave` property cannot be reassigned on an existing instance —
        changing it after construction would be misleading (the atexit hook is
        already registered or not)."""
        pf = chattree.PersistentForest(tmp_path / "ro.json", autosave=False)
        assert pf.autosave is False
        with pytest.raises(AttributeError):
            pf.autosave = True


# ---------------------------------------------------------------------------
# Directory creation
# ---------------------------------------------------------------------------

class TestDirectoryCreation:
    def test_load_creates_missing_datastore_directory(self, tmp_path, llm_settings):
        nested = tmp_path / "some" / "nested" / "dir"
        _load(tmp_path,
              llm_settings,
              datastore_name=str(pathlib.Path("some") / "nested" / "dir" / "data.json"),
              state_name=str(pathlib.Path("some") / "nested" / "dir" / "state.json"))
        assert nested.exists()


# ---------------------------------------------------------------------------
# Sidecar GC configuration
#
# These guard a failure mode that destroys user data without any error: the mark phase deletes whatever it does
# not mark, so an extractor that misses a kind of reference sweeps away live attachments on the next cleanup.
# Nothing else names a sidecar — they are content-addressed — so a wrong sweep is unrecoverable and silent.
# ---------------------------------------------------------------------------

def _payload_with_refs(image_url=None, text_file_url=None, original_sidecar=None):
    """A node payload referencing the given sidecars, in the shapes the mark phase has to recognize."""
    content = []
    if image_url is not None:
        content.append({"type": "image_url", "image_url": {"url": image_url}})
    if text_file_url is not None:
        content.append({"type": "text_file", "text_file": {"url": text_file_url, "name": "doc.pdf"}})
    payload = {"message": {"role": "user", "content": content}, "general_metadata": {}}
    if original_sidecar is not None:
        # The preserved-original case: referenced from provenance, with no content-part of its own.
        payload["general_metadata"]["sidecars"] = {"primary.png": {"original_sidecar": original_sidecar}}
    return payload


class TestSidecarRefsInPayload:
    def test_collects_image_document_and_preserved_original(self):
        refs = appstate.sidecar_refs_in_payload(_payload_with_refs(image_url="sidecar:img.png",
                                                                  text_file_url="sidecar:doc.pdf",
                                                                  original_sidecar="orig.png"))
        assert refs == {"img.png", "doc.pdf", "orig.png"}

    def test_documents_alone_are_seen(self):
        # The regression this pins: an extractor that only knew about images would return an empty set here,
        # and every attached document in the datastore would be swept on the next cleanup.
        assert appstate.sidecar_refs_in_payload(_payload_with_refs(text_file_url="sidecar:doc.pdf")) == {"doc.pdf"}

    def test_images_alone_are_seen(self):
        assert appstate.sidecar_refs_in_payload(_payload_with_refs(image_url="sidecar:img.png")) == {"img.png"}

    def test_payload_without_attachments_yields_nothing(self):
        assert appstate.sidecar_refs_in_payload(_payload_with_refs()) == set()


class TestLoadConfiguresSidecarGC:
    def test_loaded_datastore_can_determine_references(self, tmp_path, llm_settings):
        # Without an extractor configured, `unreferenced_sidecars` reports nothing at all (it fails safe rather
        # than deleting blind), which would leave the cleanup feature inert while looking like it worked.
        datastore, _, _, _ = _load(tmp_path, llm_settings)
        orphan = datastore.store_sidecar(b"not referenced by anything", "png")
        assert datastore.unreferenced_sidecars() == [orphan]

    def test_referenced_attachments_survive_a_sweep(self, tmp_path, llm_settings):
        # The end-to-end property both halves of the extractor exist for: an image and a document attached to a
        # reachable node are still there after a cleanup, and only the orphan goes.
        datastore, state, _, _ = _load(tmp_path, llm_settings)
        image = datastore.store_sidecar(b"image bytes", "png")
        document = datastore.store_sidecar(b"document bytes", "pdf")
        orphan = datastore.store_sidecar(b"orphan bytes", "png")
        datastore.create_node(_payload_with_refs(image_url=f"sidecar:{image}",
                                                text_file_url=f"sidecar:{document}"),
                              parent_id=state["HEAD"])

        assert datastore.prune_unreferenced_sidecars() == [orphan]
        assert datastore.sidecar_path(image).exists()
        assert datastore.sidecar_path(document).exists()
        assert not datastore.sidecar_path(orphan).exists()
