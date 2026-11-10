"""Tests for `raven.librarian.skills`, and its wiring into the turn."""

import json
import pathlib
import time

import pytest
from unpythonic import dyn
from unpythonic.env import env

from raven.librarian import chattree
from raven.librarian import chatutil
from raven.librarian import llmclient
from raven.librarian import llmtools
from raven.librarian import scaffold
from raven.librarian import skills


def _write_skill(root: pathlib.Path, name: str, *, description: str = "Does a thing. Use when asked.", body: str = "Step one.") -> pathlib.Path:
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {description}\n---\n\n{body}\n", encoding="utf-8")
    return directory

@pytest.fixture
def settings():
    model_info = env(label="m", model_id="m", context_length=4096, is_vlm=False, loaded=True)
    return llmclient.configure(model_info=model_info, backend_flavor="generic", backend_url="http://x", quiet=True)

@pytest.fixture
def root(tmp_path):
    return tmp_path / "skills"

@pytest.fixture
def library(settings, root):
    _write_skill(root, "alpha")
    the_library = skills.SkillLibrary(settings, directory=root)
    the_library.start()
    yield the_library
    the_library.shutdown()

def _spec(settings) -> dict:
    return next(spec for spec in settings.tools if spec["function"]["name"] == llmtools.SKILL_TOOL_NAME)

def _activate(settings, **arguments):
    return settings.tool_entrypoints[llmtools.SKILL_TOOL_NAME](**arguments)


class TestScan:
    def test_a_missing_directory_is_no_skills_and_no_problems(self, root):
        assert skills.scan(root) == ({}, {})

    def test_finds_skills_and_accepts_a_string_path(self, root):
        _write_skill(root, "beta")
        _write_skill(root, "alpha")
        found, problems = skills.scan(str(root))
        assert list(found) == ["alpha", "beta"]
        assert problems == {}

    def test_ignores_a_folder_with_no_skill_file(self, root):
        (root / "not-a-skill").mkdir(parents=True)
        assert skills.scan(root) == ({}, {})

    def test_a_name_differing_from_its_directory_warns_and_loads(self, root, caplog):
        directory = _write_skill(root, "folder-name")
        (directory / "SKILL.md").write_text("---\nname: other-name\ndescription: D.\n---\nB\n", encoding="utf-8")
        found, _problems = skills.scan(root)
        assert list(found) == ["other-name"]
        assert "differs" in caplog.text

    def test_a_missing_description_is_reported_and_the_others_load(self, root):
        directory = _write_skill(root, "broken")
        (directory / "SKILL.md").write_text("---\nname: broken\n---\nB\n", encoding="utf-8")
        _write_skill(root, "fine")
        found, problems = skills.scan(root)
        assert list(found) == ["fine"]
        assert "description" in problems["broken"]

    @pytest.mark.parametrize("content", ["no frontmatter at all", "---\n: : :\n  - [\n---\nB\n", "---\n- a list\n---\nB\n"])
    def test_an_unusable_file_is_a_problem_and_not_an_exception(self, root, content):
        directory = _write_skill(root, "broken")
        (directory / "SKILL.md").write_text(content, encoding="utf-8")
        found, problems = skills.scan(root)
        assert found == {}
        assert "broken" in problems

    def test_an_unquoted_colon_in_a_value_is_tolerated(self, root):
        directory = _write_skill(root, "colon")
        (directory / "SKILL.md").write_text("---\nname: colon\ndescription: Use this when: the user asks\n---\nB\n", encoding="utf-8")
        found, problems = skills.scan(root)
        assert problems == {}
        assert found["colon"].description == "Use this when: the user asks"


class TestRegistration:
    def test_no_skills_means_no_tool_and_an_unchanged_request(self, settings, root):
        before = json.dumps(settings.request_data, sort_keys=True, default=str)
        with skills.loaded(settings, directory=root):
            assert llmtools.SKILL_TOOL_NAME not in settings.tool_entrypoints
            assert json.dumps(settings.request_data, sort_keys=True, default=str) == before

    def test_the_catalog_is_in_the_tools_description_and_the_name_is_constrained(self, settings, library):
        spec = _spec(settings)
        assert "alpha: Does a thing." in spec["function"]["description"]
        assert spec["function"]["parameters"]["properties"]["name"]["enum"] == ["alpha"]

    def test_leaving_the_block_removes_it(self, settings, root):
        _write_skill(root, "alpha")
        before = list(settings.tools)
        with skills.loaded(settings, directory=root):
            assert settings.tools != before
        assert settings.tools == before
        assert settings.skill_names == frozenset()
        assert llmtools.SKILL_TOOL_NAME not in settings.tool_entrypoints

    def test_the_module_level_tables_are_left_alone(self, library):
        assert llmtools.SKILL_TOOL_NAME not in llmtools.TOOL_ENTRYPOINTS

    def test_the_gate_withholds_it(self, settings, library):
        offered = llmtools.maybe_tool_names_for_turn(settings, documents_available=True, internet_available=True, mcp_available=True, skills_available=False)
        assert llmtools.SKILL_TOOL_NAME not in offered
        assert "websearch" in offered
        assert llmtools.maybe_tool_names_for_turn(settings, documents_available=True, internet_available=True, mcp_available=True, skills_available=True) is None

    def test_its_cost_is_counted_in_its_own_group(self, settings, library):
        assert llmclient.tool_cost_by_group(settings)["skills"] > 0

    def test_it_survives_a_reconnect(self, settings, library, monkeypatch):
        monkeypatch.setattr(llmclient, "detect_backend_flavor", lambda backend_url: "generic")
        monkeypatch.setattr(llmclient, "resolve_model_info", lambda backend_url, backend_flavor: settings.model_record)
        llmclient.reconnect(settings)
        assert llmtools.SKILL_TOOL_NAME in settings.tool_entrypoints


class TestActivation:
    def test_returns_the_body_without_the_frontmatter_and_says_which(self, settings, library):
        text, metadata = _activate(settings, name="alpha")
        assert "Step one." in text
        assert "description:" not in text
        assert text.startswith('<skill_content name="alpha">')
        assert metadata == {"skill": "alpha"}

    def test_an_edit_takes_effect_without_a_reload(self, settings, library, root):
        _write_skill(root, "alpha", body="Step two.")
        assert "Step two." in _activate(settings, name="alpha")[0]

    def test_an_unknown_name_is_answered(self, settings, library):
        assert _activate(settings, name="nope") == skills.CANONICAL_NO_SUCH_SKILL.format(name="nope")

    def test_a_skill_already_on_the_branch_is_not_loaded_again(self, settings, library):
        with dyn.let(tool_context=env(activated_skills=frozenset({"alpha"}))):
            assert _activate(settings, name="alpha") == skills.CANONICAL_SKILL_ALREADY_ACTIVE.format(name="alpha")

    def test_bundled_files_are_listed_and_not_read(self, settings, library, root):
        (root / "alpha" / "references").mkdir()
        (root / "alpha" / "references" / "guide.md").write_text("The secret of the guide.", encoding="utf-8")
        text, _metadata = _activate(settings, name="alpha")
        assert "<file>references/guide.md</file>" in text
        assert "The secret of the guide." not in text

    def test_a_bundled_file_can_be_read(self, settings, library, root):
        (root / "alpha" / "guide.md").write_text("The guide.", encoding="utf-8")
        text, metadata = _activate(settings, name="alpha", file="guide.md")
        assert text == "The guide."
        assert metadata == {"skill_file": "alpha/guide.md"}

    @pytest.mark.parametrize("escape", ["../secret.txt", "../../secret.txt", "/etc/passwd"])
    def test_a_path_outside_the_skill_is_refused(self, settings, library, root, escape):
        (root / "secret.txt").write_text("secret", encoding="utf-8")
        result = _activate(settings, name="alpha", file=escape)
        assert result == skills.CANONICAL_NO_SUCH_SKILL_FILE.format(name="alpha", file=escape)

    def test_a_skill_with_scripts_says_they_cannot_be_run(self, settings, library, root):
        assert skills.CANONICAL_SCRIPTS_UNAVAILABLE not in _activate(settings, name="alpha")[0], "this skill has scripts already, so this fixture cannot tell"
        (root / "alpha" / "scripts").mkdir()
        (root / "alpha" / "scripts" / "run.py").write_text("print(1)", encoding="utf-8")
        text = _activate(settings, name="alpha")[0]
        assert skills.CANONICAL_SCRIPTS_UNAVAILABLE in text
        assert "run.py" not in text


class TestUserActivation:
    @pytest.fixture
    def chat(self, settings):
        datastore = chattree.Forest()
        root_id = datastore.create_node(payload=chatutil.create_payload(llm_settings=settings, message=chatutil.create_initial_system_message(settings)), parent_id=None)
        return datastore, root_id

    def _turn(self, settings, datastore, head, text, enabled=True):
        return scaffold.user_turn(llm_settings=settings, datastore=datastore, head_node_id=head, user_message_text=text, skills_enabled=enabled)

    @staticmethod
    def _roles(datastore, head):
        return [datastore.get_payload(node_id)["message"]["role"] for node_id in datastore.linearize_up(head)]

    @pytest.mark.parametrize("text, expected", [("/alpha do it", ("alpha", "do it")),
                                                ("/alpha", ("alpha", "")),
                                                ("/nope do it", (None, "/nope do it")),
                                                ("/home/jje/notes", (None, "/home/jje/notes")),
                                                ("no command", (None, "no command"))])
    def test_parsing(self, settings, library, text, expected):
        assert skills.parse_user_activation(settings, text) == expected

    def test_the_exchange_is_stored_ahead_of_the_message(self, settings, library, chat):
        datastore, root_id = chat
        head = self._turn(settings, datastore, root_id, "/alpha do it")
        assert self._roles(datastore, head) == ["system", "assistant", "tool", "user"]
        assert chatutil.content_to_text(datastore.get_payload(head)["message"]["content"]).endswith(": do it"), "the message keeps what follows the command"

    def test_the_synthetic_message_has_no_text_and_no_reasoning_and_is_marked_automatic(self, settings, library, chat):
        datastore, root_id = chat
        head = self._turn(settings, datastore, root_id, "/alpha do it")
        call = datastore.get_payload(datastore.linearize_up(head)[1])
        assert chatutil.content_to_text(call["message"]["content"]) == ""
        assert "reasoning_content" not in call["message"]
        assert call["generation_metadata"]["automatic"] is True

    def test_the_tool_node_names_the_skill(self, settings, library, chat):
        datastore, root_id = chat
        head = self._turn(settings, datastore, root_id, "/alpha do it")
        assert datastore.get_payload(datastore.linearize_up(head)[2])["generation_metadata"]["skill"] == "alpha"

    def test_a_second_time_on_the_branch_stores_only_the_message(self, settings, library, chat):
        datastore, root_id = chat
        head = self._turn(settings, datastore, root_id, "/alpha first")
        head = self._turn(settings, datastore, head, "/alpha second")
        assert self._roles(datastore, head) == ["system", "assistant", "tool", "user", "user"]

    def test_another_branch_can_activate_it_again(self, settings, library, chat):
        datastore, root_id = chat
        self._turn(settings, datastore, root_id, "/alpha first")
        head = self._turn(settings, datastore, root_id, "/alpha elsewhere")
        assert self._roles(datastore, head) == ["system", "assistant", "tool", "user"]

    def test_with_skills_off_it_is_ordinary_text(self, settings, library, chat):
        datastore, root_id = chat
        head = self._turn(settings, datastore, root_id, "/alpha do it", enabled=False)
        assert self._roles(datastore, head) == ["system", "user"]
        assert chatutil.content_to_text(datastore.get_payload(head)["message"]["content"]).endswith(": /alpha do it")


class TestWatching:
    @pytest.fixture
    def quick_polls(self, monkeypatch):
        monkeypatch.setattr(skills.librarian_config, "skills_poll_interval", 0.05)

    @staticmethod
    def _wait_until(predicate, timeout=10.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(0.02)
        return False

    def test_a_new_skill_appears_and_is_announced(self, settings, root, quick_polls):
        _write_skill(root, "alpha")
        calls = []
        library = skills.SkillLibrary(settings, directory=root, on_changed=lambda: calls.append(1))
        library.start(watch=True)
        try:
            assert not calls
            _write_skill(root, "beta")
            assert self._wait_until(lambda: "beta" in settings.skill_names)
            assert self._wait_until(lambda: calls)
            assert _spec(settings)["function"]["parameters"]["properties"]["name"]["enum"] == ["alpha", "beta"]
        finally:
            library.shutdown()

    def test_an_edit_to_a_body_is_not_announced(self, settings, root, quick_polls):
        _write_skill(root, "alpha")
        calls = []
        library = skills.SkillLibrary(settings, directory=root, on_changed=lambda: calls.append(1))
        library.start(watch=True)
        try:
            _write_skill(root, "alpha", body="A much longer body than the one before it.")
            _write_skill(root, "beta")
            assert self._wait_until(lambda: "beta" in settings.skill_names), "the watcher did not run, so this fixture cannot tell a quiet edit from no watching"
            assert len(calls) <= 1
        finally:
            library.shutdown()

    def test_the_first_skill_makes_the_tool_appear(self, settings, root, quick_polls):
        root.mkdir()
        library = skills.SkillLibrary(settings, directory=root)
        library.start(watch=True)
        try:
            assert llmtools.SKILL_TOOL_NAME not in settings.tool_entrypoints
            _write_skill(root, "alpha")
            assert self._wait_until(lambda: llmtools.SKILL_TOOL_NAME in settings.tool_entrypoints)
        finally:
            library.shutdown()

    def test_a_script_does_not_follow_the_directory(self, settings, root, quick_polls):
        _write_skill(root, "alpha")
        with skills.loaded(settings, directory=root):
            _write_skill(root, "beta")
            time.sleep(0.3)
            assert "beta" not in settings.skill_names
