"""Tests for `raven.configoverrides`.

The module's whole job is to let a machine's own answers live outside the repository, so what is asserted
here is the two halves of that: a setting named correctly takes effect, and one named incorrectly is
refused and reported rather than silently inventing a field nobody reads.

**Several tests here assert that something did *not* happen**, which is an outcome an inert loader would
produce just as well. Each of those therefore also asserts that a sibling setting in the same file *was*
applied — the negative control, without which the test would pass against a `configoverrides.apply` whose
body was `return []`.
"""

import ast
import json
import pathlib

import pytest

from unpythonic.env import env

from raven import configoverrides

#: Every config module in the constellation, found rather than listed — a new component's config is then
#: covered by the wiring test below from the day it lands, which is the day it would otherwise be missed.
_RAVEN = pathlib.Path(configoverrides.__file__).resolve().parent
_CONFIG_MODULES = frozenset(path for path in _RAVEN.rglob("config*.py")
                            if "vendor" not in path.parts and "tests" not in path.parts
                            and path.name != "configoverrides.py")  # the loader itself, not a thing it loads


@pytest.fixture
def write_overrides(tmp_path):
    """Return a function that writes an override file and hands back its path."""
    def write(contents: dict) -> pathlib.Path:
        path = tmp_path / "overrides.json"
        path.write_text(json.dumps(contents), encoding="utf-8")
        return path
    return write


def make_namespace() -> dict:
    """A stand-in config module's globals, holding one of each shape a real config module has."""
    return {"a_string": "shipped",
            "a_number": 5,
            "a_float": 0.5,
            "a_flag": True,
            "a_color": (0, 0, 0, 255),
            "a_path": pathlib.Path("/shipped"),
            "unset": None,
            "gui_config": env(width=768, height=768)}


def test_no_override_file_leaves_every_default_alone(tmp_path):
    """The ordinary case, and the one every user who never writes the file is in."""
    namespace = make_namespace()
    applied = configoverrides.apply("raven.demo.config", namespace, path=tmp_path / "absent.json")
    assert applied == []
    assert namespace == make_namespace()


def test_a_module_level_setting_is_rebound(write_overrides):
    path = write_overrides({"raven.demo.config": {"a_string": "local"}})
    namespace = make_namespace()
    assert configoverrides.apply("raven.demo.config", namespace, path=path) == ["a_string"]
    assert namespace["a_string"] == "local"


def test_a_dotted_name_reaches_inside_an_env(write_overrides):
    """The Visualizer's and Librarian's `gui_config` is an `env`, so half their settings live one level down."""
    path = write_overrides({"raven.demo.config": {"gui_config.width": 1024}})
    namespace = make_namespace()
    assert configoverrides.apply("raven.demo.config", namespace, path=path) == ["gui_config.width"]
    assert namespace["gui_config"].width == 1024
    assert namespace["gui_config"].height == 768  # its neighbour is untouched


def test_entries_for_other_modules_are_not_applied(write_overrides):
    path = write_overrides({"raven.other.config": {"a_string": "wrong module"},
                            "raven.demo.config": {"a_number": 7}})
    namespace = make_namespace()
    assert configoverrides.apply("raven.demo.config", namespace, path=path) == ["a_number"]
    assert namespace["a_string"] == "shipped"


def test_a_name_that_matches_nothing_is_refused_and_reported(write_overrides, caplog):
    """A typo must not create a setting: a config module's names are its whole API, and a new one is dead."""
    path = write_overrides({"raven.demo.config": {"a_strnig": "typo", "a_number": 7}})
    namespace = make_namespace()
    with caplog.at_level("WARNING", logger="raven.configoverrides"):
        applied = configoverrides.apply("raven.demo.config", namespace, path=path)
    assert applied == ["a_number"], "the good sibling did not apply either, so this fixture cannot tell a refusal from an inert loader"
    assert "a_strnig" not in namespace
    assert "a_strnig" in caplog.text


def test_a_typo_inside_an_env_does_not_add_a_field(write_overrides, caplog):
    """`env` accepts a brand-new binding without complaint, so nothing downstream would notice. Checked here."""
    path = write_overrides({"raven.demo.config": {"gui_config.widht": 1024, "gui_config.height": 512}})
    namespace = make_namespace()
    with caplog.at_level("WARNING", logger="raven.configoverrides"):
        applied = configoverrides.apply("raven.demo.config", namespace, path=path)
    assert applied == ["gui_config.height"], "the good sibling did not apply either, so this fixture proves nothing about the typo"
    assert not hasattr(namespace["gui_config"], "widht")
    assert "widht" in caplog.text


def test_a_path_arrives_as_a_string_and_is_expanded(write_overrides):
    """JSON has no path type, and `~` in a hand-written config file is what a person would write."""
    path = write_overrides({"raven.demo.config": {"a_path": "~/elsewhere"}})
    namespace = make_namespace()
    configoverrides.apply("raven.demo.config", namespace, path=path)
    assert namespace["a_path"] == pathlib.Path.home() / "elsewhere"
    assert namespace["a_path"].is_absolute()


def test_a_tuple_default_accepts_a_json_list(write_overrides):
    """JSON has one kind of sequence; the colors and sizes in these configs are tuples."""
    path = write_overrides({"raven.demo.config": {"a_color": [255, 128, 0, 255]}})
    namespace = make_namespace()
    configoverrides.apply("raven.demo.config", namespace, path=path)
    assert namespace["a_color"] == (255, 128, 0, 255)
    assert isinstance(namespace["a_color"], tuple)


def test_a_float_default_accepts_a_whole_number(write_overrides):
    """JSON writes `3` rather than `3.0`, and refusing that would be pedantry rather than safety."""
    path = write_overrides({"raven.demo.config": {"a_float": 3}})
    namespace = make_namespace()
    configoverrides.apply("raven.demo.config", namespace, path=path)
    assert namespace["a_float"] == 3.0
    assert isinstance(namespace["a_float"], float)


def test_a_default_of_none_accepts_whatever_it_is_given(write_overrides):
    """Plenty of settings ship as "unset" — an audio device, an API key path — so there is no shape to fit to."""
    path = write_overrides({"raven.demo.config": {"unset": "Built-in Audio Analog Stereo"}})
    namespace = make_namespace()
    configoverrides.apply("raven.demo.config", namespace, path=path)
    assert namespace["unset"] == "Built-in Audio Analog Stereo"


@pytest.mark.parametrize("name, bad_value", [("a_string", 5),
                                             ("a_number", "five"),
                                             ("a_flag", 1),          # `bool` is an `int`; the check must not let this through
                                             ("a_number", True)])    # ...nor this, the other way round
def test_a_value_of_the_wrong_shape_leaves_the_default_in_place(write_overrides, caplog, name, bad_value):
    path = write_overrides({"raven.demo.config": {name: bad_value, "a_color": [1, 2, 3, 4]}})
    namespace = make_namespace()
    with caplog.at_level("WARNING", logger="raven.configoverrides"):
        applied = configoverrides.apply("raven.demo.config", namespace, path=path)
    assert applied == ["a_color"], "the good sibling did not apply either, so this fixture cannot tell a refusal from an inert loader"
    assert namespace[name] == make_namespace()[name]
    assert name in caplog.text


def test_a_malformed_file_is_reported_and_does_not_stop_the_app(tmp_path, caplog):
    """An app that refused to start over a stray comma would be worse than one that says so and carries on."""
    path = tmp_path / "overrides.json"
    path.write_text('{"raven.demo.config": {"a_string": "local",}}', encoding="utf-8")  # trailing comma
    namespace = make_namespace()
    with caplog.at_level("ERROR", logger="raven.configoverrides"):
        applied = configoverrides.apply("raven.demo.config", namespace, path=path)
    assert applied == []
    assert namespace == make_namespace()
    assert str(path) in caplog.text, "nothing was reported, so a user would see only a setting that did not take effect"


def test_a_file_that_is_not_an_object_is_reported(tmp_path, caplog):
    path = tmp_path / "overrides.json"
    path.write_text('["raven.demo.config"]', encoding="utf-8")
    namespace = make_namespace()
    with caplog.at_level("ERROR", logger="raven.configoverrides"):
        assert configoverrides.apply("raven.demo.config", namespace, path=path) == []
    assert str(path) in caplog.text


def test_a_top_level_key_that_is_not_a_config_module_is_reported(write_overrides, caplog):
    """A mistyped *module* name is claimed by nobody, so no module can report it. This is the one guard there is."""
    path = write_overrides({"librarian": {"llm_backend_url": "http://elsewhere:1234"}})
    namespace = make_namespace()
    with caplog.at_level("WARNING", logger="raven.configoverrides"):
        configoverrides.apply("raven.demo.config", namespace, path=path)
    assert "librarian" in caplog.text


@pytest.mark.parametrize("module_path", sorted(_CONFIG_MODULES))
def test_every_config_module_applies_overrides_as_its_last_statement(module_path):
    """Guards the wiring rather than the loader, across ten call sites that have nothing else in common.

    A config module that forgot the call is the failure with no symptom: it looks configurable, the
    override file's entry for it is claimed by nobody, and the user sees a setting that did not take
    effect. Nothing in the loader can report that, because the loader never runs.

    **Read as source rather than imported**, which is what lets the assertion be about *position*. The call
    has to come last, below every setting, because a name cannot be rebound before it exists — and an
    import can only show that the module loaded, not that a setting defined after the call was left at its
    shipped value.
    """
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    calls = [(index, node) for index, node in enumerate(tree.body)
             if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
             and ast.unparse(node.value.func) == "configoverrides.apply"]
    assert len(calls) == 1, f"{module_path.name} should call `configoverrides.apply` exactly once at module level"

    index, node = calls[0]
    assert index == len(tree.body) - 1, f"in {module_path.name}, settings are defined after the `configoverrides.apply` call, and those cannot be overridden"
    assert [ast.unparse(arg) for arg in node.value.args] == ["__name__", "globals()"], f"in {module_path.name}, the call must pass its own module name and namespace"
