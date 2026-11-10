"""Agent skills: procedures written in Markdown, which the LLM follows when a task matches.

A skill is a folder holding a `SKILL.md`, in the format of the [Agent Skills](https://agentskills.io) standard:
YAML frontmatter with a `name` and a `description`, and a Markdown body with the instructions. Skills say how
to use the tools that exist, which is material that belongs in no single tool's description.

The LLM sees a catalog of names and descriptions, in the description of one tool, and calls that tool to read
a skill's instructions when it decides that a task matches. In a script::

    llm_settings = llmclient.setup(backend_url)
    with skills.loaded(llm_settings):
        record = agent.turn(llm_settings, "Do the thing.", skills_enabled=True)

A long-lived frontend calls `SkillLibrary.start` and `SkillLibrary.shutdown` instead.
"""

__all__ = ["CANONICAL_NO_SUCH_SKILL", "CANONICAL_SKILL_ALREADY_ACTIVE",
           "CANONICAL_NO_SUCH_SKILL_FILE", "CANONICAL_SCRIPTS_UNAVAILABLE",

           "Skill", "scan",

           "parse_user_activation",

           "SkillLibrary", "loaded"]

import contextlib
import dataclasses
import json
import logging
import pathlib
import re
import threading
from collections.abc import Callable, Iterator
from typing import Any

import yaml
from unpythonic import dyn
from unpythonic.env import env

from . import config as librarian_config
from . import llmtools

logger = logging.getLogger(__name__)

_SKILL_FILE = "SKILL.md"
_MAX_NAME_LENGTH = 64
_MAX_LISTED_RESOURCES = 50

# What the model is told. Statements of fact, like the built-in tools' canned replies.
CANONICAL_NO_SUCH_SKILL = "There is no skill named '{name}'."
CANONICAL_SKILL_ALREADY_ACTIVE = "The skill '{name}' is already loaded in this conversation; its instructions are above."
CANONICAL_NO_SUCH_SKILL_FILE = "The skill '{name}' has no readable file at '{file}'."
CANONICAL_SCRIPTS_UNAVAILABLE = "This skill bundles scripts. Scripts cannot be run here."


# --------------------------------------------------------------------------------
# Finding and reading skills

@dataclasses.dataclass(frozen=True)
class Skill:
    """One installed skill. `directory` is the folder holding its `SKILL.md`."""
    name: str
    description: str
    directory: pathlib.Path

def _split_frontmatter(text: str) -> tuple[str, str] | None:
    """Split a `SKILL.md` into `(frontmatter, body)`, or return `None` if it has no frontmatter."""
    match = re.match(r"\A---[ \t]*\r?\n(.*?)^---[ \t]*\r?$\n?(.*)\Z", text, flags=re.DOTALL | re.MULTILINE)
    return (match.group(1), match.group(2).strip()) if match else None

def _parse_yaml(frontmatter: str) -> Any:
    """Parse frontmatter, tolerating the commonest invalidity in skills written for other clients.

    That is an unquoted value containing a colon, `description: Use when: the user asks`. Their parsers accept
    it, so on a failure the plain top-level values are quoted and it is tried once more.
    """
    try:
        return yaml.safe_load(frontmatter)
    except yaml.YAMLError:
        def quote(match: re.Match) -> str:
            return f"{match.group(1)}: {json.dumps(match.group(2))}"
        return yaml.safe_load(re.sub(r"^([A-Za-z_-]+):[ \t]+([^\s\"'|>\[{].*?)[ \t]*$", quote, frontmatter, flags=re.MULTILINE))

def _read_skill(skill_file: pathlib.Path) -> Skill:
    """Read one `SKILL.md`. Raises `ValueError` saying why it cannot be used.

    Lenient, as the standard's integration guide recommends, so that skills written for other clients still
    load: a problem with the name is logged, and only a missing description or unusable YAML is fatal.
    """
    directory = skill_file.parent
    try:
        parts = _split_frontmatter(skill_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"cannot read it: {type(exc)}: {exc}") from exc
    if parts is None:
        raise ValueError("it has no YAML frontmatter")
    try:
        metadata = _parse_yaml(parts[0])
    except yaml.YAMLError as exc:
        raise ValueError(f"its YAML cannot be parsed: {exc}") from exc
    if not isinstance(metadata, dict):
        raise ValueError("its frontmatter is not a mapping")

    description = metadata.get("description")
    if not isinstance(description, str) or not description.strip():
        raise ValueError("it has no description")
    name = str(metadata.get("name") or directory.name)
    if name != directory.name:
        logger.warning(f"_read_skill: '{skill_file}': the name '{name}' differs from its directory's; loading it anyway.")
    if len(name) > _MAX_NAME_LENGTH:
        logger.warning(f"_read_skill: '{skill_file}': the name is longer than {_MAX_NAME_LENGTH} characters; loading it anyway.")
    return Skill(name=name, description=description.strip(), directory=directory)

def scan(directory: str | pathlib.Path) -> tuple[dict[str, Skill], dict[str, str]]:
    """Find the skills in `directory`, each in a subfolder of its own.

    Returns `(skills, problems)`. `skills` maps names to `Skill`, in name order. `problems` maps a subfolder's
    name to why it could not be used. A missing directory means no skills and no problems.
    """
    directory = pathlib.Path(directory).expanduser()
    skills, problems = {}, {}
    for skill_file in sorted(directory.glob(f"*/{_SKILL_FILE}")):
        try:
            skill = _read_skill(skill_file)
        except ValueError as exc:
            problems[skill_file.parent.name] = str(exc)
            continue
        if skill.name in skills:
            logger.warning(f"scan: two skills are named '{skill.name}'; keeping the one in '{skills[skill.name].directory}'.")
            continue
        skills[skill.name] = skill
    return dict(sorted(skills.items())), problems


def parse_user_activation(settings: env, text: str) -> tuple[str | None, str]:
    """If a user's message asks for a skill, as `/skill-name ...`, return `(name, rest_of_message)`.

    Otherwise return `(None, text)`. A `/word` that names no installed skill is ordinary text, as it may be a path.
    """
    match = re.match(r"\A/([a-z0-9-]+)(?:\s+(.*))?\Z", text.strip(), flags=re.DOTALL)
    if match is None or match.group(1) not in settings.skill_names:
        return None, text
    return match.group(1), (match.group(2) or "").strip()


# --------------------------------------------------------------------------------
# The library

def _tool_spec(skills: dict[str, Skill]) -> dict[str, Any]:
    """The specification of the activation tool, with the catalog in its description."""
    catalog = "\n".join(f"- {skill.name}: {skill.description}" for skill in skills.values())
    return {"type": "function",
            "function": {"name": llmtools.SKILL_TOOL_NAME,
                         "description": ("Load the instructions of a skill, which is a procedure for a specific kind of task. "
                                         "When a task matches one of the skills below, call this first and then follow the instructions.\n\n"
                                         f"Skills:\n{catalog}"),
                         "parameters": {"type": "object",
                                        "properties": {"name": {"type": "string",
                                                                "enum": sorted(skills),
                                                                "description": "Which skill to load."},
                                                       "file": {"type": "string",
                                                                "description": "A file the skill's instructions refer to, as a path relative to the skill. Omit this to load the instructions."}},
                                        "required": ["name"],
                                        "additionalProperties": False}}}

def _list_resources(skill: Skill) -> list[str]:
    """The files bundled with a skill that the model could ask for, as relative paths."""
    root = skill.directory.resolve()
    files = (path.relative_to(root).as_posix() for path in sorted(root.rglob("*")) if path.is_file())
    return [name for name in files if name != _SKILL_FILE and not name.startswith("scripts/")][:_MAX_LISTED_RESOURCES]

class SkillLibrary:
    """The installed skills, and the tool that lets the LLM read them.

    `settings`: from `llmclient.setup` or `llmclient.connect`. `start` adds the tool to it, and `shutdown` removes it.

    `directory`: where the skills are. Default is `librarian_config.skills_dir`.

    `on_changed`: 0-argument callable, called on a background thread when the set of skills, or a description,
                  has changed on disk.
    """
    def __init__(self,
                 settings: env,
                 *,
                 directory: str | pathlib.Path | None = None,
                 on_changed: Callable[[], None] | None = None) -> None:
        self.settings = settings
        self.directory = pathlib.Path(directory if directory is not None else librarian_config.skills_dir).expanduser()
        self.on_changed = on_changed
        self._skills: dict[str, Skill] = {}
        self._problems: dict[str, str] = {}
        self._registered = False
        self._maybe_watcher: threading.Thread | None = None
        self._stop = threading.Event()

    def _get_skill_names(self) -> list[str]:
        """Copy, so that a caller cannot edit the library's record."""
        return list(self._skills)
    skill_names = property(fget=_get_skill_names, doc="The names of the installed skills, in name order.")

    def _get_problems(self) -> dict[str, str]:
        """Copy, so that a caller cannot edit the library's record."""
        return dict(self._problems)
    problems = property(fget=_get_problems, doc="Why a skill could not be used, by the name of its folder.")

    def start(self, *, watch: bool = False) -> None:
        """Find the skills, and add the activation tool to the settings if there are any.

        `watch`: follow changes to the directory. For a long-lived frontend. A script leaves this off, so that
                 what it offers cannot change during a run.
        """
        signature = self._signature()  # before reading, so that an edit right after the read is seen
        self._refresh()
        if watch:
            self._maybe_watcher = threading.Thread(target=self._watch, args=(signature,), name="skill_library_watcher", daemon=True)
            self._maybe_watcher.start()

    def shutdown(self) -> None:
        """Remove the tool from the settings, and stop following the directory. Safe to call more than once."""
        self._stop.set()
        if self._maybe_watcher is not None:
            self._maybe_watcher.join(timeout=2 * librarian_config.skills_poll_interval)
            self._maybe_watcher = None
        self._skills = {}
        self._install()

    def _refresh(self) -> None:
        self._skills, self._problems = scan(self.directory)
        for name, problem in self._problems.items():
            logger.error(f"SkillLibrary: skipping '{name}': {problem}")
        self._install()
        plural_s = "s" if len(self._skills) != 1 else ""
        logger.info(f"SkillLibrary: {len(self._skills)} skill{plural_s} in '{self.directory}'.")

    def _install(self) -> None:
        settings = self.settings
        settings.skill_names = frozenset(self._skills)
        specs = [_tool_spec(self._skills)] if self._skills else []
        entrypoints = {llmtools.SKILL_TOOL_NAME: self._activate} if self._skills else {}
        previous = {llmtools.SKILL_TOOL_NAME} if self._registered else set()
        llmtools.replace_tools(settings, previous_names=previous, specs=specs, entrypoints=entrypoints)
        self._registered = bool(self._skills)

    def _signature(self) -> tuple:
        def stamp(path: pathlib.Path) -> tuple:
            try:
                info = path.stat()
                return (str(path), info.st_mtime_ns, info.st_size)
            except OSError:  # deleted between the listing and now
                return (str(path), None, None)
        return tuple(sorted(stamp(path) for path in self.directory.glob(f"*/{_SKILL_FILE}")))

    def _catalog(self) -> list[tuple[str, str]]:
        return [(skill.name, skill.description) for skill in self._skills.values()]

    def _watch(self, signature: tuple) -> None:
        while not self._stop.wait(librarian_config.skills_poll_interval):
            current = self._signature()
            if current == signature:
                continue
            signature = current
            catalog = self._catalog()
            try:
                self._refresh()
            except Exception:  # noqa: BLE001 -- the next change to the directory gets another try
                logger.exception("SkillLibrary: re-reading the skills failed")
                continue
            # An edit to a skill's body needs no notice, the body being read at activation.
            if self._catalog() != catalog and self.on_changed is not None:
                try:
                    self.on_changed()
                except Exception:  # noqa: BLE001 -- a frontend's handler must not stop the watcher
                    logger.exception("SkillLibrary: ignoring exception from `on_changed`")

    def _activate(self, name: str, file: str | None = None) -> str | tuple[str, dict[str, str]]:
        """The activation tool: the instructions of skill `name`, or one of its bundled files."""
        skill = self._skills.get(name)
        if skill is None:
            return CANONICAL_NO_SUCH_SKILL.format(name=name)
        if file is not None:
            return self._read_bundled_file(skill, file)
        if name in getattr(dyn.tool_context, "activated_skills", frozenset()):
            return CANONICAL_SKILL_ALREADY_ACTIVE.format(name=name)

        # Read here rather than at discovery, so that an edit takes effect without a reload.
        try:
            parts = _split_frontmatter((skill.directory / _SKILL_FILE).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError) as exc:
            logger.warning(f"SkillLibrary._activate: '{name}': {type(exc)}: {exc}")
            parts = None
        if parts is None:
            return CANONICAL_NO_SUCH_SKILL_FILE.format(name=name, file=_SKILL_FILE)

        lines = [f'<skill_content name="{name}">', parts[1], ""]
        # TODO: Remove this line if script support is ever added.
        if (skill.directory / "scripts").is_dir():
            lines += [CANONICAL_SCRIPTS_UNAVAILABLE, ""]
        resources = _list_resources(skill)
        if resources:
            lines += ["<skill_resources>", *(f"  <file>{resource}</file>" for resource in resources), "</skill_resources>"]
        lines.append("</skill_content>")
        # The metadata is what lets the branch be asked which skills are active, and later lets context
        # compaction tell a skill from other tool results.
        return "\n".join(lines), {"skill": name}

    @staticmethod
    def _read_bundled_file(skill: Skill, file: str) -> str | tuple[str, dict[str, str]]:
        root = skill.directory.resolve()
        path = (root / file).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            return CANONICAL_NO_SUCH_SKILL_FILE.format(name=skill.name, file=file)
        try:
            return path.read_text(encoding="utf-8"), {"skill_file": f"{skill.name}/{file}"}
        except (OSError, UnicodeDecodeError) as exc:
            logger.warning(f"SkillLibrary._read_bundled_file: '{path}': {type(exc)}: {exc}")
            return CANONICAL_NO_SUCH_SKILL_FILE.format(name=skill.name, file=file)


@contextlib.contextmanager
def loaded(settings: env, *, directory: str | pathlib.Path | None = None) -> Iterator[SkillLibrary]:
    """Have the installed skills available in `settings` for the duration of a `with` block.

    Arguments as for `SkillLibrary`. Yields the library, whose `problems` say which skills could not be used.
    """
    library = SkillLibrary(settings, directory=directory)
    try:
        library.start()
        yield library
    finally:
        library.shutdown()
