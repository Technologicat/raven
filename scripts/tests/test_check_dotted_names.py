"""Whether `scripts/check_dotted_names.py` tells a real name from a stale one, and honours its markers.

Resolved against the real tree, so each case names something chosen to exercise one rule: a module, a
namespace package, a top-level definition, a name brought in by an import, a class member, an attribute
assigned in a method. Every positive case is paired with a near miss that must fail, since a resolver that
accepts everything passes every positive case.
"""

import pytest

from scripts.check_dotted_names import resolves, unresolved


@pytest.mark.parametrize("dotted", [
    "raven.librarian.scaffold",                                   # a module
    "raven.librarian",                                            # a namespace package, no `__init__.py`
    "raven.librarian.scaffold.ai_turn",                           # a top-level function
    "raven.librarian.llmclient.search_documents",                 # imported into the module from `llmtools`
    "raven.common.gui.utils.MinimumShowTime.hide",                # a class member
    "raven.client.avatar_controller.DPGAvatarController.speak_task",
    "raven.common.gui.utils.MinimumShowTime.min_duration",        # assigned as `self.min_duration` in a method
])
def test_a_name_that_exists_resolves(dotted):
    assert resolves(dotted)


@pytest.mark.parametrize("dotted", [
    "raven.librarian.scaffoldx",                                  # no such module
    "raven.librarian.scaffold.ai_turnx",                          # no such top-level name
    "raven.common.gui.utils.MinimumShowTime.hidex",               # no such class member
    "raven.client.avatar_controller.speak_task",                  # a method named as a module function
])
def test_a_near_miss_does_not(dotted):
    assert not resolves(dotted)


def test_markers_say_absent_on_purpose_and_are_reported_once_wrong():
    sites = ["somewhere:1"]
    references = {("raven.common.nonexistent", "planned"): sites,  # absent, marked: fine
                  ("raven.import_nothing", "stale"): sites,        # absent, marked: fine
                  ("raven.librarian.scaffold", "planned"): sites,  # marked, but it exists now
                  ("raven.librarian.nonexistent", None): sites,    # absent and unmarked
                  ("raven.librarian.scaffold", None): sites}       # exists, unmarked: fine
    assert set(unresolved(references)) == {("raven.librarian.scaffold", "planned"),
                                           ("raven.librarian.nonexistent", None)}
