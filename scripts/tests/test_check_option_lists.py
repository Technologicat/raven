"""Tests for `scripts/check_option_lists.py`: how a help-card key cell is read as keys."""

import pytest

from scripts.check_option_lists import key_alternatives


@pytest.mark.parametrize("cell, keys", [
    ("Shift+Left / Right", ["Shift+Left", "Shift+Right"]),
    ("Ctrl+Shift+Z / Y", ["Ctrl+Shift+Z", "Ctrl+Shift+Y"]),
])
def test_a_bare_second_key_shares_the_first_key_s_modifiers(cell, keys):
    assert key_alternatives(cell) == keys


@pytest.mark.parametrize("cell, keys", [
    ("F4 / Shift+F4", ["F4", "Shift+F4"]),           # the second has modifiers of its own
    ("Left / Right", ["Left", "Right"]),             # the first has none to share
    ("Numpad + / Numpad -", ["Numpad +", "Numpad -"]),  # a "+" that is the key
    ("Numpad /", ["Numpad /"]),
    ("Left / Right (A / D)", ["Left", "Right", "A", "D"]),
])
def test_other_pairs_are_taken_as_written(cell, keys):
    """Control for the test above."""
    assert key_alternatives(cell) == keys
