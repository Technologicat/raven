"""Unit tests for raven.common.text.plural."""

from raven.common import text


class TestPluralS:
    def test_one_takes_the_singular(self):
        assert text.plural_s(1) == ""

    def test_zero_takes_the_plural(self):
        # As English does: "0 nodes deleted", not "0 node deleted". The tempting spelling of this
        # helper is `count > 1`, which gets this one wrong and is otherwise indistinguishable.
        assert text.plural_s(0) == "s"

    def test_many_take_the_plural(self):
        for count in (2, 3, 17, 5167):
            assert text.plural_s(count) == "s", count

    def test_it_reads_as_the_noun_it_agrees_with(self):
        # How it is meant to be used, and the only assertion here that would catch the suffix being
        # returned with a stray space or the noun being pluralized twice.
        assert [f"{n} node{text.plural_s(n)}" for n in (0, 1, 2)] == ["0 nodes", "1 node", "2 nodes"]
