"""Unit tests for raven.common.navhistory — the back/forward history shared by the views that need one.

The interesting cases are the ones the design was written around rather than the arithmetic: a state that
goes bad while it sits on the stack, a caller that refuses to arrive somewhere, and the button-enabling
question, which stops being "is the cursor at the end" the moment dead states are skipped.
"""

from raven.common.navhistory import NavigationHistory


class TestTheLine:
    def test_an_empty_history_has_nowhere_to_go(self):
        history = NavigationHistory()
        assert history.current is None
        assert not history.can_go_back and not history.can_go_forward
        assert history.back() is False and history.forward() is False

    def test_committing_moves_the_cursor_to_the_end(self):
        history = NavigationHistory("a")
        assert history.commit("b") is True
        assert history.current == "b"
        assert history.can_go_back and not history.can_go_forward

    def test_committing_the_state_already_current_records_nothing(self):
        # A view that redraws without moving should not fill the history with places nobody went.
        history = NavigationHistory("a")
        assert history.commit("a") is False
        assert len(history) == 1

    def test_committing_the_state_behind_is_still_a_commit(self):
        # The control for the above: only the *current* state is the one a commit can be a no-op against.
        # Walking back to "a" and then to "b" again is two places visited, not a duplicate.
        history = NavigationHistory("a")
        history.commit("b")
        history.back()
        assert history.commit("b") is True, "returning somewhere is a move, and the history should say so"

    def test_going_somewhere_new_abandons_what_was_ahead(self):
        history = NavigationHistory("a")
        history.commit("b")
        history.commit("c")
        history.back()          # on "b", with "c" ahead
        history.commit("d")
        assert history.states == ("a", "b", "d")
        assert not history.can_go_forward

    def test_equality_is_the_caller_s_to_define(self):
        # Visualizer's selection is the case: the same set of indices in another order is the same
        # selection, and a history using `==` would record a move that did not happen.
        history = NavigationHistory([1, 2, 3], same=lambda a, b: set(a) == set(b))
        assert history.commit([3, 2, 1]) is False
        assert history.commit([1, 2]) is True

    def test_without_a_predicate_that_reorder_is_a_move(self):
        # The control: with plain equality the very same fixture records it, so the test above is
        # measuring the predicate rather than a list that happened not to change.
        history = NavigationHistory([1, 2, 3])
        assert history.commit([3, 2, 1]) is True


class TestStatesThatGoBad:
    """A state can be pushed and then stop being somewhere one can return to."""

    def _with_a_dead_middle(self):
        """a, b, c, d — where b and c have since died. The cursor is on d."""
        dead = {"b", "c"}
        history = NavigationHistory("a", is_valid=lambda state: state not in dead)
        for state in ("b", "c", "d"):
            history.commit(state)
        return history, dead

    def test_back_steps_over_what_is_gone(self):
        # Stopping at a dead entry would make Back appear to do nothing, which is worse than going one
        # step further: a history step is a request to be somewhere you have been, and a state that
        # cannot be returned to does not satisfy it.
        history, _dead = self._with_a_dead_middle()
        assert history.back() is True
        assert history.current == "a"

    def test_forward_skips_exactly_what_back_skipped(self):
        # Or the pair stops being inverse, and a reader cannot get back to where they pressed Back from.
        history, _dead = self._with_a_dead_middle()
        history.back()
        assert history.forward() is True
        assert history.current == "d", "Forward did not return to where Back was pressed"

    def test_a_history_with_nothing_live_behind_it_cannot_go_back(self):
        # And so the button is disabled, rather than offering a press that goes nowhere.
        history = NavigationHistory("a", is_valid=lambda state: state == "b")
        history.commit("b")
        assert history.current == "b"
        assert not history.can_go_back, "there is an entry behind, but it is dead and cannot be returned to"
        assert history.back() is False

    def test_the_cursor_stays_put_when_back_finds_nothing(self):
        history = NavigationHistory("a", is_valid=lambda state: state == "b")
        history.commit("b")
        history.back()
        assert history.current == "b"

    def test_validity_is_asked_again_on_every_traversal(self):
        # Pruning once would lose a state that comes back — a remount, a recreated directory. Asking each
        # time costs a predicate call on a list this size and keeps the history honest about the present.
        alive = {"a", "b"}
        history = NavigationHistory("a", is_valid=lambda state: state in alive)
        history.commit("b")
        history.commit("c")
        alive.discard("b")
        history.back()
        assert history.current == "a", "b is dead, so back should have stepped over it"
        alive.add("b")
        history.forward()
        assert history.current == "b", "b came back and the history had already forgotten it"


class TestTheCallerHasTheLastWord:
    def test_a_refused_arrival_leaves_the_cursor_alone(self):
        # A state can pass `is_valid` and still fail to be entered — a directory that exists but cannot
        # be read. A cursor that moved anyway would disagree with where the view is, which makes every
        # later step wrong rather than only this one.
        history = NavigationHistory("a")
        history.commit("b")
        assert history.back(apply=lambda state: False) is False
        assert history.current == "b"

    def test_an_accepted_arrival_moves_it(self):
        # The control: an `apply` that always refused would satisfy the test above and mean nothing.
        history = NavigationHistory("a")
        history.commit("b")
        seen = []
        assert history.back(apply=lambda state: seen.append(state) or True) is True
        assert history.current == "a"
        assert seen == ["a"], "the callback should be handed the state it is being asked about"

    def test_the_callback_is_not_called_when_there_is_nowhere_to_go(self):
        history = NavigationHistory("a")
        calls = []
        assert history.back(apply=lambda state: calls.append(state) or True) is False
        assert calls == []
