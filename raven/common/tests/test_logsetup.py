"""Unit tests for raven.common.logsetup.

Tests verify that `configure(...)` correctly installs handlers, applies the
allowlist filter, mirrors to a logfile, and survives re-entry. The tests
mutate the global `logging.root` configuration; each test restores the
original state via the `restore_logging` fixture.
"""

import logging

import pytest

from raven.common import logsetup


@pytest.fixture
def restore_logging():
    """Snapshot root logger config; restore after the test.

    `configure` mutates `logging.root` (level, handlers). Without this
    fixture, tests would leak configuration into one another and into the
    rest of the pytest run.
    """
    saved_handlers = list(logging.root.handlers)
    saved_level = logging.root.level
    yield
    for handler in list(logging.root.handlers):
        logging.root.removeHandler(handler)
        handler.close()
    for handler in saved_handlers:
        logging.root.addHandler(handler)
    logging.root.setLevel(saved_level)


# ---------------------------------------------------------------------------
# Basic configuration
# ---------------------------------------------------------------------------

class TestBasicConfigure:
    def test_installs_stderr_handler(self, restore_logging):
        logsetup.configure(level=logging.INFO)
        assert any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
                   for h in logging.root.handlers)

    def test_sets_level(self, restore_logging):
        logsetup.configure(level=logging.DEBUG)
        assert logging.root.level == logging.DEBUG

    def test_no_logfile_means_only_stderr(self, restore_logging):
        logsetup.configure(level=logging.INFO)
        file_handlers = [h for h in logging.root.handlers if isinstance(h, logging.FileHandler)]
        assert file_handlers == []

    def test_default_format_is_timestamped(self, restore_logging, capsys):
        logsetup.configure(level=logging.INFO)
        logging.getLogger("raven.test").info("hello")
        captured = capsys.readouterr().err
        # Format is "YYYY-MM-DD HH:MM:SS,mmm LEVEL name: message"
        assert "INFO" in captured
        assert "raven.test" in captured
        assert "hello" in captured
        # Asctime: at least four digits (the year) followed by a hyphen.
        assert any(ch.isdigit() for ch in captured.split()[0])

    def test_custom_format(self, restore_logging, capsys):
        logsetup.configure(level=logging.INFO, fmt="CUSTOM %(message)s")
        logging.getLogger("raven.test").info("hi")
        assert "CUSTOM hi" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Allowlist filter
# ---------------------------------------------------------------------------

class TestAllowlist:
    def test_none_means_emit_everything(self, restore_logging, capsys):
        logsetup.configure(level=logging.INFO, allow=None)
        logging.getLogger("raven.test").info("ours")
        logging.getLogger("third_party.lib").info("theirs")
        captured = capsys.readouterr().err
        assert "ours" in captured
        assert "theirs" in captured

    def test_allow_suppresses_non_matching(self, restore_logging, capsys):
        logsetup.configure(level=logging.INFO, allow=["raven"])
        logging.getLogger("raven.test").info("ours")
        logging.getLogger("third_party.lib").info("theirs")
        captured = capsys.readouterr().err
        assert "ours" in captured
        assert "theirs" not in captured

    def test_allow_is_prefix_match(self, restore_logging, capsys):
        # Synthetic logger names — see test_multiple_prefixes_or_combined for why.
        logsetup.configure(level=logging.INFO, allow=["pkg_one.sub"])
        logging.getLogger("pkg_one.sub.deep").info("INSIDE")
        logging.getLogger("pkg_one.elsewhere").info("OUTSIDE")
        captured = capsys.readouterr().err
        assert "INSIDE" in captured
        assert "OUTSIDE" not in captured

    def test_multiple_prefixes_or_combined(self, restore_logging, capsys):
        # Use synthetic logger names so module-import side effects (e.g.
        # `raven.common.bgtask` pins its own level to WARNING) can't bias
        # the test when the suite runs in a different order.
        logsetup.configure(level=logging.INFO, allow=["pkg_one", "pkg_two"])
        logging.getLogger("pkg_one.alpha").info("ALPHA")
        logging.getLogger("pkg_two.beta").info("BETA")
        logging.getLogger("pkg_three.gamma").info("GAMMA")
        captured = capsys.readouterr().err
        assert "ALPHA" in captured
        assert "BETA" in captured
        assert "pkg_three" not in captured
        assert "GAMMA" not in captured

    def test_filter_applied_to_every_handler(self, restore_logging, tmp_path):
        logfile = tmp_path / "log.txt"
        logsetup.configure(level=logging.INFO, logfile=str(logfile), allow=["raven"])
        # Both handlers should carry the filter.
        for handler in logging.root.handlers:
            assert handler.filters, f"handler {handler!r} has no filters"


# ---------------------------------------------------------------------------
# Logfile mirroring
# ---------------------------------------------------------------------------

class TestLogfile:
    def test_writes_to_file(self, restore_logging, tmp_path):
        logfile = tmp_path / "session.log"
        logsetup.configure(level=logging.INFO, logfile=str(logfile))
        logging.getLogger("raven.test").info("written")
        # Flush handlers so the FileHandler buffer reaches disk.
        for handler in logging.root.handlers:
            handler.flush()
        assert "written" in logfile.read_text(encoding="utf-8")

    def test_overwrites_existing_file(self, restore_logging, tmp_path):
        logfile = tmp_path / "session.log"
        logfile.write_text("old contents that should be gone\n", encoding="utf-8")
        logsetup.configure(level=logging.INFO, logfile=str(logfile))
        logging.getLogger("raven.test").info("fresh")
        for handler in logging.root.handlers:
            handler.flush()
        contents = logfile.read_text(encoding="utf-8")
        assert "old contents" not in contents
        assert "fresh" in contents

    def test_file_and_stderr_both_get_record(self, restore_logging, tmp_path, capsys):
        logfile = tmp_path / "mirror.log"
        logsetup.configure(level=logging.INFO, logfile=str(logfile))
        logging.getLogger("raven.test").info("twice")
        for handler in logging.root.handlers:
            handler.flush()
        assert "twice" in capsys.readouterr().err
        assert "twice" in logfile.read_text(encoding="utf-8")

    def test_allow_applies_to_file_too(self, restore_logging, tmp_path):
        logfile = tmp_path / "filtered.log"
        logsetup.configure(level=logging.INFO, logfile=str(logfile), allow=["raven"])
        logging.getLogger("raven.test").info("in")
        logging.getLogger("third_party.lib").info("out")
        for handler in logging.root.handlers:
            handler.flush()
        contents = logfile.read_text(encoding="utf-8")
        assert "in" in contents
        assert "out" not in contents


# ---------------------------------------------------------------------------
# Re-entry / idempotency
# ---------------------------------------------------------------------------

class TestReentry:
    def test_second_call_replaces_first(self, restore_logging):
        logsetup.configure(level=logging.INFO)
        first_handlers = list(logging.root.handlers)
        logsetup.configure(level=logging.DEBUG)
        second_handlers = list(logging.root.handlers)
        # `force=True` should have torn down the first set entirely.
        assert not any(h in second_handlers for h in first_handlers)
        assert logging.root.level == logging.DEBUG

    def test_reentry_does_not_double_emit(self, restore_logging, capsys):
        logsetup.configure(level=logging.INFO)
        logsetup.configure(level=logging.INFO)
        logging.getLogger("raven.test").info("once")
        captured = capsys.readouterr().err
        # Exactly one stderr line for the record.
        assert captured.count("once") == 1

    def test_reentry_swaps_logfile(self, restore_logging, tmp_path):
        first = tmp_path / "first.log"
        second = tmp_path / "second.log"
        logsetup.configure(level=logging.INFO, logfile=str(first))
        logging.getLogger("raven.test").info("to-first")
        for handler in logging.root.handlers:
            handler.flush()
        logsetup.configure(level=logging.INFO, logfile=str(second))
        logging.getLogger("raven.test").info("to-second")
        for handler in logging.root.handlers:
            handler.flush()
        assert "to-first" in first.read_text(encoding="utf-8")
        assert "to-second" not in first.read_text(encoding="utf-8")
        assert "to-second" in second.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Self-healing logfile (recovers from flair-style logging.shutdown)
# ---------------------------------------------------------------------------

class TestSelfHealing:
    def test_filehandler_survives_logging_shutdown(self, restore_logging, tmp_path):
        """Simulates `flair.__init__` calling `logging.config.dictConfig` —
        which invokes `logging.shutdown()` on every handler in
        `logging._handlerList`, closing their streams while leaving the
        handlers in `root.handlers`. With `mode='a'`, `FileHandler.emit`
        reopens the file on the next write; with `mode='w'`, it would not.
        """
        import logging.config
        logfile = tmp_path / "session.log"
        logsetup.configure(level=logging.INFO, logfile=str(logfile))
        logging.getLogger("raven.test").info("before-clobber")

        # Reproduce flair's exact pattern: dictConfig with no `root` key,
        # `disable_existing_loggers=False`. This closes all existing handlers
        # via `logging.shutdown()` but leaves them in `root.handlers`.
        logging.config.dictConfig({
            "version": 1,
            "disable_existing_loggers": False,
            "loggers": {"hostile_lib": {"level": "INFO"}},
        })
        logging.getLogger("raven.test").info("after-clobber")

        for handler in logging.root.handlers:
            handler.flush()

        contents = logfile.read_text(encoding="utf-8")
        assert "before-clobber" in contents
        assert "after-clobber" in contents

    def test_truncate_at_configure_replaces_old_file(self, restore_logging, tmp_path):
        # Belt-and-suspenders for the truncate-then-append trick: prior
        # session contents must not survive a fresh configure().
        logfile = tmp_path / "session.log"
        logfile.write_text("PREVIOUS SESSION\n", encoding="utf-8")
        logsetup.configure(level=logging.INFO, logfile=str(logfile))
        logging.getLogger("raven.test").info("new-session-record")
        for handler in logging.root.handlers:
            handler.flush()
        contents = logfile.read_text(encoding="utf-8")
        assert "PREVIOUS SESSION" not in contents
        assert "new-session-record" in contents


# ---------------------------------------------------------------------------
# Log level dispatch
# ---------------------------------------------------------------------------

class TestLevelDispatch:
    def test_below_level_suppressed(self, restore_logging, capsys):
        logsetup.configure(level=logging.WARNING)
        logging.getLogger("raven.test").info("info-record")
        logging.getLogger("raven.test").warning("warn-record")
        captured = capsys.readouterr().err
        assert "info-record" not in captured
        assert "warn-record" in captured

    def test_debug_level_lets_debug_through(self, restore_logging, capsys):
        logsetup.configure(level=logging.DEBUG)
        logging.getLogger("raven.test").debug("dbg")
        assert "dbg" in capsys.readouterr().err
