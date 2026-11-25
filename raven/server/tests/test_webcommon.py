"""Unit tests for raven.server.modules.webcommon: the lock a web job waits for, and the exceptions its jobs end with."""

import subprocess
import threading

import pytest

webcommon = pytest.importorskip("raven.server.modules.webcommon",
                                reason="webcommon needs colorama (not in the CI minimal dep subset)")


class TestLockUnlessCancelled:
    def test_holds_the_lock_for_the_body_and_releases_it(self):
        lock = threading.Lock()
        with webcommon.lock_unless_cancelled(lock, lambda: False):
            assert lock.locked()
        assert not lock.locked()

    def test_releases_the_lock_when_the_body_raises(self):
        lock = threading.Lock()
        with pytest.raises(ValueError):
            with webcommon.lock_unless_cancelled(lock, lambda: False):
                raise ValueError("from the body")
        assert not lock.locked()

    def test_a_cancelled_wait_raises_without_entering_the_body(self):
        # On a thread of its own with a bounded join, so that a wait that never asks fails this test rather
        # than hanging it.
        lock = threading.Lock()
        cancel = threading.Event()
        outcome = {}
        def waiter():
            try:
                with webcommon.lock_unless_cancelled(lock, cancel.is_set):
                    outcome["entered"] = True
            except webcommon.Cancelled:
                outcome["cancelled"] = True
        with lock:  # held elsewhere, so the waiter has to wait
            worker = threading.Thread(target=waiter, daemon=True)
            worker.start()
            cancel.set()
            worker.join(timeout=2.0)
            assert not worker.is_alive(), "the wait did not give up when cancelled"
        assert outcome == {"cancelled": True}

    def test_a_wait_that_is_not_cancelled_gets_the_lock(self):
        # The control for the one above: the same wait, and the lock comes free instead.
        lock = threading.Lock()
        outcome = {}
        def waiter():
            with webcommon.lock_unless_cancelled(lock, lambda: False):
                outcome["entered"] = True
        lock.acquire()
        worker = threading.Thread(target=waiter, daemon=True)
        worker.start()
        lock.release()
        worker.join(timeout=2.0)
        assert outcome == {"entered": True}


class TestExceptions:
    def test_cancelled_is_a_web_tool_exception(self):
        assert issubclass(webcommon.Cancelled, webcommon.WebToolException)


class TestQuitDriver:
    """Quitting a driver whose process may already be gone, as at exit after a Ctrl+C."""

    class FakeProcess:
        def __init__(self, returncode):
            self.returncode = returncode  # `None` while running, as `subprocess.Popen` has it

        def wait(self, timeout=None):
            if self.returncode is None:
                raise subprocess.TimeoutExpired(cmd="chromedriver", timeout=timeout)
            return self.returncode

    class FakeDriver:
        def __init__(self, process, quit_raises=None):
            self.service = type("Service", (), {"process": process})()
            self.quit_raises = quit_raises
            self.quit_called = False

        def quit(self):
            self.quit_called = True
            if self.quit_raises is not None:
                raise self.quit_raises

    def test_a_driver_whose_process_has_exited_is_not_asked_to_quit(self):
        driver = self.FakeDriver(self.FakeProcess(returncode=-2))
        webcommon.quit_driver(driver)
        assert not driver.quit_called

    def test_a_running_driver_is_quit(self):
        # The control for the one above: only the process's state differs.
        driver = self.FakeDriver(self.FakeProcess(returncode=None))
        webcommon.quit_driver(driver)
        assert driver.quit_called

    def test_a_failure_to_quit_is_logged_rather_than_raised(self, caplog):
        driver = self.FakeDriver(self.FakeProcess(returncode=None), quit_raises=ConnectionError("refused"))
        with caplog.at_level("INFO", logger=webcommon.logger.name):
            webcommon.quit_driver(driver)
        assert driver.quit_called
        assert "did not quit cleanly" in caplog.text

    def test_a_driver_without_a_process_to_ask_about_is_quit(self):
        driver = self.FakeDriver(process=None)
        webcommon.quit_driver(driver)
        assert driver.quit_called
