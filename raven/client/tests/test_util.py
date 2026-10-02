"""Unit tests for raven.client.util: `post_streamed_job` against `raven.server.util.stream_job`, served locally by waitress."""

import socket
import threading
import time

import pytest
import requests

pytest.importorskip("flask", reason="Raven-server's web stack (not in the CI minimal dep subset)")
pytest.importorskip("waitress", reason="Raven-server's web stack (not in the CI minimal dep subset)")

from flask import Flask  # noqa: E402 -- after the guards, like the imports that need them
from waitress.server import create_server  # noqa: E402

from raven.client import util  # noqa: E402
from raven.client.config import Timeout  # noqa: E402
from raven.common import netutil  # noqa: E402
from raven.server import util as serverutil  # noqa: E402

TIMEOUT = Timeout(connect=5.0, read=30.0)


@pytest.fixture
def job_server():
    """Serve one `stream_job` endpoint under waitress, as Raven-server does; yields a function taking the job and returning its URL."""
    servers = []

    def serve(job) -> str:
        app = Flask(__name__)
        app.add_url_rule("/job", "job", lambda: serverutil.stream_job(job), methods=["POST"])
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        server = create_server(app, host="127.0.0.1", port=port, channel_request_lookahead=1)
        threading.Thread(target=server.run, daemon=True).start()
        servers.append(server)
        return f"http://127.0.0.1:{port}/job"

    yield serve
    for server in reversed(servers):
        server.close()


class TestPostStreamedJob:
    def test_returns_the_jobs_result(self, job_server):
        url = job_server(lambda is_cancelled, report: {"answer": 42})
        assert util.post_streamed_job(url, {}, timeout=TIMEOUT) == {"answer": 42}

    def test_progress_arrives_while_the_job_runs(self, job_server):
        # The job waits for the client to have seen its first report. Progress held back until the end -- by a
        # buffer anywhere between the two -- would leave it waiting out its timeout.
        seen_by_client = threading.Event()
        def job(is_cancelled, report):
            report("Working…")
            return {"streamed": seen_by_client.wait(timeout=5.0)}
        url = job_server(job)
        progress = []
        def on_progress(text):
            progress.append(text)
            seen_by_client.set()
        result = util.post_streamed_job(url, {}, timeout=TIMEOUT, on_progress=on_progress)
        assert progress == ["Working…"]
        assert result == {"streamed": True}, "the progress reached the client only after the job had ended"

    def test_the_read_timeout_bounds_the_whole_job(self, job_server):
        # Progress every 0.1 s keeps resetting a between-bytes timeout of 0.5 s, so only a total deadline can
        # end this job before its 5 s are up.
        def job(is_cancelled, report):
            t0 = time.monotonic()
            while time.monotonic() - t0 < 5.0 and not is_cancelled():
                report("Still working…")
                time.sleep(0.1)
            return {"finished": True}
        url = job_server(job)
        progress = []
        t0 = time.monotonic()
        with pytest.raises(requests.Timeout):
            util.post_streamed_job(url, {}, timeout=Timeout(connect=5.0, read=0.5), on_progress=progress.append)
        assert len(progress) >= 2, "the job sent no progress, so this fixture cannot tell a deadline from a read timeout"
        assert time.monotonic() - t0 < 2.0, "the call outlasted its deadline"

    def test_a_failed_job_raises(self, job_server):
        def job(is_cancelled, report):
            raise ValueError("boom")
        url = job_server(job)
        with pytest.raises(RuntimeError, match="ValueError: boom"):
            util.post_streamed_job(url, {}, timeout=TIMEOUT)

    def test_an_abort_ends_the_call_and_the_job(self, job_server):
        seen = {}
        def job(is_cancelled, report):
            t0 = time.monotonic()
            while time.monotonic() - t0 < 10.0:
                if is_cancelled():
                    seen["cancelled"] = True
                    return {}
                time.sleep(0.05)
            return {"finished": True}
        url = job_server(job)
        abort = netutil.Abort()
        threading.Timer(0.3, abort.abort).start()

        t0 = time.monotonic()
        with pytest.raises(netutil.Aborted):
            util.post_streamed_job(url, {}, timeout=TIMEOUT, maybe_abort=abort)
        assert time.monotonic() - t0 < 2.0, "the call waited for the job"

        deadline = time.monotonic() + 3.0
        while "cancelled" not in seen and time.monotonic() < deadline:
            time.sleep(0.05)
        assert seen == {"cancelled": True}, "the server went on working for a client that had gone"

    def test_an_abort_before_the_call_sends_nothing(self, job_server):
        called = []
        url = job_server(lambda is_cancelled, report: called.append(1) or {})
        abort = netutil.Abort()
        abort.abort()
        with pytest.raises(netutil.Aborted):
            util.post_streamed_job(url, {}, timeout=TIMEOUT, maybe_abort=abort)
        assert called == []
