"""Unit tests for raven.server.util. `stream_job`: the streamed body's records, and a client abandoning the job under waitress."""

import json
import socket
import threading
import time

import pytest

pytest.importorskip("flask", reason="Raven-server's web stack (not in the CI minimal dep subset)")
pytest.importorskip("waitress", reason="Raven-server's web stack (not in the CI minimal dep subset)")

import requests  # noqa: E402 -- after the guards, like the imports that need them
from flask import Flask  # noqa: E402
from waitress.server import create_server  # noqa: E402

from raven.common import netutil  # noqa: E402
from raven.server import util as serverutil  # noqa: E402


def _app_serving(job) -> Flask:
    app = Flask(__name__)
    app.add_url_rule("/job", "job", lambda: serverutil.stream_job(job), methods=["POST"])
    return app


@pytest.fixture
def waitress_server():
    """Serve a Flask app under waitress, configured as Raven-server is; yields a function taking the app and returning its URL."""
    servers = []

    def serve(app: Flask) -> str:
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


def _records(body: bytes) -> list[dict]:
    """The JSON records in a `stream_job` body, in order."""
    return [json.loads(line) for line in body.split(b"\n") if line.strip()]


class TestBody:
    """What the client receives, which does not depend on the server."""

    def test_the_result_follows_a_leading_space(self):
        client = _app_serving(lambda is_cancelled, report: {"answer": 42}).test_client()
        body = client.post("/job").data
        assert body.startswith(b" "), "the space is what sends the headers before the job has run"
        assert _records(body) == [{"result": {"answer": 42}}]

    def test_progress_comes_before_the_result_in_the_order_reported(self):
        def job(is_cancelled, report):
            report("First step…")
            report("Second step…")
            return {"answer": 42}
        body = _app_serving(job).test_client().post("/job").data
        assert _records(body) == [{"progress": "First step…"},
                                  {"progress": "Second step…"},
                                  {"result": {"answer": 42}}]

    def test_a_failing_job_reports_in_the_body(self):
        def job(is_cancelled, report):
            report("Starting…")
            raise ValueError("boom")
        response = _app_serving(job).test_client().post("/job")
        assert response.status_code == 200, "the status has gone out before the job runs"
        assert _records(response.data) == [{"progress": "Starting…"}, {"error": "ValueError: boom"}]

    def test_outside_waitress_nothing_is_ever_cancelled(self):
        seen = []
        _app_serving(lambda is_cancelled, report: seen.append(is_cancelled()) or {}).test_client().post("/job").data  # noqa: B018 -- the job runs as the body is read
        assert seen == [False]


class TestAbandoning:
    """Under waitress, a client closing its connection is what the job's `is_cancelled` reports."""

    @staticmethod
    def _slow_job(seen: dict):
        def job(is_cancelled, report):
            t0 = time.monotonic()
            while time.monotonic() - t0 < 10.0:
                if is_cancelled():
                    seen["cancelled_after"] = time.monotonic() - t0
                    return {"stopped": True}
                time.sleep(0.05)
            return {"finished": True}
        return job

    def test_the_job_sees_the_client_go(self, waitress_server):
        seen = {}
        url = waitress_server(_app_serving(self._slow_job(seen)))
        abort = netutil.Abort()

        t0 = time.monotonic()
        response = requests.post(url, stream=True, timeout=(5, 30))
        assert time.monotonic() - t0 < 1.0, "the headers waited for the job"
        abort.arm(response)
        threading.Timer(0.3, abort.abort).start()
        with pytest.raises(requests.RequestException):
            response.content  # noqa: B018 -- reading is the point; the abort ends it
        abort.disarm()

        deadline = time.monotonic() + 3.0
        while "cancelled_after" not in seen and time.monotonic() < deadline:
            time.sleep(0.05)
        assert "cancelled_after" in seen, "the job never saw the client go"
        assert seen["cancelled_after"] < 2.0

    def test_a_client_that_stays_gets_the_result(self, waitress_server):
        # The control for the one above: the same job and server, and nobody leaves.
        seen = {}
        def quick_job(is_cancelled, report):
            seen["cancelled"] = is_cancelled()
            return {"finished": True}
        url = waitress_server(_app_serving(quick_job))
        response = requests.post(url, timeout=(5, 30))
        assert _records(response.content) == [{"result": {"finished": True}}]
        assert seen == {"cancelled": False}
