"""Tests for the GUI's non-widget logic.

Tk windows need a real window-server session, so nothing here constructs one.
What is testable without a display is everything that actually broke: how the
app locates its resources, how it builds the argv for background work, and how
the worker reports output and failure.
"""

from __future__ import annotations

import queue
import sys

import pytest

from gui import paths
from gui.assistant import KoviAI
from gui.worker import DONE, ERROR, OUTPUT, ProcessWorker

# ------------------------------------------------------------------------ paths


def test_source_workspace_is_the_repo_root():
    assert (paths.workspace() / "llms").is_dir()
    assert (paths.workspace() / "configs").is_dir()


def test_bundled_configs_resolve_from_source():
    assert paths.bundled("configs", "tiny.yaml").exists()


def test_child_command_is_a_list_so_quoting_cannot_break():
    """The old code built a string and split it, which tore prompts apart."""
    cmd = paths.child_command("sample", ["--prompt", "two words"])
    assert isinstance(cmd, list)
    assert cmd[-2:] == ["--prompt", "two words"]


def test_child_command_from_source_invokes_the_launcher():
    cmd = paths.child_command("train", ["--config", "configs/tiny.yaml"])
    assert cmd[0] == sys.executable
    assert cmd[1].endswith("llms_studio.py")
    assert cmd[2] == "train"


def test_frozen_child_command_omits_the_launcher_path(monkeypatch):
    """Frozen, sys.executable IS the app, so it takes the subcommand directly."""
    monkeypatch.setattr(paths, "is_frozen", lambda: True)
    cmd = paths.child_command("train", ["--config", "x.yaml"])
    assert cmd == [sys.executable, "train", "--config", "x.yaml"]


def test_frozen_workspace_is_under_the_home_directory(monkeypatch):
    monkeypatch.setattr(paths, "is_frozen", lambda: True)
    assert paths.workspace().parent == paths.Path.home()


# ----------------------------------------------------------------------- worker


def _drain(q: queue.Queue) -> list:
    out = []
    while not q.empty():
        out.append(q.get())
    return out


def test_worker_streams_output_then_reports_exit_code():
    q: queue.Queue = queue.Queue()
    worker = ProcessWorker([sys.executable, "-c", "print('a'); print('b')"], q)
    worker.start()
    worker.join(30)

    messages = _drain(q)
    assert [m for m in messages if m[0] == OUTPUT] == [(OUTPUT, "a\n"), (OUTPUT, "b\n")]
    assert messages[-1] == (DONE, 0)


def test_worker_reports_a_nonzero_exit():
    q: queue.Queue = queue.Queue()
    worker = ProcessWorker([sys.executable, "-c", "raise SystemExit(3)"], q)
    worker.start()
    worker.join(30)
    assert (DONE, 3) in _drain(q)


def test_worker_reports_a_missing_executable_instead_of_raising():
    q: queue.Queue = queue.Queue()
    worker = ProcessWorker(["/nonexistent/binary"], q)
    worker.start()
    worker.join(30)

    messages = _drain(q)
    assert messages[0][0] == ERROR
    assert "could not start process" in messages[0][1]
    assert messages[-1][0] == DONE


def test_worker_captures_stderr_too():
    """stderr is merged into stdout so the log shows errors in order."""
    q: queue.Queue = queue.Queue()
    worker = ProcessWorker(
        [sys.executable, "-c", "import sys; sys.stderr.write('boom\\n')"], q
    )
    worker.start()
    worker.join(30)
    assert (OUTPUT, "boom\n") in _drain(q)


def test_worker_stop_terminates_a_long_running_child():
    q: queue.Queue = queue.Queue()
    worker = ProcessWorker([sys.executable, "-c", "import time; time.sleep(60)"], q)
    worker.start()
    for _ in range(100):  # wait for Popen to actually exist
        if worker.process is not None:
            break
        import time

        time.sleep(0.05)
    worker.stop()
    worker.join(30)
    assert not worker.is_alive()


def test_worker_is_a_daemon_so_it_cannot_outlive_the_app():
    q: queue.Queue = queue.Queue()
    assert ProcessWorker([sys.executable, "-c", "pass"], q).daemon


# -------------------------------------------------------------------- assistant


@pytest.mark.parametrize(
    "prompt,expected_key",
    [
        ("make it smaller and faster", "model.n_layer"),
        ("I want a smarter, larger model", "model.n_layer"),
        ("use a high learning rate", "train.lr"),
        ("something unrelated entirely", "train.lr"),
    ],
)
def test_assistant_always_returns_usable_overrides(prompt, expected_key):
    response, overrides = KoviAI().customize(prompt, {})
    assert response
    assert expected_key in overrides


def test_assistant_overrides_are_valid_config_keys():
    """Every override must be a real 'section.key' the trainer accepts."""
    from llms.config import Config

    _, overrides = KoviAI().customize("make it smaller", {})
    for dotted in overrides:
        section, _, key = dotted.partition(".")
        assert hasattr(Config(), section), f"no such config section: {section}"
        assert hasattr(getattr(Config(), section), key), f"no such key: {dotted}"


# ------------------------------------------------------------------- web server

# The UI is served over HTTP because Apple's system Tk 8.5 cannot open a window
# on macOS 26+. These cover the endpoints the page actually calls.


@pytest.fixture
def client():
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    from gui import server

    return fastapi_testclient.TestClient(server.app)


def test_index_serves_the_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "LLMS Studio" in response.text
    assert "/api/stream" in response.text  # the live-log wiring is present


def test_state_lists_configs_and_workspace(client):
    state = client.get("/api/state").json()
    assert "tiny.yaml" in state["configs"]
    assert state["workspace"]
    assert state["busy"] is False


def test_customize_returns_overrides_and_remembers_them(client):
    body = client.post("/api/customize", json={"prompt": "make it smaller"}).json()
    assert body["overrides"]
    assert client.get("/api/state").json()["overrides"] == body["overrides"]


def test_train_rejects_an_unknown_config(client):
    body = client.post("/api/train", json={"config": "does-not-exist.yaml"}).json()
    assert body["ok"] is False
    assert "no such config" in body["error"]


def test_train_cannot_escape_the_configs_directory(client):
    """A path like ../../etc/passwd must not resolve outside configs/."""
    body = client.post("/api/train", json={"config": "../../../etc/passwd"}).json()
    assert body["ok"] is False


def test_sample_rejects_a_missing_checkpoint(client):
    body = client.post("/api/sample", json={"checkpoint": "out/nope/ckpt.pt"}).json()
    assert body["ok"] is False
    assert "no such checkpoint" in body["error"]


def test_stop_is_safe_when_nothing_is_running(client):
    assert client.post("/api/stop").json()["ok"] is True


def test_find_free_port_returns_a_bindable_port():
    import socket

    from gui.server import HOST, find_free_port

    port = find_free_port()
    with socket.socket() as probe:
        probe.bind((HOST, port))  # must not raise


def test_log_buffer_is_bounded():
    """A long training run must not grow the in-memory log without bound."""
    from gui.server import MAX_LOG_LINES, Session

    s = Session()
    for i in range(MAX_LOG_LINES + 500):
        s.append(f"line {i}\n")
    assert len(s.log) == MAX_LOG_LINES
    assert s.log[-1] == f"line {MAX_LOG_LINES + 499}\n"  # the tail is what is kept
