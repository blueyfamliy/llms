"""LLMS Studio as a local web app.

Why a browser and not a desktop toolkit: Apple's system Tcl/Tk is version 8.5,
deprecated for years, and on macOS 26+ a plain ``tkinter.Tk().update()`` hangs
forever -- the window is created but never mapped. Any Tk UI built against the
system Python is dead on arrival there. A local HTTP server has no such
dependency and renders identically everywhere.

The server binds to 127.0.0.1 only. It can start training runs and read files
under the workspace, so it must not be reachable from the network.
"""

from __future__ import annotations

import asyncio
import json
import queue
import socket
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from gui.assistant import KoviAI
from gui.paths import child_command, prepare_workspace
from gui.worker import DONE, ERROR, OUTPUT, ProcessWorker

HOST = "127.0.0.1"
MAX_LOG_LINES = 2000

app = FastAPI(title="LLMS Studio")
assistant = KoviAI()


class Session:
    """The one job this app runs at a time, plus its log."""

    def __init__(self) -> None:
        self.workspace: Path = prepare_workspace()
        self.worker: ProcessWorker | None = None
        self.messages: queue.Queue = queue.Queue()
        self.log: list[str] = []
        self.status: str = "Ready"
        self.overrides: dict[str, Any] = {}
        self.stop_requested: bool = False

    @property
    def busy(self) -> bool:
        return self.worker is not None and self.worker.is_alive()

    def configs(self) -> list[str]:
        return sorted(p.name for p in (self.workspace / "configs").glob("*.yaml"))

    def checkpoints(self) -> list[str]:
        found = sorted(self.workspace.glob("out/*/ckpt.pt"), key=lambda p: -p.stat().st_mtime)
        return [str(p.relative_to(self.workspace)) for p in found]

    def append(self, line: str) -> None:
        self.log.append(line)
        if len(self.log) > MAX_LOG_LINES:
            # Keep the tail; a long training run would otherwise grow without bound.
            del self.log[: len(self.log) - MAX_LOG_LINES]

    def start(self, command: list[str], status: str) -> None:
        self.messages = queue.Queue()
        self.append("$ " + " ".join(command) + "\n")
        self.status = status
        self.stop_requested = False
        self.worker = ProcessWorker(command, self.messages, cwd=str(self.workspace))
        self.worker.start()

    def drain(self) -> list[str]:
        """Move finished worker output into the log. Returns the new lines."""
        new: list[str] = []
        try:
            while True:
                kind, payload = self.messages.get_nowait()
                if kind == OUTPUT:
                    new.append(payload)
                elif kind == ERROR:
                    new.append(f"\n{payload}\n")
                elif kind == DONE:
                    new.append(f"\n-- finished with exit code {payload} --\n\n")
                    # A run the user stopped is not a failure, even though the
                    # child exits on a signal.
                    if self.stop_requested:
                        self.status = "Stopped"
                    else:
                        self.status = "Done" if payload == 0 else f"Failed (exit {payload})"
                    self.worker = None
        except queue.Empty:
            pass
        for line in new:
            self.append(line)
        return new


session = Session()


class CustomizeRequest(BaseModel):
    prompt: str = ""


class TrainRequest(BaseModel):
    config: str


class SampleRequest(BaseModel):
    checkpoint: str
    prompt: str = ""
    max_new_tokens: int = 200
    temperature: float = 0.8
    top_k: int = 50


@app.get("/api/state")
def get_state() -> dict:
    session.drain()
    return {
        "workspace": str(session.workspace),
        "configs": session.configs(),
        "checkpoints": session.checkpoints(),
        "busy": session.busy,
        "status": session.status,
        "overrides": session.overrides,
        "log": "".join(session.log),
    }


@app.post("/api/customize")
def customize(request: CustomizeRequest) -> dict:
    response, overrides = assistant.customize(request.prompt, {})
    session.overrides = overrides
    return {"response": response, "overrides": overrides}


@app.post("/api/train")
def train(request: TrainRequest) -> dict:
    if session.busy:
        return {"ok": False, "error": "a run is already in progress"}
    config = session.workspace / "configs" / Path(request.config).name
    if not config.exists():
        return {"ok": False, "error": f"no such config: {config.name}"}

    args = ["--config", str(config)]
    args += [f"--{key}={value}" for key, value in session.overrides.items()]
    session.start(child_command("train", args), f"Training on {config.name}...")
    return {"ok": True}


@app.post("/api/sample")
def sample(request: SampleRequest) -> dict:
    if session.busy:
        return {"ok": False, "error": "a run is already in progress"}
    ckpt = session.workspace / request.checkpoint
    if not ckpt.exists():
        return {"ok": False, "error": f"no such checkpoint: {request.checkpoint}"}

    args = [
        "--ckpt", str(ckpt),
        "--prompt", request.prompt,
        "--max-new-tokens", str(request.max_new_tokens),
        "--temperature", str(request.temperature),
        "--top-k", str(request.top_k),
    ]
    session.start(child_command("sample", args), f"Sampling from {request.checkpoint}...")
    return {"ok": True}


@app.post("/api/stop")
def stop() -> dict:
    if session.worker is not None:
        session.stop_requested = True
        session.worker.stop()
        session.status = "Stopping..."
    return {"ok": True}


@app.get("/api/stream")
async def stream() -> StreamingResponse:
    """Server-sent events carrying new log lines and status changes."""

    async def events():
        while True:
            new = session.drain()
            if new:
                payload = {"lines": "".join(new), "status": session.status, "busy": session.busy}
                yield f"data: {json.dumps(payload)}\n\n"
            else:
                # A heartbeat keeps proxies and the browser from closing an idle stream.
                yield f"data: {json.dumps({'status': session.status, 'busy': session.busy})}\n\n"
            await asyncio.sleep(0.4)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (Path(__file__).parent / "index.html").read_text(encoding="utf-8")


def find_free_port(preferred: int = 8731) -> int:
    """Use the preferred port when it is free, else let the OS choose one."""
    with socket.socket() as probe:
        try:
            probe.bind((HOST, preferred))
            return preferred
        except OSError:
            pass
    with socket.socket() as probe:
        probe.bind((HOST, 0))
        return probe.getsockname()[1]
