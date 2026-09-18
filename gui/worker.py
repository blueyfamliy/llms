"""Running a child process without blocking or touching Tk from a thread.

Output is pushed onto a ``queue.Queue`` rather than handed to a callback
directly: Tk widgets may only be touched from the thread that created them, and
calling ``insert`` from the reader thread is a crash waiting to happen. The GUI
drains the queue from its own event loop instead.
"""

from __future__ import annotations

import queue
import subprocess
import threading
from collections.abc import Sequence

# Queue message kinds.
OUTPUT = "output"
DONE = "done"
ERROR = "error"


class ProcessWorker(threading.Thread):
    """Runs ``command`` and streams its merged stdout/stderr onto ``messages``."""

    def __init__(
        self,
        command: Sequence[str],
        messages: queue.Queue,
        cwd: str | None = None,
    ) -> None:
        super().__init__(daemon=True)  # never keep the app alive on quit
        self.command = list(command)
        self.messages = messages
        self.cwd = cwd
        self.process: subprocess.Popen | None = None
        self._stopping = threading.Event()

    def stop(self) -> None:
        """Ask the child to exit, escalating to a kill if it ignores us."""
        self._stopping.set()
        process = self.process
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()

    def run(self) -> None:
        try:
            self.process = subprocess.Popen(
                self.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # one interleaved stream, in order
                text=True,
                bufsize=1,
                cwd=self.cwd,
            )
        except OSError as exc:
            # A missing executable or unwritable cwd lands here, not in the loop.
            self.messages.put((ERROR, f"could not start process: {exc}"))
            self.messages.put((DONE, -1))
            return

        try:
            assert self.process.stdout is not None
            for line in self.process.stdout:
                self.messages.put((OUTPUT, line))
            returncode = self.process.wait()
        except Exception as exc:  # noqa: BLE001 - surface anything to the log
            self.messages.put((ERROR, f"worker error: {exc}"))
            returncode = -1
        finally:
            if self.process.stdout is not None:
                self.process.stdout.close()

        if self._stopping.is_set():
            self.messages.put((OUTPUT, "\n-- stopped --\n"))
        self.messages.put((DONE, returncode))
