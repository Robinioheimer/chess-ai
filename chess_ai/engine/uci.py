"""Minimal, robust UCI client implemented directly on top of subprocess.

Using the raw UCI protocol (rather than a higher level wrapper) keeps the
communication transparent and testable: every command and every parsed line
is explicit, and the same code drives real Stockfish and the fake UCI engine
used in the test-suite.
"""
from __future__ import annotations

import os
import queue
import shutil
import subprocess
import threading
from typing import Callable, Optional

from ..logging_setup import get_logger
from .types import UciInfo

LOGGER = get_logger(__name__)

_INFO_INT_KEYS = {
    "depth": "depth",
    "seldepth": "seldepth",
    "multipv": "multipv",
    "nodes": "nodes",
    "nps": "nps",
    "time": "time_ms",
}


def parse_info_line(line: str) -> Optional[UciInfo]:
    """Parse a UCI ``info`` line into :class:`UciInfo`.

    Returns ``None`` for non-info lines or ``info string`` chatter.
    """
    if not line.startswith("info"):
        return None
    tokens = line.split()
    info = UciInfo()
    i = 1
    while i < len(tokens):
        token = tokens[i]
        if token == "string":
            info.is_string = True
            return info  # free-form engine message, no structured payload
        if token in _INFO_INT_KEYS and i + 1 < len(tokens):
            try:
                setattr(info, _INFO_INT_KEYS[token], int(tokens[i + 1]))
            except ValueError:
                pass
            i += 2
            continue
        if token == "score" and i + 2 < len(tokens):
            kind = tokens[i + 1]
            try:
                value = int(tokens[i + 2])
            except ValueError:
                i += 3
                continue
            if kind == "cp":
                info.score_cp = value
            elif kind == "mate":
                info.score_mate = value
            i += 3
            # Optional bound flag.
            if i < len(tokens) and tokens[i] in ("lowerbound", "upperbound"):
                info.bound = tokens[i]
                i += 1
            continue
        if token == "pv":
            info.pv = tokens[i + 1:]
            break
        i += 1
    return info


def parse_bestmove_line(line: str) -> Optional[str]:
    """Extract the move from a ``bestmove <move> [ponder <move>]`` line."""
    if not line.startswith("bestmove"):
        return None
    parts = line.split()
    if len(parts) < 2 or parts[1] in ("(none)", ""):
        return None
    return parts[1]


class UciEngineError(RuntimeError):
    """Raised when the UCI process cannot be started or misbehaves."""


class UciEngine:
    """Long-lived UCI engine process.

    The process is started once and reused for every analysis request, which
    avoids expensive hash/table reinitialisation. A background reader thread
    feeds stdout lines into a queue so searches can be interrupted cleanly.
    """

    def __init__(self, executable: str, startup_timeout: float = 15.0) -> None:
        self.executable = executable
        self.startup_timeout = startup_timeout
        self._process: Optional[subprocess.Popen] = None
        self._reader: Optional[threading.Thread] = None
        self._lines: "queue.Queue[str]" = queue.Queue()
        self._lock = threading.RLock()
        self.id_name: str = ""
        self.id_author: str = ""
        self.options: dict[str, str] = {}
        self.supported_options: dict[str, dict] = {}
        self._searching: bool = False

    # ------------------------------------------------------------ lifecycle
    def start(self) -> None:
        with self._lock:
            if self._process is not None:
                return
            if not self.executable:
                raise UciEngineError("No engine executable configured.")
            # Interpreter scripts (e.g. "python fake_engine.py") are validated
            # by checking the script portion exists.
            if self.executable.endswith(".py"):
                script = self._command()[-1]
                if not os.path.isfile(script):
                    raise UciEngineError(f"Engine script not found: {script!r}")
            elif "--uci-engine" in self.executable:
                exe_part = self._command()[0]
                if not shutil.which(exe_part) and not os.path.isfile(exe_part):
                    raise UciEngineError(f"Engine executable not found: {exe_part!r}")
            elif not shutil.which(self.executable) and not os.path.isfile(self.executable):
                raise UciEngineError(f"Engine executable not found: {self.executable!r}")
            kwargs = {
                "stdin": subprocess.PIPE,
                "stdout": subprocess.PIPE,
                "stderr": subprocess.STDOUT,
                "text": True,
                "bufsize": 1,  # line buffered
                "encoding": "utf-8",
            }
            if os.name == "nt":
                kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
            try:
                self._process = subprocess.Popen(self._command(), **kwargs)
            except OSError as exc:
                raise UciEngineError(f"Failed to start engine {self.executable!r}: {exc}") from exc

            self._reader = threading.Thread(
                target=self._read_loop, name="uci-reader", daemon=True
            )
            self._reader.start()
            self._handshake()

    def _command(self) -> list[str]:
        # A python fake/built-in engine (or any interpreter script) is supported
        # via executable strings containing arguments.
        if "--uci-engine" in self.executable:
            exe_part = self.executable.replace("--uci-engine", "").strip().strip('"')
            return [exe_part, "--uci-engine"]
        if self.executable.endswith(".py"):
            parts = self.executable.split(" ", 1)
            return [p.strip('"') for p in parts] if len(parts) == 2 else [self.executable]
        return [self.executable]

    def _read_loop(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        try:
            for line in self._process.stdout:
                self._lines.put(line.rstrip("\r\n"))
        except (ValueError, OSError) as exc:
            LOGGER.debug("Engine reader stopped: %s", exc)

    def _handshake(self) -> None:
        self._send("uci")
        self._drain_until(lambda line: line == "uciok", self.startup_timeout)
        self._send("isready")
        self._drain_until(lambda line: line == "readyok", self.startup_timeout)
        LOGGER.info("UCI engine ready: %s", self.id_name or self.executable)

    def _drain_until(self, predicate: Callable[[str], bool], timeout: float) -> list[str]:
        import time
        deadline = time.monotonic() + timeout
        seen: list[str] = []
        while time.monotonic() < deadline:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                line = self._lines.get(timeout=min(0.2, remaining) or 0.05)
            except queue.Empty:
                if self._process and self._process.poll() is not None:
                    raise UciEngineError("Engine process exited unexpectedly.")
                continue
            seen.append(line)
            self._capture_handshake_info(line)
            if predicate(line):
                return seen
        raise UciEngineError(f"Timed out waiting for engine response ({timeout:.0f}s).")

    def _capture_handshake_info(self, line: str) -> None:
        if line.startswith("id name"):
            self.id_name = line[len("id name"):].strip()
        elif line.startswith("id author"):
            self.id_author = line[len("id author"):].strip()
        elif line.startswith("option"):
            self._parse_option_line(line)

    def _parse_option_line(self, line: str) -> None:
        tokens = line.split()
        if "name" not in tokens:
            return
        try:
            name_start = tokens.index("name") + 1
            type_start = tokens.index("type")
            name = " ".join(tokens[name_start:type_start])
            kind = tokens[type_start + 1] if type_start + 1 < len(tokens) else ""
            default = None
            if "default" in tokens:
                default = tokens[tokens.index("default") + 1]
            self.supported_options[name] = {"type": kind, "default": default}
        except (IndexError, ValueError):
            LOGGER.debug("Could not parse option line: %s", line)

    # ------------------------------------------------------------- commands
    def _send(self, command: str) -> None:
        if self._process is None or self._process.stdin is None:
            raise UciEngineError("Engine is not running.")
        LOGGER.debug("uci> %s", command)
        self._process.stdin.write(command + "\n")
        self._process.stdin.flush()

    def set_option(self, name: str, value: object) -> None:
        with self._lock:
            self.options[name] = str(value)
            self._send(f"setoption name {name} value {value}")

    def new_game(self) -> None:
        with self._lock:
            self._send("ucinewgame")
            self._send("isready")
            self._drain_until(lambda line: line == "readyok", 10.0)

    def is_ready(self) -> bool:
        with self._lock:
            self._send("isready")
            try:
                self._drain_until(lambda line: line == "readyok", 5.0)
                return True
            except UciEngineError:
                return False

    def analyze(
        self,
        fen: str,
        *,
        depth: Optional[int] = None,
        movetime_ms: Optional[int] = None,
        multipv: Optional[int] = None,
        on_info: Optional[Callable[[UciInfo], None]] = None,
        stop_event: Optional[threading.Event] = None,
        timeout: float = 3600.0,
    ) -> Optional[str]:
        """Run ``go`` on a position and stream :class:`UciInfo` updates.

        Always returns the engine's bestmove (``None`` if unavailable). A
        previous search is stopped first, so the underlying process is reused
        without ever being restarted.
        """
        with self._lock:
            self._stop_search_locked()
            if multipv is not None:
                self.set_option("MultiPV", multipv)
            self._send("position fen " + fen)
            go_tokens = ["go"]
            if depth:
                go_tokens += ["depth", str(depth)]
            if movetime_ms:
                go_tokens += ["movetime", str(movetime_ms)]
            if len(go_tokens) == 1:
                go_tokens = ["go", "infinite"]
            self._send(" ".join(go_tokens))
            self._searching = True

            import time
            deadline = time.monotonic() + timeout
            bestmove: Optional[str] = None
            try:
                while time.monotonic() < deadline:
                    if stop_event is not None and stop_event.is_set():
                        self._send("stop")
                    try:
                        line = self._lines.get(timeout=0.05)
                    except queue.Empty:
                        if self._process and self._process.poll() is not None:
                            raise UciEngineError("Engine process exited during search.")
                        continue
                    if line.startswith("info"):
                        info = parse_info_line(line)
                        if info and not info.is_string and on_info is not None:
                            on_info(info)
                    elif line.startswith("bestmove"):
                        bestmove = parse_bestmove_line(line)
                        self._searching = False
                        break
                else:
                    raise UciEngineError("Analysis timed out.")
            finally:
                if bestmove is not None:
                    self._searching = False
            return bestmove

    def stop(self) -> None:
        """Interrupt any running search (blocking until bestmove arrives)."""
        with self._lock:
            self._stop_search_locked()

    def _stop_search_locked(self) -> None:
        if self._process is None or not self._searching:
            return
        self._send("stop")
        try:
            self._drain_until(lambda line: line.startswith("bestmove"), 2.0)
        except UciEngineError:
            LOGGER.debug("No bestmove seen after stop (engine may have been idle).")
        finally:
            self._searching = False

    def quit(self) -> None:
        with self._lock:
            process = self._process
            reader = self._reader
            if process is None:
                return
            try:
                self._send("quit")
                process.wait(timeout=5)
            except (UciEngineError, subprocess.TimeoutExpired, OSError):
                process.kill()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:  # pragma: no cover
                    pass
            finally:
                # Explicitly close pipes and reap the reader thread so no
                # file descriptors or daemon threads leak across analyses.
                for stream in (process.stdin, process.stdout, process.stderr):
                    try:
                        if stream is not None:
                            stream.close()
                    except OSError:
                        pass
                self._process = None
                self._reader = None
        if reader is not None and reader.is_alive():
            reader.join(timeout=3)

    # ------------------------------------------------------------- aliases
    @property
    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def __enter__(self) -> "UciEngine":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.quit()
