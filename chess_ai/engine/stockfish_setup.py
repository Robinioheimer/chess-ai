"""Stockfish discovery, version probing and automated download.

Lookup order:

1. Explicit path from configuration / ``STOCKFISH_PATH`` environment variable.
2. ``stockfish`` on the system ``PATH``.
3. Well known Windows install locations (official installer, Scoop, Chocolatey).
4. A bundled binary next to the application / PyInstaller bundle (``bin/``).

If nothing is found and auto-download is enabled, the official Stockfish
GitHub release matching the platform is fetched and extracted into the user's
application directory.
"""
from __future__ import annotations

import io
import os
import platform
import shutil
import stat
import subprocess
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from ..config import config_dir
from ..logging_setup import get_logger
from .uci import UciEngineError

LOGGER = get_logger(__name__)

STOCKFISH_RELEASES_API = "https://api.github.com/repos/official-stockfish/Stockfish/releases/latest"


@dataclass
class StockfishInfo:
    path: str
    name: str
    source: str


# --------------------------------------------------------------------- lookup
def _candidate_paths() -> list[Path]:
    paths: list[Path] = []
    if os.name == "nt":
        program_files = [
            os.environ.get("ProgramFiles", r"C:\Program Files"),
            os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
            os.environ.get("LOCALAPPDATA", ""),
        ]
        patterns = [
            "Stockfish*/stockfish*.exe",
            "stockfish*/stockfish*.exe",
            "Scoop/apps/stockfish/current/stockfish*.exe",
            "chocolatey/bin/stockfish*.exe",
            "Programs/stockfish*/stockfish*.exe",
        ]
        for base in program_files:
            if not base:
                continue
            for pattern in patterns:
                paths.extend(Path(base).glob(pattern))
    local_dirs = [
        Path.cwd() / "bin",
        config_dir() / "bin",
    ]
    # PyInstaller temporary extraction directory.
    if hasattr(sys, "_MEIPASS"):  # pragma: no cover - packaging only
        local_dirs.append(Path(sys._MEIPASS) / "bin")  # type: ignore[attr-defined]
    for directory in local_dirs:
        if os.name == "nt":
            paths.extend(directory.glob("stockfish*.exe"))
        else:
            paths.extend(directory.glob("stockfish*"))
    return [p for p in paths if p.is_file()]


def find_stockfish(explicit_path: str = "") -> StockfishInfo | None:
    """Return information about a locally available Stockfish, or ``None``."""
    env_path = os.environ.get("STOCKFISH_PATH", "")
    for raw, source in (
        (explicit_path, "configuration"),
        (env_path, "STOCKFISH_PATH environment variable"),
    ):
        if raw and Path(raw).is_file():
            return StockfishInfo(str(Path(raw)), Path(raw).name, source)
    on_path = shutil.which("stockfish") or shutil.which("stockfish.exe")
    if on_path:
        return StockfishInfo(on_path, Path(on_path).name, "system PATH")
    for candidate in _candidate_paths():
        if os.access(candidate, os.X_OK) or candidate.suffix == ".exe":
            return StockfishInfo(str(candidate), candidate.name, "local installation")
    return None


def probe_stockfish(path: str, timeout: float = 10.0) -> str:
    """Run the UCI handshake and return the engine's ``id name`` string."""
    kwargs = {
        "stdin": subprocess.PIPE,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "text": True,
    }
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
    command = path.split() if path.endswith(".py") else [path]
    try:
        process = subprocess.Popen(command, **kwargs)
    except OSError as exc:
        raise UciEngineError(f"Could not launch {path!r}: {exc}") from exc
    assert process.stdin is not None and process.stdout is not None
    try:
        process.stdin.write("uci\n")
        process.stdin.flush()
        name = ""
        while True:
            line = process.stdout.readline()
            if not line:
                raise UciEngineError(f"Engine at {path!r} did not respond to UCI handshake.")
            line = line.strip()
            if line.startswith("id name"):
                name = line[len("id name"):].strip()
            if line == "uciok":
                return name or Path(path).name
    finally:
        try:
            process.stdin.write("quit\n")
            process.stdin.flush()
            process.wait(timeout=timeout)
        except (OSError, subprocess.TimeoutExpired):
            process.kill()


# ------------------------------------------------------------------- download
def _select_release_asset(assets: list[dict]) -> str:
    """Pick the best Stockfish release asset for this machine."""
    is_windows = os.name == "nt"
    machine = platform.machine().lower()
    arch_token = "x86-64" if machine in ("amd64", "x86_64") else "aarch64"
    os_token = "windows" if is_windows else "ubuntu"
    extension = ".zip" if is_windows else ".tar"
    candidates = [
        a for a in assets
        if a.get("name", "").lower().endswith(extension)
        and os_token in a.get("name", "").lower()
        and arch_token in a.get("name", "").lower()
    ]
    # Prefer modern but widely supported micro-architecture levels.
    for preference in ("avx2", "sse41", "popcnt", ""):
        for asset in candidates:
            name = asset["name"].lower()
            if preference in name:
                return asset["browser_download_url"]
    if candidates:
        return candidates[0]["browser_download_url"]
    raise UciEngineError(
        "No compatible Stockfish release found for this platform. "
        "Install Stockfish manually and configure its path in Settings."
    )


def fetch_latest_release_url() -> str:
    """Query the GitHub API for the latest release's matching asset URL."""
    request = urllib.request.Request(
        STOCKFISH_RELEASES_API, headers={"Accept": "application/vnd.github+json"}
    )
    import json
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310 (fixed URL)
        payload = json.loads(response.read().decode("utf-8"))
    return _select_release_asset(payload.get("assets", []))


def _extract_archive(archive: bytes, destination: Path) -> Path:
    """Extract a downloaded zip/tar archive and return the engine binary."""
    destination.mkdir(parents=True, exist_ok=True)
    if archive[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(archive)) as zf:
            zf.extractall(destination)
    else:
        import tarfile
        with tarfile.open(fileobj=io.BytesIO(archive)) as tf:
            tf.extractall(destination)  # noqa: S202 (trusted official source)
    executables = [
        p for p in destination.rglob("*")
        if p.is_file() and "stockfish" in p.name.lower()
        and (p.suffix == ".exe" or os.access(p, os.X_OK) or not p.suffix)
    ]
    if not executables:
        raise UciEngineError("Stockfish archive downloaded but no executable found inside.")
    binary = executables[0]
    if binary.suffix != ".exe":
        binary.chmod(binary.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return binary


def download_stockfish(destination: Path | None = None, url: str | None = None) -> StockfishInfo:
    """Download and extract Stockfish into the application directory."""
    destination = destination or (config_dir() / "bin")
    url = url or fetch_latest_release_url()
    LOGGER.info("Downloading Stockfish from %s", url)
    with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310
        archive = response.read()
    binary = _extract_archive(archive, destination)
    LOGGER.info("Stockfish installed at %s", binary)
    return StockfishInfo(str(binary), binary.name, "automatic download")


def ensure_stockfish(explicit_path: str = "", auto_download: bool = True) -> StockfishInfo:
    """Find Stockfish or install it; raise :class:`UciEngineError` on failure."""
    info = find_stockfish(explicit_path)
    if info is not None:
        return info
    if not auto_download:
        raise UciEngineError(
            "Stockfish was not found. Install it, set STOCKFISH_PATH, or enable auto-download."
        )
    return download_stockfish()
