"""Stockfish discovery, validation and optional download helper.

The overlay relies on a working UCI chess engine. This module locates an
existing Stockfish binary across common locations, validates that it speaks
UCI, downloads an official release archive on demand, and provides a built-in
Python UCI engine fallback so move analysis and board arrows always work even
without an external binary or internet access.
"""
from __future__ import annotations

import io
import json
import os
import platform
import shutil
import stat
import sys
import tarfile
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from ..config import config_dir
from ..logging_setup import get_logger
from .uci import UciEngine, UciEngineError

LOGGER = get_logger(__name__)

GITHUB_RELEASES_API = (
    "https://api.github.com/repos/official-stockfish/Stockfish/releases/latest"
)


@dataclass(frozen=True)
class StockfishInfo:
    """Resolved Stockfish binary and its reported engine name."""

    path: str
    name: str
    source: str


def builtin_engine_command() -> str:
    """Return the command string that launches the built-in UCI engine."""
    if getattr(sys, "frozen", False):
        return f"{sys.executable} --uci-engine"
    script = Path(__file__).with_name("builtin_engine.py")
    return f"{sys.executable} {script}"


def builtin_stockfish_info() -> StockfishInfo:
    """Return StockfishInfo pointing to the built-in Python UCI engine."""
    return StockfishInfo(
        path=builtin_engine_command(),
        name="ChessAI Built-in Engine 1.0",
        source="built-in engine",
    )


def _rank_stockfish_binary(path: Path) -> tuple[int, str]:
    """Rank candidate Stockfish binaries from most CPU-compatible to least.

    Universal Stockfish archives contain avx512, vnni, bmi2, avx2, sse41-popcnt
    and base x86-64 builds. Picking avx512 first crashes with SIGILL on CPUs
    without AVX-512 support.
    """
    name = path.name.lower()
    if "avx512" in name or "vnni" in name or "amx" in name:
        rank = 90
    elif "bmi2" in name:
        rank = 70
    elif "avx2" in name:
        rank = 10
    elif "sse41" in name or "popcnt" in name or "modern" in name:
        rank = 20
    elif "x86-64" in name or name in ("stockfish", "stockfish.exe"):
        rank = 30
    else:
        rank = 50
    return (rank, name)


def _candidate_paths() -> list[tuple[Path, str]]:
    candidates: list[tuple[Path, str]] = []
    bin_dir = config_dir() / "bin"
    if bin_dir.exists():
        found = sorted(
            [p for p in bin_dir.rglob("stockfish*") if p.is_file()],
            key=_rank_stockfish_binary,
        )
        for path in found:
            candidates.append((path, "managed download directory"))
    local_dir = Path.cwd() / "stockfish"
    if local_dir.exists():
        found = sorted(
            [p for p in local_dir.rglob("stockfish*") if p.is_file()],
            key=_rank_stockfish_binary,
        )
        for path in found:
            candidates.append((path, "working directory"))
    if platform.system() == "Windows":
        program_files = [
            os.environ.get("ProgramFiles", r"C:\Program Files"),
            os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
            os.environ.get("LOCALAPPDATA", ""),
        ]
        for base in filter(None, program_files):
            folder = Path(base) / "Stockfish"
            if folder.exists():
                found = sorted(
                    [p for p in folder.rglob("stockfish*.exe") if p.is_file()],
                    key=_rank_stockfish_binary,
                )
                for path in found:
                    candidates.append((path, "Program Files"))
    else:
        for sys_path in ("/usr/games/stockfish", "/usr/local/bin/stockfish", "/usr/bin/stockfish"):
            p = Path(sys_path)
            if p.exists():
                candidates.append((p, "system package"))
    return candidates


def _is_executable_file(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    if platform.system() == "Windows":
        return path.suffix.lower() in (".exe", ".bat", ".cmd") or os.access(path, os.X_OK)
    return os.access(path, os.X_OK)


def find_stockfish(explicit_path: str = "") -> Optional[StockfishInfo]:
    """Locate a Stockfish executable without starting a search."""
    if explicit_path:
        path = Path(explicit_path).expanduser()
        if _is_executable_file(path):
            return StockfishInfo(str(path), "Stockfish (configured)", "settings")
        # Allow command strings such as `python fake_uci_engine.py` in tests.
        parts = explicit_path.split()
        if parts and (Path(parts[0]).exists() or shutil.which(parts[0])):
            return StockfishInfo(explicit_path, "Custom UCI engine", "settings")

    env_path = os.environ.get("STOCKFISH_PATH", "").strip()
    if env_path:
        path = Path(env_path).expanduser()
        if _is_executable_file(path):
            return StockfishInfo(str(path), "Stockfish (STOCKFISH_PATH)", "STOCKFISH_PATH environment variable")

    which_hit = shutil.which("stockfish")
    if which_hit:
        return StockfishInfo(which_hit, "Stockfish (PATH)", "system PATH")

    candidates = [
        (path, source)
        for path, source in _candidate_paths()
        if _is_executable_file(path)
    ]
    if len(candidates) > 1:
        for path, source in candidates:
            try:
                name = probe_stockfish(str(path))
                return StockfishInfo(str(path), name, source)
            except Exception:
                continue
    for path, source in candidates:
        return StockfishInfo(str(path), "Stockfish", source)
    return None


def probe_stockfish(path: str) -> str:
    """Start the binary briefly, perform UCI handshake and return its name."""
    engine = UciEngine(path)
    engine.start()
    try:
        return engine.id_name or "Stockfish"
    finally:
        engine.quit()


def _select_release_asset(assets: list[dict]) -> Optional[dict]:
    system = platform.system().lower()
    machine = platform.machine().lower()
    is_arm = "arm" in machine or "aarch64" in machine

    if system == "windows":
        os_tokens = ("windows",)
        archive_exts = (".zip",)
    elif system == "darwin":
        os_tokens = ("macos", "darwin")
        archive_exts = (".tar.gz", ".tar", ".zip")
    else:
        os_tokens = ("linux", "ubuntu")
        archive_exts = (".tar.gz", ".tar", ".zip")

    if is_arm:
        arch_preferences = ["arm64", "aarch64", "universal"]
        forbidden_arch = ("x86-64",)
    else:
        arch_preferences = [
            "x86-64-avx2",
            "x86-64-sse41-popcnt",
            "x86-64-modern",
            "x86-64-universal",
            "x86-64",
        ]
        forbidden_arch = ("arm64", "aarch64", "android", "armv")

    for pref in arch_preferences:
        for asset in assets:
            name = asset.get("name", "").lower()
            if (
                any(tok in name for tok in os_tokens)
                and pref in name
                and not any(bad in name for bad in forbidden_arch)
                and any(name.endswith(ext) for ext in archive_exts)
            ):
                return asset
    return None


def download_stockfish(
    progress: Optional[Callable[[str], None]] = None,
    target_dir: Optional[Path] = None,
) -> StockfishInfo:
    """Download the official Stockfish release for the active platform."""
    target = (target_dir or (config_dir() / "bin")).expanduser()
    target.mkdir(parents=True, exist_ok=True)
    if progress:
        progress("Querying official Stockfish release metadata...")
    try:
        req = urllib.request.Request(
            GITHUB_RELEASES_API,
            headers={"User-Agent": "ChessAI-Overlay/1.0"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            release = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise UciEngineError(
            f"Could not query Stockfish release information: {exc}. "
            "Please download Stockfish from https://stockfishchess.org/download/ "
            "and select the executable in Settings."
        ) from exc

    asset = _select_release_asset(release.get("assets", []))
    if not asset:
        raise UciEngineError(
            "No compatible prebuilt Stockfish archive found for this platform. "
            "Please install Stockfish manually and configure its path."
        )

    url = asset["browser_download_url"]
    name = asset["name"]
    if progress:
        progress(f"Downloading {name}...")
    try:
        dl_req = urllib.request.Request(
            url, headers={"User-Agent": "ChessAI-Overlay/1.0"}
        )
        with urllib.request.urlopen(dl_req, timeout=60) as resp:
            payload = resp.read()
    except (urllib.error.URLError, TimeoutError) as exc:
        raise UciEngineError(f"Download of {name} failed: {exc}") from exc

    if progress:
        progress("Extracting Stockfish binary...")
    binary = _extract_archive(payload, target)
    engine_name = probe_stockfish(str(binary))
    LOGGER.info("Downloaded Stockfish binary %s (%s)", binary, engine_name)
    return StockfishInfo(str(binary), engine_name, "automatic download")


def _extract_archive(payload: bytes, target: Path) -> Path:
    buffer = io.BytesIO(payload)
    if zipfile.is_zipfile(buffer):
        buffer.seek(0)
        with zipfile.ZipFile(buffer) as zf:
            zf.extractall(target)
    else:
        buffer.seek(0)
        with tarfile.open(fileobj=buffer) as tf:
            tf.extractall(target)

    executables = [
        p for p in target.rglob("stockfish*")
        if p.is_file() and not p.name.endswith((".zip", ".tar", ".gz", ".txt", ".md", ".1"))
    ]
    if not executables:
        raise UciEngineError("Downloaded archive did not contain a Stockfish executable.")
    executables.sort(key=_rank_stockfish_binary)
    for binary in executables:
        if platform.system() != "Windows":
            binary.chmod(binary.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    if len(executables) > 1:
        for candidate in executables:
            try:
                probe_stockfish(str(candidate))
                return candidate
            except Exception:
                continue
    return executables[0]


def ensure_stockfish(
    explicit_path: str = "",
    auto_download: bool = True,
    progress: Optional[Callable[[str], None]] = None,
) -> StockfishInfo:
    """Find an existing Stockfish binary, download one, or use the built-in engine."""
    info = find_stockfish(explicit_path)
    if info is not None:
        return info
    if not auto_download:
        raise UciEngineError(
            "Stockfish binary not found. Install Stockfish or set its path in Settings."
        )
    try:
        return download_stockfish(progress=progress)
    except Exception as exc:
        LOGGER.warning(
            "External Stockfish download failed (%s); falling back to built-in UCI engine.",
            exc,
        )
        return builtin_stockfish_info()
