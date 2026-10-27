"""Centralised logging configuration.

A rotating log file is written to the per-user application directory so that
long analysis sessions can be diagnosed without flooding the console.
"""
from __future__ import annotations

import logging
import logging.handlers
import os
from pathlib import Path

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
LOG_MAX_BYTES = 1024 * 1024  # 1 MB per file
LOG_BACKUPS = 3


def default_log_dir() -> Path:
    """Return (and create) the platform specific log directory."""
    if os.name == "nt":  # Windows
        base = os.environ.get("APPDATA") or str(Path.home())
        log_dir = Path(base) / "ChessAI" / "logs"
    else:
        log_dir = Path.home() / ".chessai" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def configure_logging(level: int = logging.INFO, log_dir: Path | None = None) -> Path:
    """Configure root logging once.

    Returns the path of the rotating log file.
    """
    log_dir = log_dir or default_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "chess_ai.log"

    root = logging.getLogger()
    # Avoid attaching duplicate handlers when called twice (tests, reloads).
    if not any(isinstance(h, logging.handlers.RotatingFileHandler)
               and getattr(h, "_chess_ai", False) for h in root.handlers):
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUPS, encoding="utf-8"
        )
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        file_handler._chess_ai = True  # type: ignore[attr-defined]
        root.addHandler(file_handler)

    if not any(isinstance(h, logging.StreamHandler)
               and not isinstance(h, logging.handlers.RotatingFileHandler)
               for h in root.handlers):
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter(LOG_FORMAT))
        root.addHandler(console)

    root.setLevel(level)
    # Third-party libraries are chatty; keep them at warning level.
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("mss").setLevel(logging.WARNING)
    return log_file


def get_logger(name: str) -> logging.Logger:
    """Convenience accessor for module loggers."""
    return logging.getLogger(name)
