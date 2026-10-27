"""Application configuration with JSON persistence.

All user-tunable settings live here. The file is stored in the per-user
application directory so it survives application upgrades and never pollutes
the installation folder.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from .logging_setup import get_logger

LOGGER = get_logger(__name__)


def config_dir() -> Path:
    """Return the per-user configuration directory."""
    if os.name == "nt":
        base = os.environ.get("APPDATA") or str(Path.home())
        directory = Path(base) / "ChessAI"
    else:
        directory = Path.home() / ".chessai"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


@dataclass
class EngineConfig:
    """Settings related to the Stockfish subprocess."""

    stockfish_path: str = ""          # empty -> auto-detect
    threads: int = 2
    hash_mb: int = 256
    depth: int = 22                   # target search depth
    movetime_ms: int = 0              # 0 -> search by depth, otherwise time cap
    multipv: int = 5                  # number of candidate lines (max 500)
    analyze_all_legal: bool = False   # override multipv with number of legal moves
    ponder: bool = False


@dataclass
class VisionConfig:
    """Settings related to screen capture and board recognition."""

    poll_interval_ms: int = 1000      # time between screen captures
    warp_size: int = 480              # size of the warped board image
    stable_readings: int = 2          # identical readings required to accept change
    min_board_fraction: float = 0.02  # board must cover >=2% of the screenshot
    auto_orientation: bool = True
    perspective: str = "auto"         # "auto" | "white" | "black"
    side_to_move_default: str = "w"   # used before any move is observed


@dataclass
class OverlayConfig:
    """Settings related to the overlay window."""

    opacity: float = 0.95             # 0.3 .. 1.0
    scale: float = 1.0                # UI scale factor
    show_hotkey: str = "<ctrl>+<alt>+s"
    click_through: bool = False
    panel_on_right: bool = True
    show_arrows: bool = True
    marker_color_best: str = "#0B2A6B"  # dark blue, required by spec
    candidate_colors: list[str] = field(default_factory=lambda: [
        "#E8862C",  # orange
        "#2E9E5B",  # green
        "#8E44AD",  # purple
        "#16A6B8",  # teal
        "#C73E63",  # rose
        "#B8860B",  # dark goldenrod
        "#5D6D7E",  # slate
        "#7D6608",  # olive
        "#6C3483",
    ])


@dataclass
class AppConfig:
    """Root configuration object."""

    engine: EngineConfig = field(default_factory=EngineConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    overlay: OverlayConfig = field(default_factory=OverlayConfig)
    paused: bool = False

    # ------------------------------------------------------------------ io
    @classmethod
    def config_file(cls) -> Path:
        return config_dir() / "config.json"

    @classmethod
    def load(cls, path: Path | None = None) -> "AppConfig":
        """Load configuration, gracefully ignoring corrupt or legacy files."""
        path = path or cls.config_file()
        if not path.exists():
            return cls()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            return cls._from_dict(raw)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            LOGGER.warning("Ignoring invalid config file %s: %s", path, exc)
            return cls()

    def save(self, path: Path | None = None) -> Path:
        path = path or self.config_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return path

    # ------------------------------------------------------------- helpers
    @classmethod
    def _from_dict(cls, raw: dict) -> "AppConfig":
        cfg = cls()
        for section_name, section_cls in (
            ("engine", EngineConfig),
            ("vision", VisionConfig),
            ("overlay", OverlayConfig),
        ):
            section = getattr(cfg, section_name)
            payload = raw.get(section_name) or {}
            valid = {f.name for f in fields(section_cls)}
            for key, value in payload.items():
                if key in valid:
                    setattr(section, key, value)
        if "paused" in raw:
            cfg.paused = bool(raw["paused"])
        return cfg

    def validate(self) -> list[str]:
        """Return a list of human readable validation problems (empty = OK)."""
        problems: list[str] = []
        if self.engine.threads < 1:
            problems.append("Engine threads must be >= 1.")
        if self.engine.hash_mb < 1:
            problems.append("Engine hash memory must be >= 1 MB.")
        if not 1 <= self.engine.depth <= 100:
            problems.append("Engine depth must be between 1 and 100.")
        if not 1 <= self.engine.multipv <= 500:
            problems.append("MultiPV must be between 1 and 500.")
        if not 0.3 <= self.overlay.opacity <= 1.0:
            problems.append("Overlay opacity must be between 0.3 and 1.0.")
        if not 0.5 <= self.overlay.scale <= 3.0:
            problems.append("Overlay scale must be between 0.5 and 3.0.")
        if self.vision.poll_interval_ms < 200:
            problems.append("Poll interval must be at least 200 ms.")
        return problems
