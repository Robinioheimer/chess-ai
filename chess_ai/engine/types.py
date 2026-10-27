"""Data types shared by the UCI wrapper and the analysis manager."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class UciInfo:
    """Parsed contents of a single UCI ``info`` line."""

    depth: Optional[int] = None
    seldepth: Optional[int] = None
    multipv: int = 1
    score_cp: Optional[int] = None
    score_mate: Optional[int] = None
    bound: Optional[str] = None  # "lowerbound" | "upperbound" | None
    nodes: Optional[int] = None
    nps: Optional[int] = None
    time_ms: Optional[int] = None
    pv: list[str] = field(default_factory=list)
    is_string: bool = False      # "info string ..." chatter, ignored for analysis

    @property
    def has_score(self) -> bool:
        return self.score_cp is not None or self.score_mate is not None


@dataclass
class MoveEvaluation:
    """One candidate move with all information shown by the overlay."""

    multipv_index: int
    uci: str
    san: str
    pv_uci: list[str]
    pv_san: list[str]
    depth: int
    cp: Optional[int]
    mate: Optional[int]
    win_probability: float
    opponent_win_probability: float
    opponent_eval: str
    nodes: int
    nps: int
    time_ms: int
    explanation: str
    visible: bool = True

    @property
    def eval_text(self) -> str:
        from ..core.notation import format_eval
        return format_eval(self.cp, self.mate)

    @property
    def win_percent_text(self) -> str:
        return f"{self.win_probability * 100:.1f}%"


@dataclass
class AnalysisResult:
    """Snapshot of an ongoing or completed analysis."""

    fen: str
    side_to_move: str
    moves: list[MoveEvaluation] = field(default_factory=list)
    searching: bool = False
    depth: int = 0
    target_depth: int = 0
    nodes: int = 0
    nps: int = 0
    progress: float = 0.0
    confidence: float = 0.0
    confidence_label: str = "Very low"
    error: Optional[str] = None

    @property
    def best_move(self) -> Optional[MoveEvaluation]:
        return self.moves[0] if self.moves else None
