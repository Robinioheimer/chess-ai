"""Calibrated winning-probability model and analysis confidence.

Three distinct concepts are modelled here and they must never be conflated in
the UI:

1. **Engine evaluation** - Stockfish's centipawn / mate score. A pure search
   output, not a probability.
2. **Estimated winning probability** - a *calibrated estimate* derived from the
   evaluation with the logistic model that Lichess uses for its win-percent
   display. It is an expectation over many games, not the chance that the
   single best move wins the current game.
3. **Analysis confidence** - a technical metric describing how trustworthy the
   current search output is (depth reached and stability of the best move).
   It is deliberately separate from the winning probability.

Calibration source
------------------
Lichess fits empirical game results to the logistic curve

    P(win) = 1 / (1 + exp(-k * cp))

with ``k = 0.003682237`` (cp in centipawns, from the side-to-move perspective).
Mate scores are mapped to a bounded pseudo-centipawn value using Lichess's
piecewise rule ``cp = sign * (21 - min(10, |mate|)) * 100`` so that a forced
mate approaches but never reaches 100%.

References:
* https://github.com/lichess-org/lila/blob/master/modules/eval/src/main/Eval.scala
* https://lichess.org/page/accuracy
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# Logistic growth rate fitted by Lichess on millions of rated games.
LICHESS_K = 0.003682237
# Pseudo centipawn cap for mate scores (mate in 1 -> 2000 cp ~ 99.94%).
MATE_CP_BASE = 21
MATE_CP_MIN_PLY = 10
MATE_CP_SCALE = 100


def mate_to_pseudo_cp(mate: int) -> int:
    """Map a UCI mate score (plies, signed from side to move) to pseudo cp.

    ``mate > 0`` means the side to move delivers mate, ``mate < 0`` gets mated.
    """
    magnitude = MATE_CP_BASE - min(MATE_CP_MIN_PLY, abs(mate))
    return (1 if mate > 0 else -1) * magnitude * MATE_CP_SCALE


def cp_to_win_probability(cp: float) -> float:
    """Convert a centipawn score (side to move perspective) into a win estimate.

    The returned value is in ``[0, 1]`` and represents the expected share of
    the game result (win + half draw), i.e. an *estimate*, never a guarantee.
    """
    return 1.0 / (1.0 + math.exp(-LICHESS_K * cp))


def score_to_win_probability(cp: float | None, mate: int | None) -> float:
    """Convert a Stockfish score (exactly one of cp/mate is set) to win %."""
    if mate is not None:
        return cp_to_win_probability(mate_to_pseudo_cp(mate))
    if cp is None:
        return 0.5
    return cp_to_win_probability(cp)


def opponent_perspective(cp: float | None, mate: int | None) -> tuple[float | None, int | None]:
    """Return the same score from the opponent's point of view."""
    opp_cp = None if cp is None else -cp
    opp_mate = None if mate is None else -mate
    return opp_cp, opp_mate


@dataclass
class ConfidenceInput:
    """Inputs for the analysis-confidence heuristic."""

    depth: int
    target_depth: int
    best_move_stability: float  # 0..1, share of recent iterations keeping the best move
    nodes: int = 0


def analysis_confidence(data: ConfidenceInput) -> float:
    """Return a technical confidence value in ``[0, 1]``.

    70% of the score comes from the depth reached relative to the configured
    target, 30% from best-move stability across search iterations. Node count
    only guards against trivially shallow searches.

    This measures *search reliability*, not the chance of winning.
    """
    if data.target_depth <= 0:
        depth_part = 0.0
    else:
        depth_part = min(1.0, data.depth / data.target_depth)
    stability = min(1.0, max(0.0, data.best_move_stability))
    confidence = 0.7 * depth_part + 0.3 * stability
    if data.nodes < 1000:
        confidence *= 0.5
    return round(min(1.0, max(0.0, confidence)), 4)


def confidence_label(confidence: float) -> str:
    """Human readable label for the technical confidence value."""
    if confidence >= 0.85:
        return "High"
    if confidence >= 0.6:
        return "Medium"
    if confidence >= 0.35:
        return "Low"
    return "Very low"
