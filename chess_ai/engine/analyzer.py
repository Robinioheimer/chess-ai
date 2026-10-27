"""Continuous, non-blocking Stockfish analysis with MultiPV.

The manager owns one long-lived :class:`UciEngine` process and a single worker
thread. New analysis requests replace the pending job (latest-wins semantics),
so rapid board changes never queue up stale positions. Partial results are
delivered through a callback on every throttled info update, keeping the UI
responsive while a search is in progress.
"""
from __future__ import annotations

import threading
import time
from typing import Callable, Optional

import chess

from ..config import EngineConfig
from ..core.explanations import explain_move
from ..core.fen_utils import legal_move_count
from ..core.probabilities import (
    ConfidenceInput,
    analysis_confidence,
    confidence_label,
    opponent_perspective,
    score_to_win_probability,
)
from ..logging_setup import get_logger
from .types import AnalysisResult, MoveEvaluation, UciInfo
from .uci import UciEngine, UciEngineError

LOGGER = get_logger(__name__)

ResultCallback = Callable[[AnalysisResult], None]
StatusCallback = Callable[[str], None]


class AnalysisManager:
    """Persistent engine process + worker thread producing analysis results."""

    def __init__(
        self,
        engine: UciEngine,
        config: EngineConfig,
        on_result: ResultCallback,
        on_status: Optional[StatusCallback] = None,
        callback_throttle_s: float = 0.15,
    ) -> None:
        self.engine = engine
        self.config = config
        self.on_result = on_result
        self.on_status = on_status or (lambda msg: LOGGER.info(msg))
        self._throttle = callback_throttle_s

        self._wake = threading.Event()
        self._stop_event = threading.Event()
        self._shutdown_flag = False
        self._job_lock = threading.Lock()
        self._pending_fen: Optional[str] = None
        self._current_fen: Optional[str] = None
        self._paused = False
        self._worker = threading.Thread(target=self._run, name="analysis-worker", daemon=True)
        self._last_best_move: Optional[str] = None
        self._best_move_streak = 0
        self._stability_observations = 0

    # ------------------------------------------------------------ lifecycle
    def start(self) -> None:
        if not self.engine.running:
            self.engine.start()
        self._apply_engine_options()
        self.engine.new_game()
        self._worker.start()

    def shutdown(self) -> None:
        self._shutdown_flag = True
        self._stop_event.set()
        self._wake.set()
        if self._worker.is_alive():
            self._worker.join(timeout=5)
        self.engine.quit()

    def _apply_engine_options(self) -> None:
        self.engine.set_option("Threads", max(1, self.config.threads))
        self.engine.set_option("Hash", max(1, self.config.hash_mb))
        if "MultiPV" in self.engine.supported_options:
            self.engine.set_option("MultiPV", max(1, self.config.multipv))

    # -------------------------------------------------------------- control
    def request_analysis(self, fen: str) -> None:
        """Request analysis of a position (replaces any pending request).

        An in-flight search for a stale position is interrupted immediately
        so the newest board position is analyzed without waiting for the
        previous depth target.
        """
        with self._job_lock:
            self._pending_fen = fen
        self._stop_event.set()
        self._wake.set()

    def reanalyze(self) -> None:
        """Repeat analysis for the most recently requested position."""
        with self._job_lock:
            fen = self._current_fen or self._pending_fen
        if fen:
            self.request_analysis(fen)

    def pause(self) -> None:
        self._paused = True
        self._stop_event.set()  # interrupts an in-flight search loop wait
        LOGGER.info("Analysis paused")

    def resume(self) -> None:
        self._paused = False
        # Recreate the stop event so future searches can be interrupted again.
        self._stop_event = threading.Event()
        self._wake.set()
        LOGGER.info("Analysis resumed")
        self.reanalyze()

    @property
    def paused(self) -> bool:
        return self._paused

    def update_config(self, config: EngineConfig) -> None:
        self.config = config
        if self.engine.running:
            self._apply_engine_options()
        self.reanalyze()

    # ---------------------------------------------------------------- worker
    def _run(self) -> None:
        while not self._is_shutdown():
            self._wake.wait(timeout=0.5)
            self._wake.clear()
            if self._is_shutdown():
                break
            if self._paused:
                continue
            with self._job_lock:
                fen = self._pending_fen
                self._pending_fen = None
            if not fen:
                continue
            try:
                self._analyze(fen)
            except UciEngineError as exc:
                LOGGER.exception("Engine communication failure")
                self.on_result(AnalysisResult(
                    fen=fen,
                    side_to_move=fen.split()[1] if len(fen.split()) > 1 else "w",
                    error=f"Engine communication failure: {exc}",
                ))
            except Exception:  # pragma: no cover - defensive guard
                LOGGER.exception("Unexpected analysis error")

    def _is_shutdown(self) -> bool:
        return self._shutdown_flag

    def _analyze(self, fen: str) -> None:
        self._current_fen = fen
        # Fresh interrupt event for this search; request_analysis/pause can
        # set it to stop the search without affecting later jobs.
        stop = threading.Event()
        self._stop_event = stop
        board = chess.Board(fen)
        multipv = self._effective_multipv(board)
        latest: dict[int, UciInfo] = {}
        last_emit = 0.0
        depth_seen = 0
        nodes_seen = 0
        nps_seen = 0

        def handle_info(info: UciInfo) -> None:
            nonlocal last_emit, depth_seen, nodes_seen, nps_seen
            if not info.has_score or not info.pv:
                return
            latest[info.multipv] = info
            depth_seen = max(depth_seen, info.depth or 0)
            nodes_seen = info.nodes or nodes_seen
            nps_seen = info.nps or nps_seen
            now = time.monotonic()
            if now - last_emit >= self._throttle:
                last_emit = now
                self._emit(fen, board, latest, depth_seen, nodes_seen, nps_seen,
                           multipv, searching=True)

        target_depth = self.config.depth
        self.on_status(f"Analyzing depth {target_depth} with {multipv} lines...")
        self.engine.analyze(
            fen,
            depth=target_depth if not self.config.movetime_ms else None,
            movetime_ms=self.config.movetime_ms or None,
            multipv=multipv,
            on_info=handle_info,
            stop_event=stop,
        )
        self._emit(fen, board, latest, depth_seen, nodes_seen, nps_seen,
                   multipv, searching=False)
        self.on_status("Analysis complete")

    def _effective_multipv(self, board: chess.Board) -> int:
        if self.config.analyze_all_legal:
            return max(1, min(500, legal_move_count(board)))
        return max(1, min(500, self.config.multipv, legal_move_count(board) or 1))

    # ------------------------------------------------------------- building
    def _emit(
        self,
        fen: str,
        board: chess.Board,
        latest: dict[int, UciInfo],
        depth: int,
        nodes: int,
        nps: int,
        multipv: int,
        searching: bool,
    ) -> None:
        ordered = [latest[idx] for idx in sorted(latest)]
        moves: list[MoveEvaluation] = []
        for position, info in enumerate(ordered):
            uci_move = info.pv[0]
            try:
                san = self._san_line(board, info.pv)[0]
                pv_san = self._san_line(board, info.pv)
            except (ValueError, IndexError):
                san = uci_move
                pv_san = [uci_move]
            cp, mate = info.score_cp, info.score_mate
            win = score_to_win_probability(cp, mate)
            opp_cp, opp_mate = opponent_perspective(cp, mate)
            from ..core.notation import format_eval
            second = ordered[1] if len(ordered) > 1 else None
            explanation = explain_move(
                fen, uci_move, cp, mate,
                second_cp=second.score_cp if second else None,
                second_mate=second.score_mate if second else None,
            )
            moves.append(MoveEvaluation(
                multipv_index=info.multipv,
                uci=uci_move,
                san=san,
                pv_uci=list(info.pv),
                pv_san=pv_san,
                depth=info.depth or depth,
                cp=cp,
                mate=mate,
                win_probability=win,
                opponent_win_probability=1.0 - win,
                opponent_eval=format_eval(opp_cp, opp_mate),
                nodes=info.nodes or nodes,
                nps=info.nps or nps,
                time_ms=info.time_ms or 0,
                explanation=explanation,
            ))

        self._update_stability(moves)
        confidence = analysis_confidence(ConfidenceInput(
            depth=depth,
            target_depth=max(1, self.config.depth),
            best_move_stability=self._stability(),
            nodes=nodes,
        ))
        result = AnalysisResult(
            fen=fen,
            side_to_move="w" if board.turn == chess.WHITE else "b",
            moves=moves,
            searching=searching,
            depth=depth,
            target_depth=self.config.depth,
            nodes=nodes,
            nps=nps,
            progress=min(1.0, depth / max(1, self.config.depth)),
            confidence=confidence,
            confidence_label=confidence_label(confidence),
        )
        self.on_result(result)

    def _update_stability(self, moves: list[MoveEvaluation]) -> None:
        best = moves[0].uci if moves else None
        self._stability_observations += 1
        if best is None:
            return
        if best == self._last_best_move:
            self._best_move_streak += 1
        else:
            self._best_move_streak = 1
            self._last_best_move = best

    def _stability(self) -> float:
        if self._stability_observations == 0:
            return 0.0
        return self._best_move_streak / self._stability_observations

    @staticmethod
    def _san_line(board: chess.Board, pv_uci: list[str]) -> list[str]:
        """Render a UCI principal variation as SAN without mutating board."""
        preview = board.copy()
        sans: list[str] = []
        for uci in pv_uci:
            move = chess.Move.from_uci(uci)
            if move not in preview.legal_moves:
                break
            sans.append(preview.san(move))
            preview.push(move)
        return sans
