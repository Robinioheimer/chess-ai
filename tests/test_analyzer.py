"""Tests for the continuous analysis manager using the fake engine."""
import time

import chess
import pytest

from chess_ai.config import EngineConfig
from chess_ai.engine.analyzer import AnalysisManager
from chess_ai.engine.uci import UciEngine


def _wait_for(predicate, timeout=10.0, interval=0.05):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


@pytest.fixture
def manager(fake_engine_command):
    results = []
    statuses = []
    engine = UciEngine(fake_engine_command)
    config = EngineConfig(depth=5, multipv=3, threads=1, hash_mb=16)
    mgr = AnalysisManager(engine, config, results.append, statuses.append)
    mgr.start()
    yield mgr, results, statuses
    mgr.shutdown()


def test_analysis_produces_complete_multipv_result(manager):
    mgr, results, _ = manager
    mgr.request_analysis(chess.STARTING_FEN)
    assert _wait_for(lambda: results and not results[-1].searching)
    final = results[-1]
    assert final.error is None
    assert len(final.moves) == 3
    best = final.moves[0]
    assert best.san  # SAN was generated
    assert 0.0 <= best.win_probability <= 1.0
    assert best.opponent_win_probability == pytest.approx(1 - best.win_probability)
    assert best.explanation.endswith(".")
    # The fake UCI engine caps its emitted depth at 4.
    assert final.depth == min(4, mgr.config.depth)
    assert final.progress == pytest.approx(min(1.0, final.depth / mgr.config.depth))
    assert 0.0 <= final.confidence <= 1.0


def test_partial_results_report_searching_state(manager):
    mgr, results, _ = manager
    mgr.request_analysis(chess.STARTING_FEN)
    assert _wait_for(lambda: any(r.searching for r in results))
    assert _wait_for(lambda: results and not results[-1].searching)


def test_pause_and_resume(manager):
    mgr, results, _ = manager
    mgr.request_analysis(chess.STARTING_FEN)
    assert _wait_for(lambda: results and not results[-1].searching)
    mgr.pause()
    assert mgr.paused
    mgr.resume()
    assert not mgr.paused
    # Resume triggers a fresh analysis of the current position.
    assert _wait_for(lambda: any(r.searching for r in results[-5:]))


def test_reanalyze_reuses_position(manager):
    mgr, results, _ = manager
    mgr.request_analysis(chess.STARTING_FEN)
    assert _wait_for(lambda: results and not results[-1].searching)
    count = len(results)
    mgr.reanalyze()
    assert _wait_for(lambda: len(results) > count)
    assert results[-1].fen == chess.STARTING_FEN


def test_rapid_requests_keep_only_latest_position(manager):
    mgr, results, _ = manager
    fens = []
    for san in ("e4", "d4", "c4"):
        board = chess.Board()
        board.push_san(san)
        fens.append(board.fen())
        mgr.request_analysis(board.fen())
    assert _wait_for(lambda: results and not results[-1].searching)
    # The last completed analysis must be for the most recently requested FEN.
    completed = [r for r in results if not r.searching]
    assert completed[-1].fen == fens[-1]


def test_all_legal_mode_caps_multipv(manager):
    mgr, _, _ = manager
    mgr.config.analyze_all_legal = True
    mgr.request_analysis(chess.STARTING_FEN)
    results = []
    mgr.on_result = results.append
    assert _wait_for(lambda: results and not results[-1].searching)
    # Starting position has exactly 20 legal moves.
    assert len(results[-1].moves) == 20
