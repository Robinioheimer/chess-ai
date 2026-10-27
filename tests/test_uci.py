"""Tests for UCI line parsing and the subprocess engine wrapper."""
import threading

import chess
import pytest

from chess_ai.engine.uci import (
    UciEngine,
    UciEngineError,
    parse_bestmove_line,
    parse_info_line,
)


def test_parse_info_cp_line():
    line = ("info depth 18 seldepth 27 multipv 1 score cp 37 nodes 12345 "
            "nps 900000 time 120 pv e2e4 e7e5 g1f3")
    info = parse_info_line(line)
    assert info is not None
    assert info.depth == 18
    assert info.seldepth == 27
    assert info.multipv == 1
    assert info.score_cp == 37
    assert info.score_mate is None
    assert info.nodes == 12345
    assert info.nps == 900000
    assert info.time_ms == 120
    assert info.pv[:3] == ["e2e4", "e7e5", "g1f3"]


def test_parse_info_mate_and_bounds():
    info = parse_info_line("info depth 10 multipv 2 score mate -5 upperbound pv d2d4")
    assert info.score_mate == -5
    assert info.bound == "upperbound"
    assert info.multipv == 2
    assert info.pv == ["d2d4"]


def test_parse_info_string_is_ignored():
    info = parse_info_line("info string NNUE evaluation using nn-1111cefa1111.nnue")
    assert info is not None and info.is_string


def test_parse_non_info_lines():
    assert parse_info_line("uciok") is None
    assert parse_bestmove_line("bestmove e2e4 ponder e7e5") == "e2e4"
    assert parse_bestmove_line("bestmove (none)") is None


def test_missing_executable_raises():
    with pytest.raises(UciEngineError):
        UciEngine("/nonexistent/stockfish-binary").start()


def test_full_handshake_and_analysis(fake_engine_command):
    engine = UciEngine(fake_engine_command)
    engine.start()
    try:
        assert engine.running
        assert "FakeStockfish" in engine.id_name
        assert "Threads" in engine.supported_options

        infos = []
        bestmove = engine.analyze(
            chess.STARTING_FEN, depth=4, multipv=3,
            on_info=infos.append,
        )
        assert bestmove in [m.uci() for m in chess.Board().legal_moves]
        assert infos, "expected streamed info updates"
        # MultiPV lines were requested and received.
        assert {info.multipv for info in infos} == {1, 2, 3}
        assert all(info.has_score and info.pv for info in infos)
        depths = {info.depth for info in infos}
        assert max(depths) == 4
    finally:
        engine.quit()
    assert not engine.running


def test_engine_process_is_reused_across_analyses(fake_engine_command):
    engine = UciEngine(fake_engine_command)
    engine.start()
    try:
        pid = engine._process.pid
        engine.analyze(chess.STARTING_FEN, depth=2)
        engine.analyze(chess.STARTING_FEN, depth=2)
        assert engine._process.pid == pid  # no restart between positions
    finally:
        engine.quit()


def test_stop_interrupts_search(fake_engine_command):
    engine = UciEngine(fake_engine_command)
    engine.start()
    stop = threading.Event()

    def stop_soon():
        stop.set()

    timer = threading.Timer(0.15, stop_soon)
    timer.start()
    try:
        bestmove = engine.analyze(
            chess.STARTING_FEN, depth=50, stop_event=stop, timeout=10
        )
        assert bestmove is not None
    finally:
        timer.cancel()
        engine.quit()


def test_context_manager(fake_engine_command):
    with UciEngine(fake_engine_command) as engine:
        assert engine.running
    assert not engine.running
