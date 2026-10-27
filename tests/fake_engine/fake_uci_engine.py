"""Deterministic fake UCI engine for testing the client without Stockfish.

Run with ``python fake_uci_engine.py``. Implements just enough of the UCI
protocol (uci/isready/setoption/ucinewgame/position/go/stop/quit) to drive
:class:`chess_ai.engine.uci.UciEngine`, emitting deterministic scores and
principal variations derived from the legal moves of the current position.
"""
from __future__ import annotations

import sys
import threading
import time

import chess

MULTIPV = 1
STOP_EVENT = threading.Event()
CURRENT_GO: threading.Thread | None = None
BOARD = chess.Board()


def emit(line: str) -> None:
    print(line, flush=True)


def handle_go(tokens: list[str]) -> None:
    global STOP_EVENT
    STOP_EVENT = threading.Event()
    depth = 20
    movetime = 0
    if "depth" in tokens:
        depth = int(tokens[tokens.index("depth") + 1])
    if "movetime" in tokens:
        movetime = int(tokens[tokens.index("movetime") + 1])
    # Keep the fake engine snappy for tests while still emitting several depths.
    depth = min(depth, 4)

    legal = list(BOARD.legal_moves)
    if not legal:
        emit("bestmove (none)")
        return

    def worker() -> None:
        start = time.monotonic()
        lines = min(MULTIPV, len(legal))
        best_per_pv: list[str] = []
        for d in range(1, depth + 1):
            if STOP_EVENT.is_set():
                break
            for pv_index in range(1, lines + 1):
                move = legal[(pv_index - 1) % len(legal)]
                # Deterministic, distinct scores: best line positive, declining.
                cp = 50 - (pv_index - 1) * 15 + d
                pv_moves = [move.uci()]
                preview = BOARD.copy()
                preview.push(move)
                for follow in list(preview.legal_moves)[:3]:
                    pv_moves.append(follow.uci())
                emit(
                    f"info depth {d} seldepth {d + 2} multipv {pv_index} "
                    f"score cp {cp} nodes {1000 * d * pv_index} nps 120000 "
                    f"time {d * 30} pv " + " ".join(pv_moves)
                )
                if d == depth or STOP_EVENT.is_set():
                    if len(best_per_pv) < pv_index:
                        best_per_pv.append(move.uci())
            time.sleep(0.001)
            if movetime and (time.monotonic() - start) * 1000 >= movetime:
                break
        # Guarantee a bestmove per PV was captured.
        for pv_index in range(1, lines + 1):
            if len(best_per_pv) < pv_index:
                best_per_pv.append(legal[(pv_index - 1) % len(legal)].uci())
        if not STOP_EVENT.is_set() or True:
            emit(f"bestmove {best_per_pv[0]}")

    global CURRENT_GO
    CURRENT_GO = threading.Thread(target=worker, daemon=True)
    CURRENT_GO.start()


def main() -> None:
    global MULTIPV, BOARD
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        tokens = line.split()
        command = tokens[0]
        if command == "uci":
            emit("id name FakeStockfish 1.0")
            emit("id author Chess AI Tests")
            emit("option name Threads type spin default 1 min 1 max 8")
            emit("option name Hash type spin default 16 min 1 max 4096")
            emit("option name MultiPV type spin default 1 min 1 max 500")
            emit("uciok")
        elif command == "isready":
            emit("readyok")
        elif command == "setoption":
            if "MultiPV" in tokens:
                MULTIPV = int(tokens[-1])
        elif command == "ucinewgame":
            BOARD = chess.Board()
        elif command == "position":
            if "fen" in tokens:
                fen_start = tokens.index("fen") + 1
                # FEN is six fields; "moves" terminates it.
                fen_tokens: list[str] = []
                for token in tokens[fen_start:]:
                    if token == "moves":
                        break
                    fen_tokens.append(token)
                BOARD = chess.Board(" ".join(fen_tokens))
            else:
                BOARD = chess.Board()
            if "moves" in tokens:
                for uci in tokens[tokens.index("moves") + 1:]:
                    BOARD.push_uci(uci)
        elif command == "go":
            handle_go(tokens)
        elif command == "stop":
            STOP_EVENT.set()
            if CURRENT_GO is not None:
                CURRENT_GO.join(timeout=2)
        elif command == "quit":
            STOP_EVENT.set()
            return


if __name__ == "__main__":
    main()
