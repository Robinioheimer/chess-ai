"""Built-in fallback UCI chess engine powered by python-chess.

When an external Stockfish binary is not installed or cannot be downloaded/run
on the host CPU, this built-in engine provides immediate UCI MultiPV analysis
(negamax alpha-beta search with piece-square tables, MVV-LVA move ordering and
quiescence search) so the overlay always displays candidate moves, win
percentages and board arrows out of the box.
"""
from __future__ import annotations

import os
import sys

# Ensure chess_ai/engine/types.py never shadows the stdlib 'types' module when
# this file is executed directly as a script by UciEngine.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR in sys.path:
    sys.path.remove(_SCRIPT_DIR)

import threading
import time
from typing import Optional

import chess

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000,
}

# Standard piece-square tables from White's perspective (a1=0 .. h8=63).
PAWN_PST = [
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10,-20,-20, 10, 10,  5,
     5, -5,-10,  0,  0,-10, -5,  5,
     0,  0,  0, 25, 25,  0,  0,  0,
     5,  5, 10, 27, 27, 10,  5,  5,
    10, 10, 20, 30, 30, 20, 10, 10,
    50, 50, 50, 50, 50, 50, 50, 50,
     0,  0,  0,  0,  0,  0,  0,  0,
]

KNIGHT_PST = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
]

BISHOP_PST = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
]

ROOK_PST = [
      0,  0,  0,  5,  5,  0,  0,  0,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     25, 25, 25, 25, 25, 25, 25, 25,
      0,  0,  0,  5,  5,  0,  0,  0,
]

QUEEN_PST = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -10,  5,  5,  5,  5,  5,  0,-10,
      0,  0,  5,  5,  5,  5,  0, -5,
     -5,  0,  5,  5,  5,  5,  0, -5,
    -10,  0,  5,  5,  5,  5,  0,-10,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20,
]

KING_PST = [
     20, 30, 10,  0,  0, 10, 30, 20,
     20, 20,  0,  0,  0,  0, 20, 20,
    -10,-20,-20,-20,-20,-20,-20,-10,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
]

PST_BY_TYPE = {
    chess.PAWN: PAWN_PST,
    chess.KNIGHT: KNIGHT_PST,
    chess.BISHOP: BISHOP_PST,
    chess.ROOK: ROOK_PST,
    chess.QUEEN: QUEEN_PST,
    chess.KING: KING_PST,
}

MATE_SCORE = 30000


def evaluate_board(board: chess.Board) -> int:
    """Static evaluation in centipawns from the side-to-move's perspective."""
    if board.is_checkmate():
        return -MATE_SCORE
    if board.is_stalemate() or board.is_insufficient_material():
        return 0
    score = 0
    for square, piece in board.piece_map().items():
        val = PIECE_VALUES[piece.piece_type]
        pst = PST_BY_TYPE[piece.piece_type]
        idx = square if piece.color == chess.WHITE else chess.square_mirror(square)
        pos = pst[idx]
        if piece.color == chess.WHITE:
            score += val + pos
        else:
            score -= val + pos
    return score if board.turn == chess.WHITE else -score


def _move_order_key(board: chess.Board, move: chess.Move) -> int:
    score = 0
    if board.is_capture(move):
        victim = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)
        v_val = PIECE_VALUES.get(victim.piece_type, 100) if victim else 100
        a_val = PIECE_VALUES.get(attacker.piece_type, 100) if attacker else 100
        score += 1000 + v_val - a_val // 10
    if move.promotion:
        score += PIECE_VALUES.get(move.promotion, 900)
    if board.gives_check(move):
        score += 300
    return score


def _quiescence(board: chess.Board, alpha: int, beta: int, nodes: list[int], qdepth: int = 2) -> int:
    nodes[0] += 1
    stand_pat = evaluate_board(board)
    if stand_pat >= beta:
        return beta
    if alpha < stand_pat:
        alpha = stand_pat
    if qdepth <= 0:
        return stand_pat
    captures = [m for m in board.legal_moves if board.is_capture(m) or m.promotion]
    captures.sort(key=lambda m: _move_order_key(board, m), reverse=True)
    for move in captures[:6]:
        board.push(move)
        score = -_quiescence(board, -beta, -alpha, nodes, qdepth - 1)
        board.pop()
        if score >= beta:
            return beta
        if score > alpha:
            alpha = score
    return alpha


def _negamax(
    board: chess.Board,
    depth: int,
    alpha: int,
    beta: int,
    stop_event: threading.Event,
    nodes: list[int],
) -> tuple[int, list[chess.Move]]:
    if stop_event.is_set():
        return evaluate_board(board), []
    if depth <= 0 or board.is_game_over():
        if board.is_checkmate():
            return -(MATE_SCORE + depth), []
        if board.is_Game_over() if hasattr(board, "is_Game_over") else board.is_game_over():
            return 0, []
        return _quiescence(board, alpha, beta, nodes), []

    nodes[0] += 1
    moves = list(board.legal_moves)
    if not moves:
        return (-MATE_SCORE if board.is_check() else 0), []
    moves.sort(key=lambda m: _move_order_key(board, m), reverse=True)

    best_score = -MATE_SCORE * 2
    best_pv: list[chess.Move] = [moves[0]]
    for move in moves:
        if stop_event.is_set():
            break
        board.push(move)
        child_score, child_pv = _negamax(
            board, depth - 1, -beta, -alpha, stop_event, nodes
        )
        score = -child_score
        board.pop()
        if score > best_score:
            best_score = score
            best_pv = [move] + child_pv
        if score > alpha:
            alpha = score
        if alpha >= beta:
            break
    return best_score, best_pv


MULTIPV = 5
STOP_EVENT = threading.Event()
CURRENT_GO: Optional[threading.Thread] = None
BOARD = chess.Board()


def _emit(line: str) -> None:
    print(line, flush=True)


def _handle_go(tokens: list[str]) -> None:
    global STOP_EVENT, CURRENT_GO
    STOP_EVENT = threading.Event()
    req_depth = 4
    movetime = 0
    if "depth" in tokens:
        try:
            req_depth = int(tokens[tokens.index("depth") + 1])
        except (ValueError, IndexError):
            req_depth = 4
    if "movetime" in tokens:
        try:
            movetime = int(tokens[tokens.index("movetime") + 1])
        except (ValueError, IndexError):
            movetime = 0
    max_depth = max(1, min(req_depth, 3))
    board_snapshot = BOARD.copy()
    multipv = max(1, MULTIPV)

    legal = list(board_snapshot.legal_moves)
    if not legal:
        _emit("bestmove (none)")
        return

    def worker() -> None:
        start = time.monotonic()
        nodes = [0]
        lines = min(multipv, len(legal))
        best_move_uci = legal[0].uci()
        for d in range(1, max_depth + 1):
            if STOP_EVENT.is_set() and d > 1:
                break
            scored_moves: list[tuple[int, list[chess.Move]]] = []
            ordered = sorted(legal, key=lambda m: _move_order_key(board_snapshot, m), reverse=True)
            for move in ordered:
                if STOP_EVENT.is_set() and d > 1 and len(scored_moves) >= lines:
                    break
                board_snapshot.push(move)
                child_score, child_pv = _negamax(
                    board_snapshot, d - 1, -MATE_SCORE * 2, MATE_SCORE * 2, STOP_EVENT, nodes
                )
                score = -child_score
                board_snapshot.pop()
                scored_moves.append((score, [move] + child_pv))
            scored_moves.sort(key=lambda item: item[0], reverse=True)
            if scored_moves:
                best_move_uci = scored_moves[0][1][0].uci()
            elapsed_ms = max(1, int((time.monotonic() - start) * 1000))
            nps = int(nodes[0] * 1000 / elapsed_ms)
            for pv_idx, (score, pv_line) in enumerate(scored_moves[:lines], start=1):
                pv_str = " ".join(m.uci() for m in pv_line)
                if abs(score) >= MATE_SCORE - 100:
                    mate_ply = max(1, (MATE_SCORE + 10 - abs(score) + 1) // 2)
                    mate_val = mate_ply if score > 0 else -mate_ply
                    score_token = f"score mate {mate_val}"
                else:
                    score_token = f"score cp {score}"
                _emit(
                    f"info depth {d} seldepth {d + 2} multipv {pv_idx} "
                    f"{score_token} nodes {nodes[0]} nps {nps} "
                    f"time {elapsed_ms} pv {pv_str}"
                )
            if movetime and elapsed_ms >= movetime:
                break
        _emit(f"bestmove {best_move_uci}")

    CURRENT_GO = threading.Thread(target=worker, daemon=True)
    CURRENT_GO.start()


def main() -> int:
    global MULTIPV, BOARD
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        tokens = line.split()
        cmd = tokens[0]
        if cmd == "uci":
            _emit("id name ChessAI Built-in Engine 1.0")
            _emit("id author Chess AI")
            _emit("option name Threads type spin default 1 min 1 max 16")
            _emit("option name Hash type spin default 64 min 1 max 4096")
            _emit("option name MultiPV type spin default 5 min 1 max 500")
            _emit("uciok")
        elif cmd == "isready":
            _emit("readyok")
        elif cmd == "setoption":
            if "MultiPV" in tokens:
                try:
                    MULTIPV = int(tokens[-1])
                except ValueError:
                    pass
        elif cmd == "ucinewgame":
            BOARD = chess.Board()
        elif cmd == "position":
            try:
                if "fen" in tokens:
                    fen_start = tokens.index("fen") + 1
                    fen_tokens: list[str] = []
                    for tok in tokens[fen_start:]:
                        if tok == "moves":
                            break
                        fen_tokens.append(tok)
                    BOARD = chess.Board(" ".join(fen_tokens))
                else:
                    BOARD = chess.Board()
                if "moves" in tokens:
                    for uci in tokens[tokens.index("moves") + 1:]:
                        BOARD.push_uci(uci)
            except Exception:
                BOARD = chess.Board()
        elif cmd == "go":
            _handle_go(tokens)
        elif cmd == "stop":
            STOP_EVENT.set()
            if CURRENT_GO is not None:
                CURRENT_GO.join(timeout=2)
        elif cmd == "quit":
            STOP_EVENT.set()
            return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
