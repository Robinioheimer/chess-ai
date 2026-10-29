"""Environment and engine diagnostics.

Run with ``python -m chess_ai.doctor``. Checks the Python environment, screen
capture backend and Stockfish availability, and performs a synthetic
board-recognition self-test without touching the real screen.
"""
from __future__ import annotations

import sys

from .engine.stockfish_setup import find_stockfish, probe_stockfish
from .engine.uci import UciEngineError
from .logging_setup import configure_logging


def check_imports() -> list[tuple[str, bool, str]]:
    results = []
    for module in ("chess", "cv2", "numpy", "mss", "PyQt5.QtWidgets"):
        try:
            __import__(module)
            results.append((module, True, "import OK"))
        except ImportError as exc:
            results.append((module, False, str(exc)))
    import importlib.util
    if importlib.util.find_spec("pynput") is not None:
        results.append(("pynput (global hotkeys)", True, "import OK"))
    else:
        results.append(("pynput (global hotkeys)", False,
                        "not installed; global hotkey disabled"))
    return results


def check_vision() -> tuple[bool, str]:
    try:
        import chess
        from .core.fen_utils import board_to_grid
        from .vision.pipeline import RecognitionPipeline
        from .vision.synthetic import render_board

        frame = render_board(board_to_grid(chess.Board()), square_px=60)
        pipeline = RecognitionPipeline(stable_readings=1, manual_only=False)
        result = pipeline.process_frame(frame)
        if not result.success:
            return False, result.error or "recognition failed"
        recognized = result.fen.split()[0]
        expected = chess.Board().board_fen()
        if recognized != expected:
            return False, f"FEN mismatch: got {recognized}"
        return True, "synthetic starting position recognized correctly"
    except Exception as exc:  # pragma: no cover - diagnostic path
        return False, f"{type(exc).__name__}: {exc}"


def check_engine() -> tuple[bool, str]:
    info = find_stockfish()
    if info is None:
        return False, "Stockfish not found (auto-download runs on app start)"
    try:
        name = probe_stockfish(info.path)
        return True, f"{name} at {info.path} (source: {info.source})"
    except UciEngineError as exc:
        return False, str(exc)


def main() -> int:
    configure_logging()
    print("Chess AI – environment diagnostics\n")
    failures = 0
    print("Dependencies:")
    for name, ok, detail in check_imports():
        print(f"  [{'OK' if ok else 'XX'}] {name}: {detail}")
        failures += 0 if ok else 1

    print("\nBoard recognition self-test:")
    ok, detail = check_vision()
    print(f"  [{'OK' if ok else 'XX'}] {detail}")
    failures += 0 if ok else 1

    print("\nStockfish:")
    ok, detail = check_engine()
    print(f"  [{'OK' if ok else 'XX'}] {detail}")
    failures += 0 if ok else 1

    print()
    if failures:
        print(f"{failures} issue(s) found.")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
