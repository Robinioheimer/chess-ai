"""Tests for feature extraction, heuristic classification and templates."""
import chess
import numpy as np
import pytest

from chess_ai.vision.features import extract_cell_features
from chess_ai.vision.recognizer import PieceRecognizer, classify_heuristic
from chess_ai.vision.synthetic import (
    DEFAULT_DARK_SQUARE,
    DEFAULT_LIGHT_SQUARE,
    render_piece,
)

PIECE_SYMBOLS = ["P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"]


def _make_cell(symbol: str, size: int = 60, light_bg: bool = True) -> np.ndarray:
    bg = np.array(
        DEFAULT_LIGHT_SQUARE if light_bg else DEFAULT_DARK_SQUARE, dtype=np.uint8
    )
    piece = render_piece(symbol, int(size * 0.9))
    cell = np.full((size, size, 3), bg, dtype=np.uint8)
    ps = piece.shape[0]
    offset = (size - ps) // 2
    alpha = piece[:, :, 3:4].astype(np.float32) / 255.0
    cell[offset:offset + ps, offset:offset + ps] = (
        alpha * piece[:, :, :3] + (1 - alpha) * cell[offset:offset + ps, offset:offset + ps]
    ).astype(np.uint8)
    return cell


@pytest.mark.parametrize("symbol", PIECE_SYMBOLS)
@pytest.mark.parametrize("size", [48, 54, 60, 64])
def test_heuristic_classifies_every_piece(symbol, size):
    # Render each piece on both light and dark square backgrounds.
    for light in (True, False):
        features = extract_cell_features(_make_cell(symbol, size, light))
        predicted = chess.PIECE_SYMBOLS[classify_heuristic(features)]
        assert predicted == symbol.lower()


@pytest.mark.parametrize("symbol", PIECE_SYMBOLS)
def test_piece_color_detection(symbol):
    recognizer = PieceRecognizer()
    for light in (True, False):
        piece, confidence, _ = recognizer.recognize_cell(_make_cell(symbol, 60, light))
        assert piece is not None
        expected_color = chess.WHITE if symbol.isupper() else chess.BLACK
        assert piece.color == expected_color
        assert 0.0 <= confidence <= 1.0


def test_empty_square_is_not_recognized_as_piece():
    recognizer = PieceRecognizer()
    empty = np.full((60, 60, 3), DEFAULT_DARK_SQUARE, dtype=np.uint8)
    piece, confidence, features = recognizer.recognize_cell(empty)
    assert piece is None
    assert features.occupied is False
    assert confidence > 0.9


def test_template_calibration_save_load_and_match(tmp_path):
    recognizer = PieceRecognizer()
    # Calibrate from clean 64px cells of every piece type.
    cells = np.empty((8, 8), dtype=object)
    symbols: list[list[str | None]] = [[None] * 8 for _ in range(8)]
    for index, symbol in enumerate(PIECE_SYMBOLS):
        row, col = divmod(index, 8)
        cells[row, col] = _make_cell(symbol, 64, light_bg=(symbol.islower()))
        symbols[row][col] = symbol
    recognizer.calibrate_from_cells(cells, symbols)
    assert recognizer.calibrated

    path = recognizer.save(tmp_path / "templates.npz")
    reloaded = PieceRecognizer()
    reloaded.load(path)
    assert reloaded.calibrated

    # Template classifier should now recover the exact piece types.
    for symbol in PIECE_SYMBOLS:
        piece, confidence, features = reloaded.recognize_cell(
            _make_cell(symbol, 64, light_bg=(symbol.islower()))
        )
        assert piece is not None
        assert piece.symbol().lower() == symbol.lower()
        assert confidence >= 0.5
