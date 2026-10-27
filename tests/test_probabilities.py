"""Tests for the calibrated probability model and confidence heuristic."""
import math

import pytest

from chess_ai.core.probabilities import (
    LICHESS_K,
    ConfidenceInput,
    analysis_confidence,
    confidence_label,
    cp_to_win_probability,
    mate_to_pseudo_cp,
    opponent_perspective,
    score_to_win_probability,
)


def test_zero_eval_is_fifty_percent():
    assert cp_to_win_probability(0) == pytest.approx(0.5)


def test_logistic_symmetry():
    for cp in (-500, -120, 0, 40, 330):
        assert cp_to_win_probability(cp) + cp_to_win_probability(-cp) == pytest.approx(1.0)


def test_known_lichess_point():
    # +100 cp should land around 59.1% with the documented constant.
    expected = 1 / (1 + math.exp(-LICHESS_K * 100))
    assert cp_to_win_probability(100) == pytest.approx(expected)
    assert 0.58 < cp_to_win_probability(100) < 0.60


def test_mate_pseudo_cp_bounds_and_sign():
    assert mate_to_pseudo_cp(1) == 2000
    assert mate_to_pseudo_cp(10) == 1100
    assert mate_to_pseudo_cp(99) == 1100  # capped
    assert mate_to_pseudo_cp(-2) == -1900


def test_mate_probability_near_but_below_one():
    p = score_to_win_probability(None, 1)
    assert 0.99 < p < 1.0
    assert score_to_win_probability(None, -1) == pytest.approx(1 - p)


def test_opponent_perspective():
    assert opponent_perspective(120, None) == (-120, None)
    assert opponent_perspective(None, 3) == (None, -3)


def test_confidence_scales_with_depth_and_stability():
    shallow = analysis_confidence(ConfidenceInput(depth=5, target_depth=20, best_move_stability=0.2))
    deep = analysis_confidence(ConfidenceInput(depth=20, target_depth=20, best_move_stability=1.0))
    assert 0 <= shallow < deep <= 1
    # Few nodes penalise confidence.
    weak = analysis_confidence(ConfidenceInput(10, 10, 1.0, nodes=10))
    strong = analysis_confidence(ConfidenceInput(10, 10, 1.0, nodes=1_000_000))
    assert weak < strong


def test_confidence_labels_are_defined_for_full_range():
    assert confidence_label(0.9) == "High"
    assert confidence_label(0.7) == "Medium"
    assert confidence_label(0.4) == "Low"
    assert confidence_label(0.1) == "Very low"
