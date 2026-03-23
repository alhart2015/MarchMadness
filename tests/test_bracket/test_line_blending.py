import pytest
from src.bracket.line_blending import blend_r64_probs


def test_blend_no_lines_returns_original():
    model_probs = {(1, 2): 0.7, (3, 4): 0.6}
    result = blend_r64_probs(model_probs, r64_lines={}, weight=0.35)
    assert result == model_probs


def test_blend_with_line_shifts_probability():
    model_probs = {(1, 2): 0.7}
    r64_lines = {(1, 2): 10.0}
    result = blend_r64_probs(model_probs, r64_lines, weight=0.35)
    assert 0.7 < result[(1, 2)] < 0.82


def test_blend_weight_zero_returns_model():
    model_probs = {(1, 2): 0.7}
    r64_lines = {(1, 2): 10.0}
    result = blend_r64_probs(model_probs, r64_lines, weight=0.0)
    assert abs(result[(1, 2)] - 0.7) < 1e-6


def test_blend_weight_one_returns_vegas():
    model_probs = {(1, 2): 0.7}
    r64_lines = {(1, 2): 10.0}
    result = blend_r64_probs(model_probs, r64_lines, weight=1.0)
    assert abs(result[(1, 2)] - 0.818) < 0.01
