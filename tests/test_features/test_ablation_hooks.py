"""Tests for v4 ablation env-var hooks (MM_FEATURE_DROP, MM_OUTPUT_SUFFIX).

The hooks are tiny pieces of logic embedded in enhanced_model_v3.main(),
so we test them as standalone helpers extracted into the same module
namespace. Drop logic: take a feature_cols list and an env var string,
return the filtered list plus the set of names that were unknown.
"""
import os
from src.enhanced_model_v3 import apply_feature_drop


def test_drop_env_empty_returns_unchanged():
    cols = ["a", "b", "c"]
    result, missing = apply_feature_drop(cols, "")
    assert result == cols
    assert missing == set()


def test_drop_env_removes_named_columns():
    cols = ["a", "b", "c", "d"]
    result, missing = apply_feature_drop(cols, "b,d")
    assert result == ["a", "c"]
    assert missing == set()


def test_drop_env_strips_whitespace():
    cols = ["a", "b", "c"]
    result, missing = apply_feature_drop(cols, " a , c ")
    assert result == ["b"]
    assert missing == set()


def test_drop_env_reports_unknown_names():
    cols = ["a", "b"]
    result, missing = apply_feature_drop(cols, "a,zzz")
    assert result == ["b"]
    assert missing == {"zzz"}


def test_drop_env_preserves_order():
    cols = ["d", "c", "b", "a"]
    result, _ = apply_feature_drop(cols, "c")
    assert result == ["d", "b", "a"]
