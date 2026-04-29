"""Tests for v4 ablation env-var hooks (MM_FEATURE_DROP, MM_OUTPUT_SUFFIX).

The hooks are tiny pieces of logic embedded in enhanced_model_v3.main(),
so we test them as standalone helpers extracted into the same module
namespace. Drop logic: take a feature_cols list and an env var string,
return the filtered list plus the set of names that were unknown.
"""
from src.enhanced_model_v3 import apply_feature_drop, apply_output_suffix


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


def test_suffix_empty_returns_unchanged():
    assert apply_output_suffix("output/foo.csv", "") == "output/foo.csv"


def test_suffix_inserts_before_extension():
    assert apply_output_suffix("output/foo.csv", "_drop_coach") == "output/foo_drop_coach.csv"


def test_suffix_handles_json():
    assert apply_output_suffix("output/bracket_data.json", "_x") == "output/bracket_data_x.json"


def test_suffix_no_extension():
    # Edge case: path without extension. Just append.
    assert apply_output_suffix("output/foo", "_x") == "output/foo_x"


def test_suffix_handles_path_with_dot_in_directory():
    # e.g., output/v.4/foo.csv -- only the final ext should be split.
    assert apply_output_suffix("output/v.4/foo.csv", "_x") == "output/v.4/foo_x.csv"
