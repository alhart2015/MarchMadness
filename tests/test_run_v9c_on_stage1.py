"""Unit tests for src/run_v9c_on_stage1.py.

The full end-to-end smoke (running v9-C on output/pairwise_v4.csv) is
deferred to Task 8's verification step to avoid running the expensive
v9-C trainer inside the pytest path. These tests verify the module's
shape and constants only.
"""
import pytest


def test_module_imports_and_exposes_run_v9c():
    from src.run_v9c_on_stage1 import run_v9c, W_UPSET, W_MISS, FEATURE_SET

    assert callable(run_v9c)
    assert W_UPSET == 1.25
    assert W_MISS == 0.0
    assert FEATURE_SET == "v9c"


def test_cli_parser_requires_in_and_out():
    """CLI rejects calls without both --pairwise-in and --pairwise-out."""
    from src.run_v9c_on_stage1 import main

    with pytest.raises(SystemExit):
        main(["--pairwise-in", "x.csv"])  # missing --pairwise-out
    with pytest.raises(SystemExit):
        main(["--pairwise-out", "y.csv"])  # missing --pairwise-in
