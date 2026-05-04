"""Unit tests for src/features/hierarchical_bt.py.

Five tests, ordered by what they verify:
1. test_gradient_matches_finite_difference -- analytic gradient correct.
2. test_loose_prior_recovers_strength_ranking -- sigma >> 1 recovers BT.
3. test_tight_prior_pulls_s_toward_x_beta   -- sigma << 1 collapses to features.
4. test_predict_pairs_symmetric             -- p(a beats b) + p(b beats a) = 1.
5. test_producer_version_constant           -- regression guard.
"""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from src.features.hierarchical_bt import (
    _PRODUCER_VERSION,
    _build_neg_log_posterior_args,
    _neg_log_posterior,
    fit_one_season,
    predict_pairs,
)


def _synthetic_season(n_teams=4, n_games_per_pair=3, seed=0):
    """Round-robin-ish schedule with random outcomes weighted by an
    underlying true strength vector. Returns
    (games_df, team_ids, feature_matrix_df, true_s)."""
    rng = np.random.default_rng(seed)
    team_ids = list(range(100, 100 + n_teams))
    true_s = rng.normal(0, 1, size=n_teams)

    rows = []
    daynum = 10
    for i in range(n_teams):
        for j in range(i + 1, n_teams):
            for _ in range(n_games_per_pair):
                p_i_wins = 1.0 / (1.0 + np.exp(-(true_s[i] - true_s[j])))
                if rng.random() < p_i_wins:
                    w, l = team_ids[i], team_ids[j]
                else:
                    w, l = team_ids[j], team_ids[i]
                wloc = rng.choice(["H", "A", "N"])
                rows.append({
                    "Season": 2024,
                    "DayNum": daynum,
                    "WTeamID": w,
                    "LTeamID": l,
                    "WLoc": wloc,
                })
                daynum += 1
    games_df = pd.DataFrame(rows)

    fm = pd.DataFrame({
        "TeamID": team_ids,
        "Season": [2024] * n_teams,
        "feat_a": true_s + rng.normal(0, 0.5, n_teams),
        "feat_b": rng.normal(0, 1, n_teams),
    })
    return games_df, team_ids, fm, true_s


def _stand_stats(fm, feature_cols):
    means = fm[feature_cols].mean()
    stds = fm[feature_cols].std(ddof=0).replace(0, 1)
    return means, stds


def test_gradient_matches_finite_difference():
    games, team_ids, fm, _ = _synthetic_season(seed=42)
    feature_cols = ["feat_a", "feat_b"]
    means, stds = _stand_stats(fm, feature_cols)
    n_teams, n_feat = len(team_ids), len(feature_cols)

    args = _build_neg_log_posterior_args(
        games, team_ids, fm, feature_cols, means, stds,
        sigma=0.5, sigma_beta=1.0,
    )

    rng = np.random.default_rng(0)
    theta = rng.normal(0, 0.5, n_teams + n_feat + 1)

    loss, grad = _neg_log_posterior(theta, *args)
    assert np.isfinite(loss)
    assert grad.shape == theta.shape

    eps = 1e-6
    grad_fd = np.empty_like(theta)
    for k in range(len(theta)):
        tp = theta.copy(); tp[k] += eps
        tm = theta.copy(); tm[k] -= eps
        lp, _ = _neg_log_posterior(tp, *args)
        lm, _ = _neg_log_posterior(tm, *args)
        grad_fd[k] = (lp - lm) / (2 * eps)

    max_err = np.max(np.abs(grad - grad_fd))
    assert max_err < 1e-5, f"max grad error {max_err:.2e}"


def test_loose_prior_recovers_strength_ranking():
    """sigma very large -> recovers a strength ordering matching the
    underlying true_s used to simulate games. Loose prior shouldn't
    distort the BT MLE direction."""
    games, team_ids, fm, true_s = _synthetic_season(
        n_teams=6, n_games_per_pair=20, seed=1,
    )
    feature_cols = ["feat_a", "feat_b"]
    means, stds = _stand_stats(fm, feature_cols)

    fit = fit_one_season(
        games, team_ids, fm, feature_cols, means, stds,
        sigma=1e3, sigma_beta=1e3,
    )
    assert fit["success"], f"failed to converge: fun={fit['fun']:.3f}"

    rho, _ = spearmanr(fit["s"], true_s)
    assert rho > 0.7, f"rank correlation {rho:.3f} too low"


def test_tight_prior_pulls_s_toward_x_beta():
    """sigma very small -> ||s - X @ beta|| approaches zero.

    Keep n_teams modest and games-per-pair high so the BT likelihood
    has clear preferences -- otherwise the tight prior can lock in
    on a noisy beta minimum."""
    games, team_ids, fm, _ = _synthetic_season(
        n_teams=8, n_games_per_pair=30, seed=2,
    )
    feature_cols = ["feat_a", "feat_b"]
    means, stds = _stand_stats(fm, feature_cols)

    fit = fit_one_season(
        games, team_ids, fm, feature_cols, means, stds,
        sigma=1e-3, sigma_beta=1.0,
    )
    assert fit["success"]

    s = fit["s"]
    Xz_full = ((fm.set_index("TeamID").loc[team_ids][feature_cols] - means) / stds).values
    pred = Xz_full @ fit["beta"]
    max_dev = np.max(np.abs(s - pred))
    assert max_dev < 5e-2, f"s deviates from X@beta by {max_dev:.3f}"


def test_predict_pairs_symmetric():
    games, team_ids, fm, _ = _synthetic_season(seed=3)
    feature_cols = ["feat_a", "feat_b"]
    means, stds = _stand_stats(fm, feature_cols)

    fit = fit_one_season(
        games, team_ids, fm, feature_cols, means, stds,
        sigma=0.5, sigma_beta=1.0,
    )
    pairs = [(team_ids[0], team_ids[1]), (team_ids[1], team_ids[0])]
    probs = predict_pairs(fit, pairs)
    assert np.isclose(probs[0] + probs[1], 1.0, atol=1e-9), (
        "p(a beats b) + p(b beats a) must be 1 under symmetric BT"
    )
    assert 0.0 < probs[0] < 1.0


def test_producer_version_constant():
    assert _PRODUCER_VERSION == "v1"
