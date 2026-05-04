"""Per-season hierarchical Bradley-Terry MAP solver with feature priors.

Spec: docs/superpowers/specs/2026-05-03-hierarchical-bt-feature-priors-design.md

Model (per season):
    s_team_i ~ Normal(beta . v4_features_team_i, sigma^2)
    P(team a beats team b on neutral) = sigmoid(s[a] - s[b])
    P(team a beats team b at home)    = sigmoid(s[a] - s[b] + home * h)

The joint MAP objective (negative log posterior to minimize) is:

    L(s, beta, h) =
        sum_g -log sigmoid(s[w_g] - s[l_g] + home_g * h)
      + (1 / (2 sigma^2))      * ||s - Xz @ beta||^2
      + (1 / (2 sigma_beta^2)) * ||beta||^2

where Xz is the per-LOSO-fold-standardized v4 feature matrix in
team_ids order. This is convex in (s, beta, h) jointly; we solve it
via L-BFGS-B with analytic gradient.

Anchor cells:
  sigma -> infinity  : recovers plain BT (prior uninformative).
  sigma -> 0         : s collapses onto Xz @ beta exactly.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, log_expit

_PRODUCER_VERSION = "v1"


def _extract_home_court_value(wloc: str) -> int:
    """WLoc -> home-court column value relative to the *winner*:
        H -> +1 (winner was home)
        A -> -1 (winner was away)
        N ->  0 (neutral)
    """
    if wloc == "H":
        return 1
    if wloc == "A":
        return -1
    return 0


def _pack(s: np.ndarray, beta: np.ndarray, h: float) -> np.ndarray:
    return np.concatenate([s, beta, np.array([h])])


def _unpack(theta: np.ndarray, n_teams: int, n_features: int):
    s = theta[:n_teams]
    beta = theta[n_teams:n_teams + n_features]
    h = float(theta[n_teams + n_features])
    return s, beta, h


def _build_neg_log_posterior_args(
    games_df: pd.DataFrame,
    team_ids: Sequence[int],
    feature_matrix: pd.DataFrame,
    feature_cols: Sequence[str],
    feature_means: pd.Series,
    feature_stds: pd.Series,
    sigma: float,
    sigma_beta: float,
):
    """Pack the constants the optimizer needs into a tuple of numpy
    arrays. Validates inputs."""
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    if sigma_beta <= 0:
        raise ValueError(f"sigma_beta must be > 0, got {sigma_beta}")

    team_ids = list(team_ids)
    n_teams = len(team_ids)
    feature_cols = list(feature_cols)
    n_features = len(feature_cols)

    team_idx = {int(t): i for i, t in enumerate(team_ids)}

    fm_indexed = feature_matrix.set_index("TeamID")
    missing = [t for t in team_ids if t not in fm_indexed.index]
    if missing:
        raise ValueError(
            f"feature_matrix missing rows for {len(missing)} teams; "
            f"first few: {missing[:5]}"
        )

    X = fm_indexed.loc[team_ids][feature_cols].to_numpy(dtype=np.float64)
    means = feature_means[feature_cols].to_numpy(dtype=np.float64)
    stds = feature_stds[feature_cols].to_numpy(dtype=np.float64)
    Xz = (X - means) / stds

    if not np.isfinite(Xz).all():
        n_bad = (~np.isfinite(Xz)).sum()
        raise ValueError(
            f"Xz has {n_bad} non-finite entries after standardization. "
            f"Check feature_means/feature_stds and pre-filter NaN columns."
        )

    if len(games_df) == 0:
        raise ValueError("games_df is empty")

    w_idx = games_df["WTeamID"].map(team_idx)
    l_idx = games_df["LTeamID"].map(team_idx)
    if w_idx.isna().any() or l_idx.isna().any():
        bad = games_df[w_idx.isna() | l_idx.isna()].head()
        raise ValueError(
            f"games_df has team IDs outside team_ids; sample: {bad.to_dict()}"
        )
    w_idx = w_idx.to_numpy(dtype=np.int64)
    l_idx = l_idx.to_numpy(dtype=np.int64)

    home = np.array(
        [_extract_home_court_value(v) for v in games_df["WLoc"].astype(str)],
        dtype=np.float64,
    )

    inv_sigma2 = 1.0 / (sigma * sigma)
    inv_sigma_beta2 = 1.0 / (sigma_beta * sigma_beta)

    return (
        n_teams,
        n_features,
        w_idx,
        l_idx,
        home,
        Xz,
        inv_sigma2,
        inv_sigma_beta2,
    )


def _neg_log_posterior(
    theta: np.ndarray,
    n_teams: int,
    n_features: int,
    w_idx: np.ndarray,
    l_idx: np.ndarray,
    home: np.ndarray,
    Xz: np.ndarray,
    inv_sigma2: float,
    inv_sigma_beta2: float,
):
    """Return (loss, grad) for L-BFGS-B."""
    s, beta, h = _unpack(theta, n_teams, n_features)

    z = s[w_idx] - s[l_idx] + home * h
    p = expit(z)

    nll = -float(np.sum(log_expit(z)))

    Xz_beta = Xz @ beta
    resid = s - Xz_beta
    pen_s = 0.5 * inv_sigma2 * float(resid @ resid)
    pen_beta = 0.5 * inv_sigma_beta2 * float(beta @ beta)

    loss = nll + pen_s + pen_beta

    # Gradient
    dz = p - 1.0  # d/dz of -log sigmoid(z)

    grad_s = np.zeros(n_teams, dtype=np.float64)
    np.add.at(grad_s, w_idx, dz)
    np.add.at(grad_s, l_idx, -dz)
    grad_s += inv_sigma2 * resid

    grad_beta = -inv_sigma2 * (Xz.T @ resid) + inv_sigma_beta2 * beta
    grad_h = float(np.sum(home * dz))

    grad = np.concatenate([grad_s, grad_beta, np.array([grad_h])])
    return loss, grad


def fit_one_season(
    games_df: pd.DataFrame,
    team_ids: Sequence[int],
    feature_matrix: pd.DataFrame,
    feature_cols: Sequence[str],
    feature_means: pd.Series,
    feature_stds: pd.Series,
    sigma: float,
    sigma_beta: float = 1.0,
    max_iter: int = 500,
    tol: float = 1e-8,
) -> dict:
    """Fit per-season hierarchical BT MAP via L-BFGS-B.

    Args:
        games_df:       Regular-season games with WTeamID, LTeamID, WLoc.
        team_ids:       Ordered list of team IDs for this season; the
                        column index of s. All games' WTeamID/LTeamID
                        must be in this list.
        feature_matrix: pd.DataFrame with TeamID column; one row per
                        (team, season) used as the prior on s. Must
                        contain rows for every team in team_ids.
        feature_cols:   List of feature columns to use as the prior.
                        Should already be NaN-filtered at the
                        feature-matrix level.
        feature_means:  pd.Series of train-fold means (per feature).
        feature_stds:   pd.Series of train-fold stds (per feature).
        sigma:          Std of the per-team Normal prior on (s_i - X_i beta).
                        sigma -> 0 collapses s onto X beta. sigma -> inf
                        recovers plain BT.
        sigma_beta:     Std of the L2 prior on beta. Default 1.0.
        max_iter:       L-BFGS-B max iterations.
        tol:            Convergence tolerance for both ftol and gtol.

    Returns:
        dict with keys
            s         -- np.ndarray, shape (n_teams,), fitted strengths.
            beta      -- np.ndarray, shape (n_features,), fitted coeffs.
            h         -- float, fitted home-court advantage.
            team_ids  -- list[int], same as input (for index lookup).
            success   -- bool.
            n_iter    -- int, optimizer iterations used.
            fun       -- float, final negative log posterior.
    """
    team_ids = list(team_ids)
    feature_cols = list(feature_cols)
    n_teams = len(team_ids)
    n_features = len(feature_cols)

    args = _build_neg_log_posterior_args(
        games_df, team_ids, feature_matrix, feature_cols,
        feature_means, feature_stds, sigma, sigma_beta,
    )

    theta0 = np.zeros(n_teams + n_features + 1, dtype=np.float64)

    result = minimize(
        _neg_log_posterior,
        theta0,
        args=args,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": max_iter, "ftol": tol, "gtol": tol},
    )

    s, beta, h = _unpack(result.x, n_teams, n_features)

    return {
        "s": s,
        "beta": beta,
        "h": h,
        "team_ids": team_ids,
        "success": bool(result.success),
        "n_iter": int(result.nit),
        "fun": float(result.fun),
    }


def predict_pairs(
    fit: dict,
    pairs: list[tuple[int, int]],
) -> np.ndarray:
    """Return p(a beats b) on a neutral court for each (a, b) pair.

    No home-court term applied -- tournament pairs are neutral.
    """
    team_idx = {int(t): i for i, t in enumerate(fit["team_ids"])}
    s = fit["s"]
    out = np.empty(len(pairs), dtype=np.float64)
    for k, (a, b) in enumerate(pairs):
        ia = team_idx[int(a)]
        ib = team_idx[int(b)]
        out[k] = expit(s[ia] - s[ib])
    return out
