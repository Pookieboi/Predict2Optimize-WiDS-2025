"""
Classical and robust Markowitz mean-variance optimizers (cvxpy), shared by
Week 4 (single-snapshot demonstration) and Week 5 (rolling backtest) so both
weeks solve the exact same convex program.
"""

from __future__ import annotations

import numpy as np
import cvxpy as cp

MAX_WEIGHT = 0.4  # per-asset cap: long-only, bounded, fully invested


def solve_classical(mu: np.ndarray, Sigma: np.ndarray, gamma: float = 5.0, max_w: float = MAX_WEIGHT) -> np.ndarray:
    """
    Classical Markowitz:
        max_w  mu^T w - gamma * w^T Sigma w
        s.t.   sum(w) == 1, 0 <= w <= max_w
    """
    n = len(mu)
    w = cp.Variable(n)
    objective = cp.Maximize(mu @ w - gamma * cp.quad_form(w, cp.psd_wrap(Sigma)))
    constraints = [cp.sum(w) == 1, w >= 0, w <= max_w]
    cp.Problem(objective, constraints).solve()
    return np.array(w.value)


def solve_robust(
    mu: np.ndarray,
    Sigma: np.ndarray,
    sigma_mu: np.ndarray,
    gamma: float = 5.0,
    kappa: float = 1.0,
    max_w: float = MAX_WEIGHT,
) -> np.ndarray:
    """
    Robust Markowitz with ellipsoidal uncertainty on mu (worst-case return
    over {mu_hat + delta : ||D^-1/2 delta||_2 <= kappa}, D = diag(sigma_mu^2)):
        max_w  mu^T w - kappa * || sigma_mu * w ||_2 - gamma * w^T Sigma w
        s.t.   sum(w) == 1, 0 <= w <= max_w
    kappa=0 recovers classical Markowitz exactly.
    """
    n = len(mu)
    w = cp.Variable(n)
    worst_case_penalty = kappa * cp.norm(cp.multiply(sigma_mu, w), 2)
    objective = cp.Maximize(mu @ w - worst_case_penalty - gamma * cp.quad_form(w, cp.psd_wrap(Sigma)))
    constraints = [cp.sum(w) == 1, w >= 0, w <= max_w]
    cp.Problem(objective, constraints).solve()
    return np.array(w.value)
