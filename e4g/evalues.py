"""E-value factors and betting processes.

All factors are computed from *scores* s = log T, so that T = exp(s) can be an
arbitrary positive statistic (typically a learned log density ratio) without
overflow.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


def mc_factor(s_obs: float, s_sims: np.ndarray) -> float:
    """Monte Carlo e-value (Proposition 2 of the proposal).

    E = (M+1) T(X) / (T(X) + sum_j T(Y_j)),   T = exp(s).

    If X, Y_1..Y_M are exchangeable under H0 -- the observation and M draws
    from the null simulator, all conditionally on the past -- then E[E] = 1
    exactly, for ANY score function fixed before X and the Y_j are drawn.
    No null density is evaluated.
    """
    s_sims = np.asarray(s_sims, dtype=float)
    m = s_sims.size
    # (M+1) / (1 + sum_j exp(s_j - s_obs)), in the log domain
    log_den = logsumexp(np.concatenate(([0.0], s_sims - s_obs)))
    return float(np.exp(np.log(m + 1) - log_den))


def plugin_factor(s_obs: float) -> float:
    """Naive plug-in: use the learned ratio exp(s) itself as the bet.

    Valid only if E_H0[exp(s(X))] <= 1, which a learned ratio does not
    guarantee. Included as the baseline the proposal warns against.
    """
    return float(np.exp(s_obs))


class LambdaMixture:
    """Average of the test martingales prod_t (1 - lam + lam * E_t) over a grid.

    Each factor 1 - lam + lam * E_t has conditional null mean <= 1 whenever
    E_t does, so each product is a test supermartingale, and an average of
    test supermartingales is one too. The mixture hedges the betting fraction
    instead of tuning it (a small instance of the proposal's "mixtures in the
    numerator").
    """

    def __init__(self, lams=np.linspace(0.05, 1.0, 20)):
        self.lams = np.asarray(lams, dtype=float)
        self.log_w = np.zeros_like(self.lams)

    def update(self, e: float) -> float:
        # clip only against log(0); factors are >= 1 - lam >= 0
        self.log_w += np.log(np.maximum(1.0 - self.lams + self.lams * e, 1e-300))
        return self.value

    @property
    def value(self) -> float:
        return float(np.exp(min(logsumexp(self.log_w) - np.log(self.lams.size), 700.0)))


class ProductProcess:
    """Plain product prod_t E_t (used for the oracle likelihood ratio)."""

    def __init__(self):
        self.log_w = 0.0

    def update(self, e: float) -> float:
        self.log_w += np.log(max(e, 1e-300))
        return self.value

    @property
    def value(self) -> float:
        return float(np.exp(min(self.log_w, 700.0)))
