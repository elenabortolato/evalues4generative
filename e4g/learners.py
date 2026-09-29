"""Online density-ratio estimation by classification (LFIRE-style).

A logistic regression on polynomial features is trained to separate real
observations (label 1) from simulator draws (label 0), with balanced class
weights, so its logit estimates log p_real(x) / p_sim(x). It is refitted only on
data available *before* the step at which it is used, which makes the resulting
score predictable -- the property the Monte Carlo e-value needs.
"""

from __future__ import annotations

import itertools

import numpy as np


def poly_features(x: np.ndarray, degree: int) -> np.ndarray:
    """All monomials of total degree 1..degree (no intercept)."""
    x = np.atleast_2d(x)
    n, d = x.shape
    cols = []
    for k in range(1, degree + 1):
        for idx in itertools.combinations_with_replacement(range(d), k):
            cols.append(np.prod(x[:, idx], axis=1))
    return np.column_stack(cols)


class OnlineRatio:
    """Balanced, ridge-penalised logistic regression, refitted by Newton steps."""

    def __init__(self, degree: int = 2, ridge: float = 1.0, newton_steps: int = 8,
                 feature_map=None):
        self.degree = degree
        self.ridge = ridge
        self.newton_steps = newton_steps
        self.feature_map = feature_map or (lambda x: poly_features(x, degree))
        self.beta = None
        self.mu = None
        self.sd = None

    def _design(self, x):
        f = self.feature_map(x)
        f = (f - self.mu) / self.sd
        return np.column_stack([np.ones(len(f)), f])

    def fit(self, x_real: np.ndarray, x_sim: np.ndarray) -> "OnlineRatio":
        x = np.concatenate([x_real, x_sim])
        y = np.concatenate([np.ones(len(x_real)), np.zeros(len(x_sim))])
        f = self.feature_map(x)
        self.mu = f.mean(0)
        self.sd = f.std(0) + 1e-8
        z = np.column_stack([np.ones(len(f)), (f - self.mu) / self.sd])
        n1, n0 = len(x_real), len(x_sim)
        w = np.where(y == 1, 0.5 * len(y) / n1, 0.5 * len(y) / n0)
        pen = np.full(z.shape[1], self.ridge)
        pen[0] = 0.0
        beta = np.zeros(z.shape[1])  # standardisation changes, so no warm start
        for _ in range(self.newton_steps):
            eta = np.clip(z @ beta, -30, 30)
            p = 1.0 / (1.0 + np.exp(-eta))
            grad = z.T @ (w * (p - y)) + pen * beta
            hess = (z * (w * p * (1 - p))[:, None]).T @ z + np.diag(pen + 1e-8)
            beta -= np.linalg.solve(hess, grad)
        self.beta = beta
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        """Estimated log density ratio; 0 (no bet) before the first fit."""
        x = np.atleast_2d(x)
        if self.beta is None:
            return np.zeros(len(x))
        return np.clip(self._design(x) @ self.beta, -30, 30)
