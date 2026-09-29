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


class BLRRatio:
    """Online Bayesian ratio estimator updated by the Bayesian learning rule.

    Model: the same balanced logistic regression "real (1) vs simulator (0)"
    as OnlineRatio, whose logit estimates log q(x)/p(x). Instead of refitting a
    point estimate, a Gaussian posterior N(m, S^{-1}) over the weights is updated
    at EVERY step by one natural-gradient step of the Bayesian learning rule in
    its online form (Khan & Rue 2023; BONG, Jones, Chang & Murphy 2024): starting
    from the previous posterior, with the expected gradient and Hessian of the
    new batch's log-likelihood,

        S_t = S_{t-1} + E_q[H_t],       m_t = m_{t-1} - S_t^{-1} E_q[g_t].

    For a Gaussian likelihood this is exact Bayes (the Kalman filter); for the
    logistic likelihood it is one-step variational online Newton. Expectations
    under q use the probit approximation E_q[sigmoid(eta)] ~ sigmoid(kappa mu),
    kappa = (1 + pi v / 8)^{-1/2}, with mu, v the mean and variance of eta.

    Scores (mode):
      "predictive"  log-odds of the posterior predictive, kappa * mu. The bet is
                    shrunk automatically where the posterior is uncertain.
      "mean"        plug-in posterior mean, mu (no shrinkage), for comparison.

    Features are standardised with statistics from simulator draws only
    (fit_scaler), fixed before any real data are seen, so the score stays
    predictable and the posterior is not disturbed by a moving scale.
    """

    def __init__(self, degree: int = 4, prior_var: float = 1.0, intercept_var: float = 10.0,
                 mode: str = "predictive", feature_map=None):
        self.degree = degree
        self.feature_map = feature_map or (lambda x: poly_features(x, degree))
        self.prior_var = prior_var
        self.intercept_var = intercept_var
        self.mode = mode
        self.mu = self.sd = None
        self.m = self.prec = self.cov = None

    def fit_scaler(self, x_sim: np.ndarray) -> "BLRRatio":
        f = self.feature_map(x_sim)
        self.mu, self.sd = f.mean(0), f.std(0) + 1e-8
        d = f.shape[1] + 1
        self.m = np.zeros(d)
        self.prec = np.diag(np.r_[1 / self.intercept_var, np.full(d - 1, 1 / self.prior_var)])
        self.cov = np.linalg.inv(self.prec)
        return self

    def _design(self, x):
        f = (self.feature_map(np.atleast_2d(x)) - self.mu) / self.sd
        return np.column_stack([np.ones(len(f)), f])

    def _moments(self, z):
        mean = z @ self.m
        var = np.einsum("ij,jk,ik->i", z, self.cov, z)
        kappa = 1.0 / np.sqrt(1.0 + np.pi * var / 8.0)
        return mean, var, kappa

    def update(self, x_real: np.ndarray, x_sim: np.ndarray) -> "BLRRatio":
        """One BLR step on a new batch: real rows (label 1) and simulator rows (0).

        Classes are balanced within the batch (real weight 1/n1, sims 1/n0, times
        the batch's real count), so the logit keeps estimating log q/p.
        """
        z = self._design(np.vstack([x_real, x_sim]))
        n1, n0 = len(x_real), len(x_sim)
        y = np.r_[np.ones(n1), np.zeros(n0)]
        w = np.r_[np.ones(n1), np.full(n0, n1 / n0)]
        mean, _, kappa = self._moments(z)
        p = 1.0 / (1.0 + np.exp(-np.clip(kappa * mean, -30, 30)))
        grad = z.T @ (w * (p - y))
        hess = (z * (w * kappa * p * (1 - p))[:, None]).T @ z
        self.prec = self.prec + hess
        self.cov = np.linalg.inv(self.prec)
        self.m = self.m - self.cov @ grad
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        z = self._design(x)
        mean, _, kappa = self._moments(z)
        s = kappa * mean if self.mode == "predictive" else mean
        return np.clip(s, -30, 30)
