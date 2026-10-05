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
      "predictive"  log-odds of the posterior predictive CLASS probability,
                    kappa * mu (probit approximation). The bet is shrunk
                    where the posterior is uncertain.
      "ratio"       log of the posterior predictive DENSITY RATIO,
                    log E_q[exp(eta)] = mu + v / 2 (exact for a Gaussian
                    posterior on the logit). The bet is inflated where the
                    posterior is uncertain.
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
        mean, var, kappa = self._moments(z)
        if self.mode == "predictive":
            s = kappa * mean
        elif self.mode == "ratio":
            s = mean + var / 2
        else:
            s = mean
        return np.clip(s, -30, 30)


class ChoiceBLR:
    """Bayesian learning rule on the simulation likelihood (Theorem 3).

    For a score theta' phi(x), the Monte Carlo factor of one step,

        E(theta) = (M+1) exp(theta' z_0) / sum_k exp(theta' z_k),

    with z_0 the features of the real outcome and z_1..z_M those of the
    simulations, is a conditional-logit likelihood ("which of the M+1 outcomes
    is the real one?") whose normaliser the simulations provide. A Gaussian
    posterior q = N(m, S^{-1}) over theta is updated at every step by one
    natural-gradient step of the Bayesian learning rule on the loss -log E:

        S_t = S_{t-1} + E_q[Cov_w(z)],   m_t = m_{t-1} - S_t^{-1} E_q[sum_k w_k z_k - z_0],

    where w_k = softmax(theta' z)_k. There is no intercept: it cancels in E.
    Expectations under q use fixed antithetic nodes theta_s = m + L xi_s
    (L L' = S^{-1}), so everything is a deterministic function of the past.

    Bets (mode):
      "mixture"  the posterior-predictive bet  mean_s E(theta_s), the
                 approximate version of the exact mixture of Theorem 3
                 (valid for any predictable q, Proposition 5)
      "mean"     the plug-in factor at the posterior mean, E(m)
      "shrunk"   the Monte Carlo factor of the probit-shrunk score
                 kappa(x) m'z(x), kappa = (1 + pi v(x) / 8)^{-1/2} with
                 v(x) = z(x)' S^{-1} z(x): the probability-scale predictive of
                 Section 5.5 applied to the choice posterior

    Features are standardised with statistics of simulator draws only
    (fit_scaler), fixed before any real data are seen.
    """

    def __init__(self, degree: int = 4, prior_var: float = 1.0, mode: str = "mixture",
                 n_nodes: int = 32, feature_map=None, seed: int = 0, eta: float = 1.0):
        self.eta = eta                    # learning rate of a generalised posterior
        self.degree = degree
        self.feature_map = feature_map or (lambda x: poly_features(x, degree))
        self.prior_var = prior_var
        self.mode = mode
        self.n_nodes = n_nodes
        self.seed = seed
        self.mu = self.sd = self.m = self.prec = self.xi = None

    def fit_scaler(self, x_sim: np.ndarray) -> "ChoiceBLR":
        f = self.feature_map(x_sim)
        self.mu, self.sd = f.mean(0), f.std(0) + 1e-8
        d = f.shape[1]
        self.m = np.zeros(d)
        self.prec = np.eye(d) / self.prior_var
        xi = np.random.default_rng(self.seed).standard_normal((self.n_nodes // 2, d))
        self.xi = np.vstack([xi, -xi])
        return self

    def _z(self, x):
        return (self.feature_map(np.atleast_2d(x)) - self.mu) / self.sd

    def _nodes(self):
        chol = np.linalg.cholesky(np.linalg.inv(self.prec))
        return self.m[None, :] + self.xi @ chol.T                 # (S, d)

    @staticmethod
    def _log_e(theta, z):
        """log E(theta) for thetas (S, d) and outcomes z (M+1, d), row 0 real."""
        s = theta @ z.T                                           # (S, M+1)
        smax = s.max(1, keepdims=True)
        lse = smax[:, 0] + np.log(np.exp(s - smax).sum(1))
        return np.log(z.shape[0]) + s[:, 0] - lse, s

    def factor(self, x_real: np.ndarray, x_sim: np.ndarray) -> float:
        z = self._z(np.vstack([x_real, x_sim]))
        if self.mode == "mean":
            le, _ = self._log_e(self.m[None, :], z)
            return float(np.exp(le[0]))
        if self.mode == "shrunk":
            v = np.einsum("ij,jk,ik->i", z, np.linalg.inv(self.prec), z)
            sc = (z @ self.m) / np.sqrt(1.0 + np.pi * v / 8.0)
            sc = sc - sc.max()
            return float(z.shape[0] * np.exp(sc[0]) / np.exp(sc).sum())
        le, _ = self._log_e(self._nodes(), z)
        return float(np.exp(le).mean())

    def score(self, x: np.ndarray) -> np.ndarray:
        """Posterior-mean score theta' phi(x) (defined up to a constant)."""
        return np.clip(self._z(x) @ self.m, -30, 30)

    def update(self, x_real: np.ndarray, x_sim: np.ndarray) -> "ChoiceBLR":
        z = self._z(np.vstack([x_real, x_sim]))
        theta = self._nodes()
        _, s = self._log_e(theta, z)
        w = np.exp(s - s.max(1, keepdims=True))
        w /= w.sum(1, keepdims=True)                              # (S, M+1)
        mean_z = w @ z                                            # (S, d)
        grad = (mean_z - z[0]).mean(0)
        second = np.einsum("sk,ki,kj->ij", w, z, z) / len(theta)
        hess = second - (mean_z.T @ mean_z) / len(theta)          # E_q[Cov_w(z)]
        self.prec = self.prec + self.eta * hess
        self.m = self.m - np.linalg.solve(self.prec, self.eta * grad)
        return self



class MartingalePredictive:
    """Predictive-first update of a classifier (parametric martingale posterior).

    Instead of a posterior over the weights, the learner carries the classifier's
    predictive itself, pi_t(z) = sigma(z' theta_t) for the logistic loss (real
    vs simulated rows) or softmax(z' theta_t) over the M+1 outcomes for the
    conditional-logit simulation likelihood, and updates it after every step
    by Fisher scoring on the predictive's own score (Fong, Holmes and Walker,
    2023, parametric martingale posteriors):

        I_t = I_{t-1} + J_t(theta_{t-1}),    theta_t = theta_{t-1} + I_t^{-1} s_t(theta_{t-1}),

    where s_t is the score of the new batch and J_t its Fisher information,
    which for both losses does not depend on the labels. Under the predictive,
    E[s_t] = 0, so theta_t, and hence the predictive, is a martingale: the
    update is coherent by construction. I_0 is the prior precision.

    Bets (mode):
      "plugin"    the Monte Carlo factor of the current predictive's score
      "resample"  the factor averaged over B predictive-resampling paths:
                  starting from (theta_{t-1}, I_{t-1}), future batches are drawn
                  by resampling past rows and drawing their labels from the
                  predictive, and the same update is run forward for H steps
                  (H = min(t, h_max)); the paths are refreshed every `refresh`
                  steps. This is the martingale posterior's uncertainty,
                  truncated at H steps, obtained from the predictive alone.
    """

    def __init__(self, loss: str = "logistic", mode: str = "plugin", prior_var: float = 1.0,
                 intercept_var: float = 10.0, n_paths: int = 16, h_max: int = 200,
                 refresh: int = 10, feature_map=None, degree: int = 4, seed: int = 0):
        self.loss, self.mode = loss, mode
        self.prior_var, self.intercept_var = prior_var, intercept_var
        self.n_paths, self.h_max, self.refresh = n_paths, h_max, refresh
        self.feature_map = feature_map or (lambda x: poly_features(x, degree))
        self.rng = np.random.default_rng(seed)
        self.mu = self.sd = self.theta = self.info = None
        self.past = []            # past batches of standardised designs, row 0 real
        self.paths = None
        self.t = 0

    def fit_scaler(self, x_sim: np.ndarray) -> "MartingalePredictive":
        f = self.feature_map(x_sim)
        self.mu, self.sd = f.mean(0), f.std(0) + 1e-8
        d = f.shape[1] + (1 if self.loss == "logistic" else 0)
        self.theta = np.zeros(d)
        diag = np.full(d, 1.0 / self.prior_var)
        if self.loss == "logistic":
            diag[0] = 1.0 / self.intercept_var
        self.info = np.diag(diag)
        return self

    def _design(self, x):
        z = (self.feature_map(np.atleast_2d(x)) - self.mu) / self.sd
        return np.column_stack([np.ones(len(z)), z]) if self.loss == "logistic" else z

    # score s and information J of one batch z (row 0 real), for thetas (B, d)
    def _score_info(self, theta, z, y=None):
        eta = theta @ z.T                                     # (B, n)
        n = z.shape[0]
        if self.loss == "logistic":
            w = np.r_[1.0, np.full(n - 1, 1.0 / (n - 1))]
            p = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
            yy = np.broadcast_to(np.r_[1.0, np.zeros(n - 1)], p.shape) if y is None else y
            s = ((w * (yy - p)) @ z)                          # (B, d)
            J = np.einsum("bk,ki,kj->bij", w * p * (1 - p), z, z)
            return s, J
        e = np.exp(eta - eta.max(1, keepdims=True))
        w = e / e.sum(1, keepdims=True)                       # (B, n)
        mz = w @ z
        real = np.zeros_like(w)
        if y is None:
            real[:, 0] = 1.0
        else:
            real[np.arange(len(w)), y] = 1.0
        s = real @ z - mz
        J = np.einsum("bk,ki,kj->bij", w, z, z) - np.einsum("bi,bj->bij", mz, mz)
        return s, J

    def _resample(self):
        """B predictive-resampling paths of length H from the current state."""
        b, h = self.n_paths, min(self.t, self.h_max)
        theta = np.repeat(self.theta[None, :], b, 0)
        info = np.repeat(self.info[None, :, :], b, 0)
        for _ in range(h):
            z = self.past[self.rng.integers(len(self.past))]
            eta = theta @ z.T
            if self.loss == "logistic":
                p = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
                y = (self.rng.random(p.shape) < p).astype(float)
            else:
                e = np.exp(eta - eta.max(1, keepdims=True))
                c = np.cumsum(e / e.sum(1, keepdims=True), 1)
                y = (self.rng.random((b, 1)) > c).sum(1)
            s, J = self._score_info(theta, z, y)
            info = info + J
            theta = theta + np.linalg.solve(info, s[:, :, None])[:, :, 0]
        self.paths = theta

    def _factor(self, theta, z):
        sc = theta @ z.T                                      # (B, n)
        sc = sc - sc.max(1, keepdims=True)
        e = np.exp(sc)
        return z.shape[0] * e[:, 0] / e.sum(1)

    def factor(self, x_real: np.ndarray, x_sim: np.ndarray) -> float:
        z = self._design(np.vstack([x_real, x_sim]))
        if self.mode == "resample" and self.t > 0:
            if self.paths is None or self.t % self.refresh == 0:
                self._resample()
            return float(self._factor(self.paths, z).mean())
        return float(self._factor(self.theta[None, :], z)[0])

    def score(self, x: np.ndarray) -> np.ndarray:
        return np.clip(self._design(x) @ self.theta, -30, 30)

    def update(self, x_real: np.ndarray, x_sim: np.ndarray) -> "MartingalePredictive":
        z = self._design(np.vstack([x_real, x_sim]))
        s, J = self._score_info(self.theta[None, :], z)
        self.info = self.info + J[0]
        self.theta = self.theta + np.linalg.solve(self.info, s[0])
        self.past.append(z)
        self.t += 1
        if self.mode == "resample" and self.t % self.refresh == 0:
            self.paths = None
        return self


class ForestMP:
    """Random forest as a martingale posterior under Fong's empirical predictive.

    The exchangeable units are the monitoring steps: step i contributes its
    real row (label 1, weight 1) and its M simulated rows (label 0, weight 1/M
    each), so the classes are balanced and a tree's log-odds estimates
    log q/p. Predictive resampling with the empirical (Polya urn) predictive,
    P_{n+1} = (n P_n + delta_{x_{n+1}}) / (n + 1), gives the Bayesian bootstrap
    (Fong, Holmes and Walker, 2023): a draw from the martingale posterior is the
    data reweighted by Dirichlet(1, ..., 1) weights. Each tree is fitted on the
    past steps with its own Dirichlet weights, so the forest is a sample from
    that martingale posterior, and the bet is the posterior-predictive mixture

        E_t = (1/B) sum_b E_t(tree_b),

    where E_t(tree_b) is the Monte Carlo factor of the tree's log-odds score
    (mode "mixture"), as in Theorem 3. A leaf's log-odds uses a Beta(a, a)
    prior on its weighted class counts, log((w1 + a) / (w0 + a)), so that
    leaves supported by little data bet little. Mode "forest" uses the Monte Carlo
    factor of the forest's average log-odds instead. Trees are refitted every
    `refit` steps on the past only (the score stays predictable); until the
    first fit the bet is one.
    """

    def __init__(self, n_trees: int = 16, max_depth: int = 6, min_leaf: float = 0.02,
                 refit: int = 25, burn_in: int = 20, max_steps: int = 400,
                 mode: str = "mixture", feature_map=None, seed: int = 0, leaf_prior: float = 3.0):
        self.n_trees, self.max_depth, self.min_leaf = n_trees, max_depth, min_leaf
        self.refit, self.burn_in, self.max_steps = refit, burn_in, max_steps
        self.mode, self.leaf_prior = mode, leaf_prior
        self.feature_map = feature_map or (lambda x: np.atleast_2d(x))
        self.rng = np.random.default_rng(seed)
        self.steps = []          # (real row (1, d), simulated rows (M, d))
        self.trees = None
        self.t = 0

    def fit_scaler(self, x_sim: np.ndarray) -> "ForestMP":
        return self              # trees are invariant to monotone rescaling

    def _fit(self):
        from sklearn.tree import DecisionTreeClassifier
        steps = self.steps[-self.max_steps:]
        xr = np.vstack([r for r, _ in steps])
        xs = np.vstack([s for _, s in steps])
        m = len(steps[0][1])
        x = np.vstack([xr, xs])
        y = np.r_[np.ones(len(xr)), np.zeros(len(xs))]
        step_of = np.r_[np.arange(len(steps)), np.repeat(np.arange(len(steps)), m)]
        base = np.r_[np.ones(len(xr)), np.full(len(xs), 1.0 / m)]
        self.trees = []
        for _ in range(self.n_trees):
            g = self.rng.dirichlet(np.ones(len(steps))) * len(steps)   # Bayesian bootstrap
            tree = DecisionTreeClassifier(max_depth=self.max_depth,
                                          min_weight_fraction_leaf=self.min_leaf,
                                          max_features="sqrt" if x.shape[1] > 3 else None,
                                          random_state=int(self.rng.integers(2 ** 31)))
            w = base * g[step_of]
            tree.fit(x, y, sample_weight=w)
            # leaf log-odds with a Beta(a, a) prior on the weighted class counts:
            # leaves supported by little weight bet little
            leaf = tree.apply(x)
            n = tree.tree_.node_count
            w1 = np.bincount(leaf, weights=w * y, minlength=n)
            w0 = np.bincount(leaf, weights=w * (1 - y), minlength=n)
            a = self.leaf_prior
            self.trees.append((tree, np.log((w1 + a) / (w0 + a))))

    def _scores(self, x):
        """Log-odds of every tree at rows x: array (B, n)."""
        x = np.asarray(x, dtype=np.float32)
        return np.array([table[tree.apply(x)] for tree, table in self.trees])

    def factor(self, x_real: np.ndarray, x_sim: np.ndarray) -> float:
        if self.trees is None:
            return 1.0
        z = self.feature_map(np.vstack([x_real, x_sim]))
        s = self._scores(z)                                       # (B, M+1)
        if self.mode == "forest":
            s = s.mean(0, keepdims=True)
        s = s - s.max(1, keepdims=True)
        e = np.exp(s)
        return float((z.shape[0] * e[:, 0] / e.sum(1)).mean())

    def score(self, x: np.ndarray) -> np.ndarray:
        if self.trees is None:
            return np.zeros(len(np.atleast_2d(x)))
        return self._scores(self.feature_map(np.atleast_2d(x))).mean(0)

    def update(self, x_real: np.ndarray, x_sim: np.ndarray) -> "ForestMP":
        self.steps.append((self.feature_map(np.atleast_2d(x_real)),
                           self.feature_map(np.atleast_2d(x_sim))))
        self.t += 1
        if self.t >= self.burn_in and self.t % self.refit == 0:
            self._fit()
        return self
