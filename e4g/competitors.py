"""Score learners of two concurrent papers, for a head-to-head comparison.

Both papers use the same normalised Monte Carlo bet as this package,

    E(theta) = (M+1) exp(s(x_0)) / sum_k exp(s(x_k)),

and differ from it in how the score s is chosen. Each class below exposes
``factor(x_real, x_sim)`` (the bet, computed from the past only) and
``update(x_real, x_sim)`` (called after the bet), so it plugs into
``e4g.Monitor`` like the package's own learners.

GrowthFit -- Forre and Brenner (2026), "Anytime-valid simulation-based
    hypothesis testing", arXiv:2610.08210.
    mode="online"      Section 4: the score theta' z(x) is refitted on the past
                       real observations and their stored simulations by
                       maximising the empirical growth  sum_i log E_i(theta),
                       with a ridge penalty (their eq. 95; the penalty here is
                       ||theta||^2 / (2 tau)), and the bet uses the fitted
                       score (a plug-in at the maximiser).
    mode="pretrained"  Section 3: the score is fitted once, before monitoring,
                       on simulations from a guessed alternative P1 against
                       the null, and then kept fixed.
    The objective is concave, so a few Newton steps from the previous
    solution find the maximiser.

ParametricPlugin -- Cortinovis, Battaglia and Caron (2026), "SLICE",
    arXiv:2610.13855, for testing one null parameter: the score is the ratio
    p_{theta_hat}(x) / p_{theta_0}(x) between the model at a predictable
    estimate theta_hat (from past real data) and the null. SLICE estimates
    this ratio with an amortised neural network; here it is computed exactly,
    which is the best case for their method. It needs a parametric family
    that contains the alternative, supplied as ``log_ratio(x, est)`` and
    ``estimate(past_real)``.
"""

from __future__ import annotations

import numpy as np

from .learners import poly_features


def _log_e(theta, z):
    """log E(theta) of one step: z = features (M+1, d), row 0 real."""
    s = z @ theta
    smax = s.max()
    return np.log(len(s)) + s[0] - smax - np.log(np.exp(s - smax).sum())


class GrowthFit:
    def __init__(self, degree: int = 4, tau: float = 0.03, mode: str = "online",
                 refit_every: int = 10, burn_in: int = 20, newton_steps: int = 5,
                 feature_map=None):
        self.feature_map = feature_map or (lambda x: poly_features(x, degree))
        self.tau, self.mode = tau, mode
        self.refit_every, self.burn_in, self.newton_steps = refit_every, burn_in, newton_steps
        self.mu = self.sd = self.theta = None
        self.blocks: list[np.ndarray] = []
        self.t = 0

    def fit_scaler(self, x_sim: np.ndarray) -> "GrowthFit":
        f = self.feature_map(x_sim)
        self.mu, self.sd = f.mean(0), f.std(0) + 1e-8
        self.theta = np.zeros(f.shape[1])
        return self

    def _z(self, x):
        return (self.feature_map(np.atleast_2d(x)) - self.mu) / self.sd

    def _objective(self, th, blocks):
        return sum(_log_e(th, z) for z in blocks) - th @ th / (2 * self.tau)

    def _newton(self, blocks, steps):
        """Damped Newton with backtracking on the concave penalised growth."""
        th = self.theta.copy()
        d = len(th)
        obj = self._objective(th, blocks)
        for _ in range(steps):
            g = -th / self.tau
            h = -np.eye(d) / self.tau
            for z in blocks:
                s = z @ th
                w = np.exp(s - s.max())
                w /= w.sum()
                mz = w @ z
                g += z[0] - mz
                h -= (z * w[:, None]).T @ z - np.outer(mz, mz)
            step = -np.linalg.solve(h, g)
            slope = g @ step
            if slope < 1e-10:
                break
            lr = 1.0
            while lr > 1e-6:
                new = self._objective(th + lr * step, blocks)
                if new >= obj + 1e-4 * lr * slope:
                    break
                lr /= 2
            th, obj = th + lr * step, new
        self.theta = th

    def pretrain(self, x_alt: np.ndarray, x_null: np.ndarray, m: int) -> "GrowthFit":
        """Fit once on pairs (one draw of the alternative, m of the null)."""
        n = len(x_alt)
        blocks = [self._z(np.vstack([x_alt[i:i + 1], x_null[i * m:(i + 1) * m]]))
                  for i in range(n)]
        self._newton(blocks, 50)
        return self

    def factor(self, x_real: np.ndarray, x_sim: np.ndarray) -> float:
        z = self._z(np.vstack([x_real, x_sim]))
        return float(np.exp(_log_e(self.theta, z)))

    def score(self, x: np.ndarray) -> np.ndarray:
        return np.clip(self._z(x) @ self.theta, -30, 30)

    def update(self, x_real: np.ndarray, x_sim: np.ndarray) -> "GrowthFit":
        if self.mode == "pretrained":
            return self
        self.t += 1
        self.blocks.append(self._z(np.vstack([x_real, x_sim])))
        if self.t >= self.burn_in and self.t % self.refit_every == 0:
            self._newton(self.blocks, self.newton_steps)
        return self


class ParametricPlugin:
    def __init__(self, log_ratio, estimate):
        self.log_ratio, self.estimate = log_ratio, estimate
        self.past: list[np.ndarray] = []
        self.est = None

    def factor(self, x_real: np.ndarray, x_sim: np.ndarray) -> float:
        if self.est is None:
            return 1.0
        lr = self.log_ratio(np.vstack([x_real, x_sim]), self.est)
        lr = np.clip(np.nan_to_num(lr, nan=-30.0, neginf=-30.0, posinf=30.0), -30, 30)
        lr = lr - lr.max()
        return float(len(lr) * np.exp(lr[0]) / np.exp(lr).sum())

    def update(self, x_real: np.ndarray, x_sim: np.ndarray) -> "ParametricPlugin":
        self.past.append(np.atleast_2d(x_real)[0])
        self.est = self.estimate(np.asarray(self.past))
        return self


class NeuralRatio:
    """Amortised likelihood-to-evidence network, as in SLICE (Section 3.2).

    A classifier separates joint pairs (theta, y), y ~ P_theta, from pairs
    whose theta is shuffled; its logit estimates log p_theta(y) / m(y)
    (neural ratio estimation, NRE-A). It is trained once, before monitoring,
    on simulations only. ``log_ratio(x, est)`` returns
    logit(est, x) - logit(theta_0, x), the score of SLICE for testing the
    single null theta_0 against the predictable estimate ``est``.

    ``sample(rng, n)`` must return (theta (n, p), y (n, q)) from the prior
    over theta and the simulator; ``features(theta, y)`` builds the network
    input. Training cost (simulations, seconds) is recorded.
    """

    def __init__(self, sample, features, theta0, n_pairs=200_000, hidden=(128, 128, 128),
                 seed=0, max_iter=40):
        import time
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler
        rng = np.random.default_rng(seed)
        t0 = time.time()
        th, y = sample(rng, n_pairs)
        th_neg = th[rng.permutation(n_pairs)]
        x = np.vstack([features(th, y), features(th_neg, y)])
        lab = np.r_[np.ones(n_pairs), np.zeros(n_pairs)]
        self.scaler = StandardScaler().fit(x)
        self.net = MLPClassifier(hidden_layer_sizes=hidden, activation="relu", alpha=1e-5,
                                 batch_size=1024, learning_rate_init=5e-4, max_iter=max_iter,
                                 early_stopping=True, n_iter_no_change=5, random_state=seed)
        self.net.fit(self.scaler.transform(x), lab)
        self.features, self.theta0 = features, np.atleast_1d(theta0)
        self.cost = dict(simulations=int(n_pairs), seconds=round(time.time() - t0, 1),
                         epochs=int(self.net.n_iter_))

    def logit(self, theta, y):
        """theta: one parameter (p,), or one row per outcome (len(y), p)."""
        th = np.atleast_2d(theta)
        if len(th) != len(y):
            th = np.repeat(th[:1], len(y), axis=0)
        p = self.net.predict_proba(self.scaler.transform(self.features(th, y)))[:, 1]
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return np.log(p) - np.log1p(-p)
