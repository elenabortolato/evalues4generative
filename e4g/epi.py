"""A stochastic renewal twin for daily COVID-19 ICU admissions.

Model (observation-driven renewal process, in the spirit of Cori et al. 2013):

    y_t | past ~ NegBin(mean mu_t, size k),
    mu_t = R_t * dow[weekday(t)] * sum_{s=1}^{S} w_s y_{t-s},
    log R_t = a + b * (stringency_{t-L} / 100 - 0.5),

where w is a discretised gamma generation-interval distribution (mean 6.5
days, sd 4, truncated at S = 21 days), dow are day-of-week reporting factors
(geometric mean 1), and the effect of restrictions enters with a lag of
L = 7 days. With L >= the block length, the policy that drives a 7-day block
is already observed when the block starts, so conditioning on it is exact.

The twin is calibrated by maximum likelihood on a window of data and then
frozen; the monitor asks whether later data remain consistent with it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import gamma

S, LAG = 21, 7
B_POLICY = -1.0   # log R falls by 0.9 between stringency 0 and 90 (R x 0.41)


def generation_weights(mean=6.5, sd=4.0, s=S):
    shape, scale = (mean / sd) ** 2, sd ** 2 / mean
    cdf = gamma.cdf(np.arange(s + 1), shape, scale=scale)
    w = np.diff(cdf)
    return w / w.sum()   # w[j] is the weight of lag j + 1


W = generation_weights()


@dataclass
class Twin:
    a: float
    b: float
    k: float
    dow: np.ndarray      # (7,) multiplicative, geometric mean 1

    def log_r(self, stringency_lagged):
        return self.a + self.b * (np.asarray(stringency_lagged) / 100.0 - 0.5)


def infectiousness(y, t):
    """sum_{s=1}^{S} w_s y_{t-s} for a history array y (index t is excluded)."""
    past = y[t - S:t][::-1]
    return float(np.dot(W, past))


def neg_loglik(theta, y, strg, wd, idx, b_fixed=None):
    a, b, logk = theta[:3]
    if b_fixed is not None:
        b = b_fixed
    dow = np.exp(np.r_[theta[3:], -theta[3:].sum()])
    k = np.exp(logk)
    lam = np.array([infectiousness(y, t) for t in idx])
    mu = np.exp(a + b * (strg[idx - LAG] / 100.0 - 0.5)) * dow[wd[idx]] * lam + 1e-8
    yt = y[idx]
    ll = (gammaln(yt + k) - gammaln(k) - gammaln(yt + 1)
          + k * np.log(k / (k + mu)) + yt * np.log(mu / (k + mu)))
    return -ll.sum()


def fit(y, strg, wd, start, end, b_fixed=None):
    """MLE on days start..end-1 (indices into the full arrays).

    The policy effect b is not identifiable from a short observational window
    (restrictions respond to the epidemic), so by default it is fixed at a
    value informed by the literature, and the remaining parameters are
    estimated. Pass b_fixed=None to estimate b as well (under the constraint b <= 0).
    """
    idx = np.arange(start, end)
    x0 = np.r_[0.0, -0.5, np.log(20.0), np.zeros(6)]
    # b <= 0: restrictions cannot increase transmission. Without the constraint,
    # short windows pick up reverse causation (restrictions tighten because
    # admissions rise) and give b > 0.
    bounds = [(-3, 3), (-3.0, 0.0), (np.log(1.0), np.log(1e4))] + [(-1, 1)] * 6
    res = minimize(neg_loglik, x0, args=(y, strg, wd, idx, b_fixed), method="L-BFGS-B",
                   bounds=bounds)
    th = res.x
    if b_fixed is not None:
        th[1] = b_fixed
    dow = np.exp(np.r_[th[3:], -th[3:].sum()])
    return Twin(a=th[0], b=th[1], k=float(np.exp(th[2])), dow=dow)


def simulate_block(twin, y, strg, wd, t0, days, m, rng, deterministic=False,
                   policy=None):
    """M simulated continuations of days t0..t0+days-1 given y[:t0].

    `policy` overrides the lagged stringency used by the twin (to build the
    baseline that does not condition on the policy actually in force).
    """
    hist = np.tile(y[:t0].astype(float), (m, 1))
    out = np.empty((m, days))
    for j in range(days):
        t = t0 + j
        s_lag = strg[t - LAG] if policy is None else policy
        lam = hist[:, -S:][:, ::-1] @ W
        mu = np.exp(twin.log_r(s_lag)) * twin.dow[wd[t]] * lam
        if deterministic:
            draw = mu
        else:
            p = twin.k / (twin.k + mu)
            draw = rng.negative_binomial(twin.k, np.clip(p, 1e-12, 1 - 1e-12))
        out[:, j] = draw
        hist = np.column_stack([hist, draw])
    return out
