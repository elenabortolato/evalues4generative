"""Multi-region renewal twin with colour-zone actions (Italy, 2020-2022).

For region r and day t, daily new positives follow

    y_rt | past ~ NegBin(mean mu_rt, size k),
    mu_rt = exp(a_r + beta[colour_{r,t-7}]) * dow[weekday(t)] * sum_s w_s y_{r,t-s},

with a region-specific baseline a_r, colour effects beta relative to yellow
(shared across regions), shared day-of-week factors and overdispersion. The
colour assigned to a region acts with a 7-day lag, so a 7-day block is driven
by colours that were already decided when the block starts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

from .epi import LAG, S, W

COLOURS = ["white", "yellow", "orange", "red"]
CODE = {c: i for i, c in enumerate(COLOURS)}

# approximate resident populations (millions, ISTAT, 1 January 2021), used only
# by the synthetic tier rule
POP = {"Abruzzo": 1.28, "Basilicata": 0.55, "Calabria": 1.86, "Campania": 5.62,
       "Emilia-Romagna": 4.44, "Friuli Venezia Giulia": 1.20, "Lazio": 5.73,
       "Liguria": 1.51, "Lombardia": 9.98, "Marche": 1.50, "Molise": 0.30,
       "P.A. Bolzano": 0.53, "P.A. Trento": 0.54, "Piemonte": 4.27, "Puglia": 3.93,
       "Sardegna": 1.59, "Sicilia": 4.83, "Toscana": 3.69, "Umbria": 0.86,
       "Valle d'Aosta": 0.12, "Veneto": 4.87}


def load(path):
    d = pd.read_csv(path, parse_dates=["date"])
    regions = sorted(d.region.unique())
    dates = np.sort(d.date.unique())
    y = d.pivot(index="region", columns="date", values="new_positives").loc[regions].values
    y = np.clip(np.nan_to_num(y), 0, None).astype(float)
    col = d.pivot(index="region", columns="date", values="colour").loc[regions]
    c = col.apply(lambda s: s.map(CODE)).values
    c = np.where(np.isnan(c.astype(float)), CODE["yellow"], c).astype(int)
    wd = pd.DatetimeIndex(dates).weekday.values
    return regions, pd.DatetimeIndex(dates), y, c, wd


def infectiousness(y):
    """Lambda[r, t] = sum_{s=1}^{S} w_s y[r, t-s] (NaN for t < S)."""
    lam = np.full_like(y, np.nan)
    for t in range(S, y.shape[1]):
        lam[:, t] = y[:, t - S:t][:, ::-1] @ W
    return lam


@dataclass
class Twin:
    a: np.ndarray        # (R,)
    beta: np.ndarray     # (4,) white, yellow (= 0), orange, red
    k: float
    dow: np.ndarray      # (7,)
    sigma: float = 0.0   # sd of a week-level random effect on log R


def _unpack(theta, R):
    """theta = (a_1..a_R, beta_white, u_orange, u_red, log k, dow_1..dow_6).

    Ordered effects: beta_orange = -u_orange, beta_red = beta_orange - u_red,
    with u >= 0, so red is at least as restrictive as orange, and orange as
    yellow. Without the ordering, reverse causation (regions turn red because
    cases grow) makes red look weaker than orange.
    """
    a = theta[:R]
    beta = np.array([theta[R], 0.0, -theta[R + 1], -theta[R + 1] - theta[R + 2]])
    k = np.exp(theta[R + 3])
    d6 = theta[R + 4:R + 10]
    dow = np.exp(np.r_[d6, -d6.sum()])
    return a, beta, k, dow


def fit(y, c, wd, start, end, white_as_yellow=True):
    """Pooled MLE over regions on days start..end-1."""
    R = y.shape[0]
    lam = infectiousness(y)
    idx = np.arange(start, end)
    yy, ll, cc, dd = y[:, idx], lam[:, idx], c[:, idx - LAG], wd[idx]

    def nll(theta):
        a, beta, k, dow = _unpack(theta, R)
        if white_as_yellow:
            beta = beta.copy()
            beta[0] = 0.0
        mu = np.exp(a[:, None] + beta[cc]) * dow[dd][None, :] * ll + 1e-8
        out = (gammaln(yy + k) - gammaln(k) - gammaln(yy + 1)
               + k * np.log(k / (k + mu)) + yy * np.log(mu / (k + mu)))
        return -out.sum()

    x0 = np.r_[np.zeros(R), 0.0, 0.1, 0.1, np.log(20.0), np.zeros(6)]
    bounds = ([(-3, 3)] * R + [(0.0, 3.0), (0.0, 3.0), (0.0, 3.0),
                               (np.log(1.0), np.log(1e4))] + [(-1, 1)] * 6)
    res = minimize(nll, x0, method="L-BFGS-B", bounds=bounds)
    a, beta, k, dow = _unpack(res.x, R)
    if white_as_yellow:
        beta[0] = 0.0
    return Twin(a=a, beta=beta, k=float(k), dow=dow), res


def refit_region(twin, y, c, wd, r, start, end):
    """Re-estimate the baseline a_r of one region on days start..end-1."""
    lam = infectiousness(y[r:r + 1])[0]
    idx = np.arange(start, end)

    def nll(a):
        mu = np.exp(a + twin.beta[c[r, idx - LAG]]) * twin.dow[wd[idx]] * lam[idx] + 1e-8
        yy, k = y[r, idx], twin.k
        return -(gammaln(yy + k) - gammaln(k) - gammaln(yy + 1)
                 + k * np.log(k / (k + mu)) + yy * np.log(mu / (k + mu))).sum()

    res = minimize(lambda v: nll(v[0]), [twin.a[r]], method="L-BFGS-B", bounds=[(-3, 3)])
    a = twin.a.copy()
    a[r] = res.x[0]
    return Twin(a=a, beta=twin.beta, k=twin.k, dow=twin.dow, sigma=twin.sigma)


def simulate_block(twin, r, y_r, c_r, wd, t0, days, m, rng, deterministic=False,
                   colour=None, a_shift=0.0):
    """M continuations of region r for days t0..t0+days-1 given y_r[:t0].

    `colour` overrides the lagged colour (baseline that ignores the action);
    `a_shift` adds to log R (used to generate a synthetic variant).
    """
    hist = np.tile(y_r[:t0].astype(float), (m, 1))
    out = np.empty((m, days))
    # week-level shock to transmission, shared by the days of the block
    eps = 0.0 if deterministic else twin.sigma * rng.standard_normal(m)
    for j in range(days):
        t = t0 + j
        col = c_r[t - LAG] if colour is None else colour
        lam = hist[:, -S:][:, ::-1] @ W
        mu = np.exp(twin.a[r] + a_shift + twin.beta[col] + eps) * twin.dow[wd[t]] * lam
        if deterministic:
            draw = mu
        else:
            p = twin.k / (twin.k + mu)
            draw = rng.negative_binomial(twin.k, np.clip(p, 1e-12, 1 - 1e-12)).astype(float)
        out[:, j] = draw
        hist = np.column_stack([hist, draw])
    return out


def calibrate_sigma(twin, y, c, wd, start, end, grid=np.arange(0.0, 0.61, 0.05),
                    m=200, seed=0, target=0.9):
    """Choose the week-level shock sd so that one-week-ahead 90% predictive
    intervals of weekly totals cover `target` of region-weeks in the
    calibration window. Uses calibration data only."""
    rng = np.random.default_rng(seed)
    weeks = range(start, end - 7 + 1, 7)
    best, best_gap, table = 0.0, np.inf, {}
    for sg in grid:
        tw = Twin(a=twin.a, beta=twin.beta, k=twin.k, dow=twin.dow, sigma=float(sg))
        hits = []
        for t0 in weeks:
            for r in range(y.shape[0]):
                sims = simulate_block(tw, r, y[r], c[r], wd, t0, 7, m, rng).sum(1)
                lo, hi = np.percentile(sims, [5, 95])
                hits.append(lo <= y[r, t0:t0 + 7].sum() <= hi)
        cov = float(np.mean(hits))
        table[round(float(sg), 2)] = cov
        if abs(cov - target) < best_gap:
            best, best_gap = float(sg), abs(cov - target)
    return Twin(a=twin.a, beta=twin.beta, k=twin.k, dow=twin.dow, sigma=best), table
