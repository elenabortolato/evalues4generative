"""Experiment 12: numerical check of the predictive-update theorem (Theorem 3).

Null P = N(0, 1). Features phi(x) = (clip(x, -3, 3), clip(x^2 - 1, -3, 3)) / 3,
so ||phi|| <= G = sqrt(2) and d = 2. Scores theta' phi with theta in the disc
Theta of radius D = 3, uniform prior on Theta. M = 16 simulations per step.

For each step, E_t(theta) = (M+1) exp(theta'phi(X_t)) / sum_k exp(theta'phi(Z_tk))
on a fine grid over Theta. The mixture wealth W_t = int prod_i E_i(theta) dpi_0 is
the product of posterior-predictive bets (Theorem 3(i)); it is computed on the
grid. The hindsight optimum sup_theta sum_i log E_i(theta) is computed by convex
optimisation (the objective is concave in theta). Theorem 3(ii) states that

   regret_t = sup_theta sum log E_i(theta) - log W_t <= d log t + d log(4 e G D / d)

on every path; with an interior optimum the slope in log t is d/2. Scenarios: null (X ~ P), mean shift N(0.5, 1), scale N(0, 1.5^2),
and a drift from N(0, 1) to N(0.6, 1.3^2) at t = 500.

Output: results/exp12.json
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

M, D, H = 16, 3.0, 0.03
G, DIM = np.sqrt(2.0), 2
T_MAX = 2000
CHECK = [10, 30, 100, 300, 1000, 2000]
ax = np.arange(-D, D + H / 2, H)
TH = np.array(np.meshgrid(ax, ax)).reshape(2, -1).T
TH = TH[(TH ** 2).sum(1) <= D ** 2]          # grid over the disc Theta


def phi(x):
    return np.stack([np.clip(x, -3, 3), np.clip(x ** 2 - 1, -3, 3)], -1) / 3.0


def draw(rng, scenario, t):
    if scenario == "null":
        return rng.normal()
    if scenario == "shift":
        return rng.normal(0.5, 1.0)
    if scenario == "scale":
        return rng.normal(0.0, 1.5)
    return rng.normal(0.6, 1.3) if t >= 500 else rng.normal()   # drift


def log_e(theta, f0, fz):
    """log E(theta) for thetas (n, 2), real features f0 (2,), sims fz (M, 2)."""
    s0 = theta @ f0
    sz = theta @ fz.T
    return np.log(M + 1) + s0 - logsumexp(np.column_stack([s0, sz]), axis=1)


def hindsight(f0s, fzs, start):
    def neg(th):
        th = th[None, :]
        s0 = (f0s @ th[0])
        sz = fzs @ th[0]
        v = np.log(M + 1) + s0 - logsumexp(np.column_stack([s0, sz]), axis=1)
        return -v.sum()
    cons = {"type": "ineq", "fun": lambda th: D ** 2 - th @ th}
    res = minimize(neg, start, method="SLSQP", constraints=[cons])
    return -res.fun


def run(seed, scenario):
    rng = np.random.default_rng(seed)
    F = np.zeros(len(TH))
    f0s, fzs, out = [], [], []
    for t in range(1, T_MAX + 1):
        f0 = phi(np.array(draw(rng, scenario, t)))
        fz = phi(rng.normal(size=M))
        F += log_e(TH, f0, fz)
        f0s.append(f0)
        fzs.append(fz)
        if t in CHECK:
            log_w = logsumexp(F) - np.log(len(TH))
            best = hindsight(np.array(f0s), np.array(fzs), TH[F.argmax()])
            best = max(best, F.max())
            bound = DIM * np.log(t) + DIM * np.log(4 * np.e * G * D / DIM)
            out.append(dict(t=t, log_w=float(log_w), best=float(best),
                            regret=float(best - log_w), bound=float(bound)))
    return out


def main():
    res = {}
    for sc in ("null", "shift", "scale", "drift"):
        with ProcessPoolExecutor(4) as ex:
            runs = list(ex.map(partial(run, scenario=sc), range(12)))
        rows = []
        for j, t in enumerate(CHECK):
            reg = np.array([r[j]["regret"] for r in runs])
            lw = np.array([r[j]["log_w"] for r in runs])
            rows.append(dict(t=t, regret_mean=float(reg.mean()), regret_max=float(reg.max()),
                             bound=runs[0][j]["bound"], log_w_mean=float(lw.mean()),
                             best_mean=float(np.mean([r[j]["best"] for r in runs])),
                             frac_log_w_above_log20=float((lw >= np.log(20)).mean())))
        res[sc] = rows
        print(sc)
        for r in rows:
            print("  t=%5d  regret mean %6.2f max %6.2f  bound %6.2f   log W %8.2f  best %8.2f"
                  % (r["t"], r["regret_mean"], r["regret_max"], r["bound"], r["log_w_mean"],
                     r["best_mean"]), flush=True)
        slope = (rows[-1]["regret_mean"] - rows[2]["regret_mean"]) / np.log(CHECK[-1] / CHECK[2])
        res[sc + "_slope_per_log_t"] = float(slope)
        print("  regret slope in log t between t=100 and t=2000: %.2f (d/2 = %.1f)" % (slope, DIM / 2))
        ok = all(r["regret_max"] <= r["bound"] for r in rows)
        print("  bound holds on every path and time:", ok, flush=True)
    (ROOT / "results" / "exp12.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
