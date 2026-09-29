"""Experiment 4: numerical check of the growth-rate theorem.

Null P = N(0, 1), alternative Q = N(delta, 1), score T(x) = exp(theta x), so the
tilted null Q_hat = T P / E_P[T] is N(theta, 1). theta = delta is a perfect
learner; theta != delta a misspecified one. Closed forms:

  KL(Q||P) = delta^2 / 2,   KL(Q||Q_hat) = (delta - theta)^2 / 2,
  rho = E_Q[T] / E_P[T] = exp(theta delta).

Theorem (see the proposal):
  (ii) M = infinity:  g_inf = KL(Q||P) - KL(Q||Q_hat) = theta delta - theta^2/2
  (i)  ceiling:      g_M <= KL(Q||P)
  (iii) finite M:    g_M >= g_inf - (rho - 1) / (M + 1)
  (iv) locally:      g_M ~ M / (M + 1) * g_inf

g_M = E_Q[log E_M] is estimated by Monte Carlo with common random numbers.
Output: results/exp4.json, figures/exp4_growth_rate.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MS = [1, 2, 4, 8, 16, 64, 256]
N = 400_000


def growth(delta, theta, m, rng):
    x = rng.normal(delta, 1.0, N)
    y = rng.normal(0.0, 1.0, (N, m))
    s_x, s_y = theta * x, theta * y
    # log E = log(M+1) + s_x - log(exp(s_x) + sum_j exp(s_j))
    log_den = logsumexp(np.column_stack([s_x, s_y]), axis=1)
    le = np.log(m + 1) + s_x - log_den
    return float(le.mean()), float(le.std() / np.sqrt(N))


def main():
    rng = np.random.default_rng(0)
    rows = []
    for delta in (0.25, 1.0):
        for theta in (delta, delta / 2, 1.5 * delta, 2 * delta):
            g_inf = theta * delta - theta ** 2 / 2
            rho = np.exp(theta * delta)
            for m in MS:
                g, se = growth(delta, theta, m, rng)
                rows.append(dict(delta=delta, theta=theta, m=m, g=g, se=se, g_inf=g_inf,
                                 kl=delta ** 2 / 2, bound=g_inf - (rho - 1) / (m + 1),
                                 local=m / (m + 1) * g_inf))
    print(f"{'delta':>5} {'theta':>5} {'M':>4} {'g_M (MC)':>12} {'bound (b)':>10} "
          f"{'M/(M+1) g_inf':>14} {'g_inf':>8} {'KL':>7} {'eff':>5}")
    ok, ok_ceiling = True, True
    for r in rows:
        ok &= r["g"] + 3 * r["se"] >= r["bound"]
        ok_ceiling &= r["g"] - 3 * r["se"] <= r["kl"]
        print(f"{r['delta']:5.2f} {r['theta']:5.3f} {r['m']:4d} {r['g']:9.5f}±{r['se']:.0e}"
              f" {r['bound']:10.5f} {r['local']:14.5f} {r['g_inf']:8.5f} {r['kl']:7.4f}"
              f" {r['g'] / r['g_inf'] if r['g_inf'] > 1e-12 else float('nan'):5.2f}")
    print("lower bound (iii) holds in every row (within 3 s.e.):", bool(ok))
    print("ceiling (i) g_M <= KL holds in every row (within 3 s.e.):", bool(ok_ceiling))
    (ROOT / "results" / "exp4.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
