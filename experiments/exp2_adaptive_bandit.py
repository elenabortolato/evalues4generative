"""Experiment 2: validity under adaptive allocation (Propositions 1 and 2).

A 3-arm bandit is run by Thompson sampling on the *real* outcomes, so the
allocation depends on the data. A black-box "digital twin" simulates the
outcome of any arm:

    y = mu_k + (exp(eps / 2) - e^{1/8}),   eps ~ N(0, 1),   mu = (0, 0.2, 0.4),

i.e. skewed, mean-mu_k noise. H0: the real system equals the twin on every arm.
H1: arm 0 is better in reality than the twin predicts (mean shifted by +0.5),
so Thompson sampling moves towards it once it looks good.

Methods (alpha = 0.05):
  mc         Monte Carlo e-value, M draws from the twin FOR THE CHOSEN ARM
  mc_uncond  same, but the twin is run on uniformly random arms, i.e. NOT
             conditioned on the action. Violates the exchangeability that
             Proposition 2 needs, and is included to show that it matters.
  plugin     learned ratio used directly as the bet
  oracle     exact likelihood ratio of the observed outcome given the arm
  ttest      per-arm Welch t-test of real vs twin outcomes, recomputed every 10
             steps (Bonferroni over the 3 arms, not over the looks)

Output: results/exp2.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
from scipy.stats import norm, ttest_ind

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from e4g import Monitor, OnlineRatio  # noqa: E402

K = 3
MU = np.array([0.0, 0.2, 0.4])
SHIFT = np.array([0.5, 0.0, 0.0])
C = np.exp(1 / 8)  # E[exp(eps/2)]


def twin(rng, arms):
    arms = np.asarray(arms)
    return MU[arms] + np.exp(rng.standard_normal(arms.shape) / 2) - C


def log_density(y, arm, shift):
    w = y - MU[arm] - shift + C
    if w <= 0:
        return -np.inf
    return norm.logpdf(2 * np.log(w)) + np.log(2 / w)


def arm_features(rows):
    """rows = [y, arm]; per-arm quadratic in y, so the ratio can differ by arm."""
    y, a = rows[:, 0], rows[:, 1].astype(int)
    onehot = np.eye(K)[a]
    return np.column_stack([onehot[:, 1:], onehot * y[:, None], onehot * (y ** 2)[:, None]])


def learner():
    return OnlineRatio(ridge=1.0, feature_map=arm_features)


def run_one(seed, alt, horizon, m, alpha, test_every=10):
    rng = np.random.default_rng(seed)
    shift = SHIFT if alt else np.zeros(K)
    mon = Monitor(learner(), alpha=alpha)
    mon_u = Monitor(learner(), alpha=alpha)
    # Thompson sampling with N(0, 1) priors and unit noise variance
    n = np.zeros(K)
    s = np.zeros(K)
    real_y = [[] for _ in range(K)]
    twin_y = [[] for _ in range(K)]
    t_cross = None
    pulls = np.zeros(K)
    for t in range(horizon):
        draw = rng.normal(s / (n + 1), 1 / np.sqrt(n + 1))
        a = int(np.argmax(draw))
        y = MU[a] + shift[a] + np.exp(rng.standard_normal() / 2) - C
        n[a] += 1
        s[a] += y
        pulls[a] += 1

        ys = twin(rng, np.full(m, a))
        lr = np.exp(log_density(y, a, shift=SHIFT[a]) - log_density(y, a, 0.0))
        mon.step(np.array([y, a]), np.column_stack([ys, np.full(m, a)]), oracle_lr=lr)

        arms_u = rng.integers(0, K, m)
        mon_u.step(np.array([y, a]), np.column_stack([twin(rng, arms_u), arms_u]))

        real_y[a].append(y)
        twin_y[a].extend(ys)
        if t_cross is None and (t + 1) % test_every == 0:
            ps = [ttest_ind(real_y[k], twin_y[k], equal_var=False).pvalue
                  for k in range(K) if len(real_y[k]) >= 5]
            if ps and min(ps) < alpha / K:
                t_cross = t + 1
    out = dict(mon.first_cross)
    out["mc_uncond"] = mon_u.first_cross["mc"]
    out["ttest"] = t_cross
    out["pulls"] = (pulls / horizon).tolist()
    return out


METHODS = ["mc", "mc_uncond", "plugin", "oracle", "ttest"]


def summarise(runs, horizon):
    grid = np.arange(1, horizon + 1)
    curves, rates, med = {}, {}, {}
    for k in METHODS:
        times = np.array([r[k] if r[k] is not None else np.inf for r in runs])
        curves[k] = [(times <= g).mean() for g in grid]
        rates[k] = float(np.isfinite(times).mean())
        md = np.median(times)
        med[k] = float(md) if np.isfinite(md) else None
    pulls = np.mean([r["pulls"] for r in runs], axis=0).tolist()
    return curves, rates, med, pulls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=400)
    ap.add_argument("--horizon", type=int, default=400)
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    res = {"config": vars(args), "mu": MU.tolist(), "shift": SHIFT.tolist()}
    for label, alt, offset in [("null", False, 0), ("alt", True, 10**6)]:
        t0 = time.time()
        job = partial(run_one, alt=alt, horizon=args.horizon, m=args.m, alpha=args.alpha)
        with ProcessPoolExecutor() as ex:
            runs = list(ex.map(job, [offset + i for i in range(args.reps)], chunksize=8))
        curves, rates, med, pulls = summarise(runs, args.horizon)
        res[label] = {"reject_rate": rates, "median_stop": med, "curves": curves,
                      "mean_pull_share": pulls}
        print(f"[{label}] {time.time()-t0:.0f}s  reject by T={args.horizon}: "
              + "  ".join(f"{k}={v:.3f}" for k, v in rates.items()), flush=True)
        print(f"        pull share per arm: {np.round(pulls, 3).tolist()}", flush=True)
        if alt:
            print("        median stopping time: "
                  + "  ".join(f"{k}={v}" for k, v in med.items()), flush=True)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "exp2.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
