"""Experiment 1: anytime-valid criticism of a black-box generative model.

H0: the stream of real data is distributed as the generative model G.
G is the 2-D "banana" simulator  x = (z1, z2 + b (z1^2 - 1)),  z ~ N(0, I),
with b = 0.5. The monitor may only *sample* from G.

H1: the real system has drifted to b = 0.8. The drift leaves both marginal
means and the marginal of x1 unchanged; it only changes the curvature, so a
detector has to learn the shape of the discrepancy.

Two ratio learners are used, to show that validity does not depend on the
quality of the learner:
  regularised  degree-4 polynomial logistic regression, ridge 1
  flexible     degree-6 polynomial logistic regression, ridge 1e-3 (overfits)

Methods (alpha = 0.05, reject when the process reaches 20):
  mc      Monte Carlo e-value with an online-learned ratio (the proposal)
  plugin  the same learned ratio used directly as the bet (no calibration)
  oracle  exact likelihood ratio p_{b=0.8} / p_{b=0.5} (needs both densities)
  ks      KS test per coordinate (Bonferroni over the 2 coordinates), recomputed
          every 10 steps against a pool of simulator draws ("peeking")

Output: results/exp1.json and figures/exp1_*.png
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
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from e4g import BLRRatio, Monitor, OnlineRatio  # noqa: E402

B0, B1 = 0.5, 0.8
LEARNERS = {"regularised": dict(degree=4, ridge=1.0),
            "flexible": dict(degree=6, ridge=1e-3)}


def banana(rng, n, b):
    z = rng.standard_normal((n, 2))
    return np.column_stack([z[:, 0], z[:, 1] + b * (z[:, 0] ** 2 - 1)])


def log_banana(x, b):
    # volume-preserving transform, so the density is phi(x1) phi(x2 - b(x1^2-1))
    u = x[:, 1] - b * (x[:, 0] ** 2 - 1)
    return -0.5 * (x[:, 0] ** 2 + u ** 2) - np.log(2 * np.pi)


BLR_LEARNERS = {"blr-deg4-predictive": dict(degree=4, mode="predictive"),
                "blr-deg4-mean": dict(degree=4, mode="mean"),
                "blr-deg6-predictive": dict(degree=6, mode="predictive"),
                "blr-deg6-mean": dict(degree=6, mode="mean"),
                "blr-deg4-ratio": dict(degree=4, mode="ratio"),
                "blr-deg6-ratio": dict(degree=6, mode="ratio")}


def make_learner(name, seed):
    if name in LEARNERS:
        return OnlineRatio(**LEARNERS[name])
    # the scaler sees simulator draws only, from a stream independent of the data
    sims = banana(np.random.default_rng(10**7 + seed), 2000, B0)
    return BLRRatio(**BLR_LEARNERS[name]).fit_scaler(sims)


def run_one(seed, b_real, horizon, m, alpha, learner, ks_every=10, ks=True):
    rng = np.random.default_rng(seed)
    mon = Monitor(make_learner(learner, seed), alpha=alpha)
    real_all = banana(rng, horizon, b_real)
    ks_cross = None
    for t in range(horizon):
        x = real_all[t : t + 1]
        y = banana(rng, m, B0)
        lr = float(np.exp(log_banana(x, B1) - log_banana(x, B0))[0])
        mon.step(x[0], y, oracle_lr=lr)
        if ks and ks_cross is None and (t + 1) >= 20 and (t + 1) % ks_every == 0:
            pool = np.asarray(mon.sims)
            p = min(ks_2samp(real_all[: t + 1, j], pool[:, j]).pvalue for j in range(2))
            if p < alpha / 2:
                ks_cross = t + 1
    out = dict(mon.first_cross)
    out["ks"] = ks_cross
    return out


def summarise(runs, horizon):
    methods = ["mc", "plugin", "oracle", "ks"]
    curves, rates, med = {}, {}, {}
    for k in methods:
        times = np.array([r[k] if r[k] is not None else np.inf for r in runs])
        grid = np.arange(1, horizon + 1)
        curves[k] = [(times <= g).mean() for g in grid]
        rates[k] = float(np.isfinite(times).mean())
        med[k] = float(np.median(times)) if np.isfinite(np.median(times)) else None
    return curves, rates, med


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=400)
    ap.add_argument("--horizon", type=int, default=500)
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    res = {"config": vars(args), "b0": B0, "b1": B1, "learners": LEARNERS}
    for learner in LEARNERS:
        res[learner] = {}
        for label, b_real, offset in [("null", B0, 0), ("alt", B1, 10**6)]:
            t0 = time.time()
            # same seeds for both learners: identical data, only the learner differs
            job = partial(run_one, b_real=b_real, horizon=args.horizon, m=args.m,
                          alpha=args.alpha, learner=learner)
            with ProcessPoolExecutor() as ex:
                runs = list(ex.map(job, [offset + i for i in range(args.reps)], chunksize=8))
            curves, rates, med = summarise(runs, args.horizon)
            res[learner][label] = {"reject_rate": rates, "median_stop": med,
                                   "curves": curves}
            print(f"[{learner}/{label}] {time.time()-t0:.0f}s  reject by T={args.horizon}: "
                  + "  ".join(f"{k}={v:.3f}" for k, v in rates.items()), flush=True)
            if label == "alt":
                print("        median stopping time: "
                      + "  ".join(f"{k}={v}" for k, v in med.items()), flush=True)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "exp1.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
