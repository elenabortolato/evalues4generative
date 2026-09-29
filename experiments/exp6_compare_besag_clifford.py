"""Experiment 6: head-to-head with the Besag-Clifford e-process.

The Besag-Clifford (BC) e-process of Dombowsky, Engelhardt & Ramdas (2026, Sec. 4)
bets, at each step, the soft-rank e-value
    U_t = (M+1) T(X_t) / (T(X_t) + sum_j T(Y_tj))
with M fresh null draws, for an i.i.d. null. They recommend the likelihood
ratio as statistic (their Thm 1) and, for a composite alternative, a plug-in
estimate of it (their Secs 3.2, 6.3); betting is all-in (lambda = 1, their "ULR
process") or GRAPA-type. Our method uses the same e-value; it differs in the
statistic (learned from real-vs-simulated data, posterior predictive, online
Bayesian learning rule) and in the betting (a mixture over lambda).

On the generative-drift task (i.i.d. null, so their setting applies) we compare,
on identical data and identical simulator draws:

  statistic                     needs
  LR    true p_{b=0.8}/p_{b=0.5} both densities and the true alternative (oracle)
  MLE   p_{b_hat}/p_{b=0.5},      the parametric family's density (their Sec 6.3,
        b_hat predictable MLE     made predictable as in Wasserman et al. 2020)
  BLR   our predictive score      only samples from the simulator

  betting: all-in (lambda = 1), aGRAPA (Waudby-Smith & Ramdas 2023), and our
  mixture over a lambda grid.

The bandit comparison (their i.i.d. formulation vs conditioning on the action)
is Experiment 2: the i.i.d. formulation rejects a true null in every run.

Output: results/exp6.json
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from e4g import LambdaMixture, mc_factor  # noqa: E402
from exp1_generative_drift import B0, B1, banana, log_banana, make_learner  # noqa: E402

REPS, HORIZON, M, ALPHA = 400, 500, 16, 0.05
SCORES = ["LR", "MLE", "BLR"]
BETS = ["allin", "agrapa", "mixture"]


class AllIn:
    def __init__(self):
        self.log_w = 0.0

    def update(self, e):
        self.log_w += np.log(max(e, 1e-300))
        return np.exp(min(self.log_w, 700.0))


class AGrapa:
    """Approximate GRAPA: lambda_t = clip(mu / (sigma^2 + mu^2), 0, c) from past factors."""

    def __init__(self, c=0.9):
        self.c, self.log_w, self.hist = c, 0.0, []

    def update(self, e):
        if len(self.hist) >= 2:
            d = np.asarray(self.hist) - 1.0
            mu, var = d.mean(), d.var()
            lam = float(np.clip(mu / (var + mu ** 2 + 1e-12), 0.0, self.c))
        else:
            lam = 0.0
        self.log_w += np.log(max(1.0 - lam + lam * e, 1e-300))
        self.hist.append(e)
        return np.exp(min(self.log_w, 700.0))


def b_mle(x_past):
    """MLE of the banana curvature b from past real data (closed form)."""
    if len(x_past) < 5:
        return B0
    x = np.asarray(x_past)
    u = x[:, 0] ** 2 - 1
    return float((x[:, 1] * u).sum() / (u ** 2).sum())


def run_one(seed, b_real):
    rng = np.random.default_rng(seed)
    blr = make_learner("blr-deg4-predictive", seed)
    real = banana(rng, HORIZON, b_real)
    procs = {(s, b): {"allin": AllIn, "agrapa": AGrapa, "mixture": LambdaMixture}[b]()
             for s in SCORES for b in BETS}
    first = {k: None for k in procs}
    for t in range(HORIZON):
        x = real[t : t + 1]
        y = banana(rng, M, B0)
        z = np.vstack([x, y])
        bh = b_mle(real[:t])
        scores = {"LR": log_banana(z, B1) - log_banana(z, B0),
                  "MLE": log_banana(z, bh) - log_banana(z, B0),
                  "BLR": blr.score(z)}
        for s in SCORES:
            e = mc_factor(scores[s][0], scores[s][1:])
            for b in BETS:
                v = procs[(s, b)].update(e)
                if first[(s, b)] is None and v >= 1 / ALPHA:
                    first[(s, b)] = t + 1
        blr.update(x, y)
    return {f"{s}|{b}": first[(s, b)] for s in SCORES for b in BETS}


def summarise(runs):
    out = {}
    for k in runs[0]:
        times = np.array([r[k] if r[k] is not None else np.inf for r in runs])
        md = np.median(times)
        out[k] = {"reject": float(np.isfinite(times).mean()),
                  "median_stop": float(md) if np.isfinite(md) else None}
    return out


def main():
    res = {"config": dict(reps=REPS, horizon=HORIZON, m=M, alpha=ALPHA)}
    for label, b, off in [("null", B0, 0), ("alt", B1, 10**6)]:
        t0 = time.time()
        with ProcessPoolExecutor() as ex:
            runs = list(ex.map(partial(run_one, b_real=b), [off + i for i in range(REPS)],
                               chunksize=8))
        res[label] = summarise(runs)
        print(f"[{label}] {time.time() - t0:.0f}s", flush=True)
    print(f"{'statistic':10s} {'betting':8s} {'false alarms':>12s} {'power':>6s} {'median':>7s}")
    for s in SCORES:
        for b in BETS:
            k = f"{s}|{b}"
            print(f"{s:10s} {b:8s} {res['null'][k]['reject']:12.3f} "
                  f"{res['alt'][k]['reject']:6.3f} {res['alt'][k]['median_stop']}")
    (ROOT / "results" / "exp6.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
