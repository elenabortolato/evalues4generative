"""Experiment 13: the Bayesian learning rule on the simulation likelihood.

Compares, on the drift task of Experiment 1 (same seeds, so identical data), the
learner of Experiments 1 and 5 (BLR on the balanced logistic loss, probit-scale
predictive score, e4g.BLRRatio) with the learner that Theorem 3 is about
(BLR on the conditional-logit simulation likelihood, e4g.ChoiceBLR), whose bet is
the posterior-predictive factor mean_s E(theta_s) over posterior nodes
("mixture") or the factor at the posterior mean ("mean"). Prior variance 1
(the default) and 0.03 (tuned on 24 alternative runs of Experiment 1).

Also reports, for the alternative, the betting diagnostics that explain the
difference: E[E] - 1, sd(E), and the best betting fraction in steps 100-250.

Output: results/exp13.json
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

import exp1_generative_drift as D  # noqa: E402
from e4g import LambdaMixture, Monitor  # noqa: E402

LEARNERS = ["blr-deg4-predictive", "choice-deg4-mixture", "choice-deg4-mixture-p003",
            "choice-deg4-mean-p003", "blr-deg6-predictive", "choice-deg6-mixture-p003"]
REPS = 400


def run(seed, name, b_real):
    return D.run_one(seed, b_real=b_real, horizon=500, m=16, alpha=0.05, learner=name,
                     ks=False)["mc"]


def diagnostics(name, n=16, lo=100, hi=250):
    lams = LambdaMixture().lams
    es = []
    for i in range(n):
        seed = 10 ** 6 + i
        rng = np.random.default_rng(seed)
        mon = Monitor(D.make_learner(name, seed))
        real = D.banana(rng, hi, D.B1)
        e = [mon.step(real[t], D.banana(rng, 16, D.B0))["mc"] for t in range(hi)]
        es.append(e[lo:hi])
    e = np.ravel(es)
    g = [float(np.log(1 - l + l * e).mean()) for l in lams]
    k = int(np.argmax(g))
    return dict(mean_minus_1=float(e.mean() - 1), sd=float(e.std()), best_lambda=float(lams[k]),
                best_growth=g[k], growth_lambda1=g[-1])


def main():
    res = {}
    for name in LEARNERS:
        t0 = time.time()
        out = {}
        for label, b, off in (("null", D.B0, 0), ("alt", D.B1, 10 ** 6)):
            with ProcessPoolExecutor() as ex:
                times = list(ex.map(partial(run, name=name, b_real=b),
                                    [off + i for i in range(REPS)], chunksize=8))
            t = np.array([x if x is not None else np.inf for x in times])
            med = np.median(t)
            out[label] = dict(reject=float(np.isfinite(t).mean()),
                              median_stop=float(med) if np.isfinite(med) else None)
        out["diagnostics"] = diagnostics(name)
        res[name] = out
        print(f"{name:26s} false alarms {out['null']['reject']:.3f}  power {out['alt']['reject']:.3f}"
              f"  median stop {out['alt']['median_stop']}  | {out['diagnostics']}"
              f"  ({time.time() - t0:.0f}s)", flush=True)
    (ROOT / "results" / "exp13.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
