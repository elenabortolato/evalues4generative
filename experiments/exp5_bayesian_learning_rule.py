"""Experiment 5: updating the score at every step with the Bayesian learning rule.

The ratio model (balanced logistic regression, real vs simulator) gets a
Gaussian posterior updated at every step by one natural-gradient step of the
Bayesian learning rule (online form, as in BONG), and the bet uses either the
posterior PREDICTIVE log-odds (shrunk where the posterior is uncertain) or the
posterior MEAN. Compared with the point-estimate learner of Experiments 1-2,
refitted every 10 steps after a 20-step burn-in. Same seeds as Experiments 1-2,
so the real data are identical and only the learner differs.

Validity cannot change (any predictable score is valid); the question is power.

Output: results/exp5.json
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import exp1_generative_drift as drift  # noqa: E402
import exp2_adaptive_bandit as bandit  # noqa: E402


def stats(runs, key):
    times = np.array([r[key] if r[key] is not None else np.inf for r in runs])
    md = np.median(times)
    return {"reject": float(np.isfinite(times).mean()),
            "median_stop": float(md) if np.isfinite(md) else None,
            "curve": [(times <= g).mean() for g in range(1, int(np.max(times[np.isfinite(times)], initial=1)) + 2)]}


def run_block(job, seeds):
    with ProcessPoolExecutor() as ex:
        return list(ex.map(job, seeds, chunksize=8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=400)
    args = ap.parse_args()
    res = {"config": vars(args), "drift": {}, "bandit": {}}

    for name in ["regularised", "flexible", *drift.BLR_LEARNERS]:
        row = {}
        for label, b, off in [("null", drift.B0, 0), ("alt", drift.B1, 10**6)]:
            t0 = time.time()
            job = partial(drift.run_one, b_real=b, horizon=500, m=16, alpha=0.05,
                          learner=name, ks=False)
            runs = run_block(job, [off + i for i in range(args.reps)])
            row[label] = {k: stats(runs, k) for k in ("mc", "plugin")}
            row[label]["seconds"] = time.time() - t0
        res["drift"][name] = row
        print(f"drift  {name:22s} false alarms mc={row['null']['mc']['reject']:.3f} "
              f"plugin={row['null']['plugin']['reject']:.3f} | power mc={row['alt']['mc']['reject']:.3f} "
              f"median stop={row['alt']['mc']['median_stop']}", flush=True)

    for kind in ["refit", "blr-predictive", "blr-mean"]:
        row = {}
        for label, alt, off in [("null", False, 0), ("alt", True, 10**6)]:
            t0 = time.time()
            job = partial(bandit.run_one, alt=alt, horizon=400, m=16, alpha=0.05, kind=kind)
            runs = run_block(job, [off + i for i in range(args.reps)])
            row[label] = {k: stats(runs, k) for k in ("mc", "mc_uncond", "plugin")}
            row[label]["seconds"] = time.time() - t0
        res["bandit"][kind] = row
        print(f"bandit {kind:22s} false alarms mc={row['null']['mc']['reject']:.3f} "
              f"uncond={row['null']['mc_uncond']['reject']:.3f} plugin={row['null']['plugin']['reject']:.3f}"
              f" | power mc={row['alt']['mc']['reject']:.3f} median stop={row['alt']['mc']['median_stop']}",
              flush=True)

    (ROOT / "results" / "exp5.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
