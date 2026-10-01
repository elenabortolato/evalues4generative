"""Experiment 14: predictive-first updates of the classifier (martingale posteriors).

For each Gaussian-approximation learner of the paper (BLR on the logistic loss,
BLR on the simulation likelihood), the corresponding predictive-first learner
(e4g.MartingalePredictive): the classifier's predictive is updated directly
by Fisher scoring on its own score, a parametric martingale posterior, and the
bet uses either the current predictive ("plugin") or an average over
predictive-resampling paths ("resample"). Same seeds as Experiments 1, 2, 5
and 13, so the data are identical and only the learner differs.

Output: results/exp14.json
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

import exp1_generative_drift as drift  # noqa: E402
import exp2_adaptive_bandit as bandit  # noqa: E402

REPS = 400
DRIFT = ["mp-logistic-plugin", "mp-logistic-resample", "mp-choice-plugin", "mp-choice-resample",
         "mp-logistic-plugin-deg6", "mp-logistic-resample-deg6"]
BANDIT = ["mp-logistic-plugin", "mp-logistic-resample", "mp-choice-plugin", "mp-choice-resample"]


def stat(times):
    t = np.array([x if x is not None else np.inf for x in times])
    med = np.median(t)
    return dict(reject=float(np.isfinite(t).mean()), median_stop=float(med) if np.isfinite(med) else None)


def main():
    res = {"drift": {}, "bandit": {}}
    for name in DRIFT:
        t0, row = time.time(), {}
        for label, b, off in (("null", drift.B0, 0), ("alt", drift.B1, 10 ** 6)):
            job = partial(drift.run_one, b_real=b, horizon=500, m=16, alpha=0.05, learner=name,
                          ks=False)
            with ProcessPoolExecutor() as ex:
                runs = list(ex.map(job, [off + i for i in range(REPS)], chunksize=8))
            row[label] = stat([r["mc"] for r in runs])
        res["drift"][name] = row
        print(f"drift  {name:28s} false alarms {row['null']['reject']:.3f} power "
              f"{row['alt']['reject']:.3f} median {row['alt']['median_stop']} ({time.time()-t0:.0f}s)",
              flush=True)
    for kind in BANDIT:
        t0, row = time.time(), {}
        for label, alt, off in (("null", False, 0), ("alt", True, 10 ** 6)):
            job = partial(bandit.run_one, alt=alt, horizon=400, m=16, alpha=0.05, kind=kind)
            with ProcessPoolExecutor() as ex:
                runs = list(ex.map(job, [off + i for i in range(REPS)], chunksize=8))
            row[label] = {k: stat([r[k] for r in runs]) for k in ("mc", "mc_uncond")}
        res["bandit"][kind] = row
        print(f"bandit {kind:28s} false alarms {row['null']['mc']['reject']:.3f} (uncond "
              f"{row['null']['mc_uncond']['reject']:.3f}) power {row['alt']['mc']['reject']:.3f} "
              f"median {row['alt']['mc']['median_stop']} ({time.time()-t0:.0f}s)", flush=True)
    (ROOT / "results" / "exp14.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
