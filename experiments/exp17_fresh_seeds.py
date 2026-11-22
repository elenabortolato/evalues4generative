"""Experiment 17: the drift rows of Table 4 on replications disjoint from tuning.

The prior variance 0.03 of the simulation-likelihood learner was chosen on 24
alternative runs of the drift task (seeds 10**6 + i), which are among the 400
replications of Experiments 5, 13 and 14. This script reruns the four drift
learners of Table 4 on 400 fresh replications with seeds 5*10**6 + i (null) and
6*10**6 + i (alternative), so that no reported run was used for tuning; all
learners see identical data.

Output: results/exp17.json
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

LEARNERS = {"refitted point estimate": "regularised",
            "BLR, logistic": "blr-deg4-predictive",
            "BLR, simulation likelihood": "choice-deg4-mixture-p003",
            "predictive update, simulation likelihood": "mp-choice-resample"}
REPS, OFFSET = 400, 5 * 10**6


def run(seed, name, b_real):
    return D.run_one(seed, b_real=b_real, horizon=500, m=16, alpha=0.05, learner=name,
                     ks=False)["mc"]


def main():
    res = {"seeds": {"null": [OFFSET, OFFSET + REPS - 1],
                     "alt": [OFFSET + 10**6, OFFSET + 10**6 + REPS - 1]}}
    for label_row, name in LEARNERS.items():
        t0, out = time.time(), {}
        for label, b, off in (("null", D.B0, OFFSET), ("alt", D.B1, OFFSET + 10**6)):
            with ProcessPoolExecutor() as ex:
                times = list(ex.map(partial(run, name=name, b_real=b),
                                    [off + i for i in range(REPS)], chunksize=8))
            t = np.array([x if x is not None else np.inf for x in times])
            med = np.median(t)
            out[label] = dict(reject=float(np.isfinite(t).mean()),
                              median_stop=float(med) if np.isfinite(med) else None)
        res[label_row] = dict(learner=name, **out)
        print(f"{label_row:42s} false alarms {out['null']['reject']:.3f}  "
              f"power {out['alt']['reject']:.3f}  median {out['alt']['median_stop']}  "
              f"({time.time() - t0:.0f}s)", flush=True)
        (ROOT / "results" / "exp17.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
