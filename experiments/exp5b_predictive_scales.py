"""Experiment 5b: which posterior predictive? Probability scale vs ratio scale.

Adds the ratio-scale posterior predictive score, log E_q[exp(eta)] = mu + v/2,
to Experiment 5. It is the predictive that Proposition 3 of the paper is about
(the posterior predictive density ratio of the tilted family), and it inflates
the bet where the posterior is uncertain, whereas the probability-scale
predictive kappa * mu shrinks it. Same seeds as Experiments 1, 2 and 5.

Output: results/exp5b.json
"""

from __future__ import annotations

import json
import sys
import time
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import exp1_generative_drift as drift  # noqa: E402
import exp2_adaptive_bandit as bandit  # noqa: E402
from exp5_bayesian_learning_rule import run_block, stats  # noqa: E402

REPS = 400


def main():
    res = {"drift": {}, "bandit": {}}
    for name in ["blr-deg4-ratio", "blr-deg6-ratio"]:
        row = {}
        for label, b, off in [("null", drift.B0, 0), ("alt", drift.B1, 10**6)]:
            t0 = time.time()
            job = partial(drift.run_one, b_real=b, horizon=500, m=16, alpha=0.05,
                          learner=name, ks=False)
            runs = run_block(job, [off + i for i in range(REPS)])
            row[label] = {k: stats(runs, k) for k in ("mc", "plugin")}
            row[label]["seconds"] = time.time() - t0
        res["drift"][name] = row
        print(f"drift  {name:18s} false alarms mc={row['null']['mc']['reject']:.3f} "
              f"plugin={row['null']['plugin']['reject']:.3f} | power mc={row['alt']['mc']['reject']:.3f} "
              f"median stop={row['alt']['mc']['median_stop']}", flush=True)
    row = {}
    for label, alt, off in [("null", False, 0), ("alt", True, 10**6)]:
        job = partial(bandit.run_one, alt=alt, horizon=400, m=16, alpha=0.05, kind="blr-ratio")
        runs = run_block(job, [off + i for i in range(REPS)])
        row[label] = {k: stats(runs, k) for k in ("mc", "mc_uncond", "plugin")}
    res["bandit"]["blr-ratio"] = row
    print(f"bandit blr-ratio           false alarms mc={row['null']['mc']['reject']:.3f} "
          f"plugin={row['null']['plugin']['reject']:.3f} | power mc={row['alt']['mc']['reject']:.3f} "
          f"median stop={row['alt']['mc']['median_stop']}", flush=True)
    (ROOT / "results" / "exp5b.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
