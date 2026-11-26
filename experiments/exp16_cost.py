"""Cost of reaching the alarm in the language-model audit (exp16), our design.

For every method run with the prompts chosen by UCB and every scenario, the
runs of exp16 are repeated with the same seeds, hence the same replies and the
same stopping times, and the computing time spent on the bets and the updates
up to the alarm is recorded. For a score whose update costs the same at every
step this is about the time per query times the number of queries; for the
refit of Forre and Brenner, which revisits the whole history, it grows faster.
The replies generated to reach the alarm are those prepared before the audit
plus M = 8 per query. Each worker runs single-threaded.

Output: results/exp16_cost.json
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT)]

import exp16_llm_audit as ex  # noqa: E402

METHODS = ["slice-bandit", "fb-bandit", "fb-refit-bandit", "combo-bandit",
           "ours-bandit", "ours-simlik-bandit"]
SCENARIOS = ["all-q4", "some-q4", "all-q3", "all-05b", "some-05b", "sharp"]
PREPARED = {"slice-bandit": 6000, "fb-bandit": 6000, "fb-refit-bandit": 800,
            "combo-bandit": 6800, "ours-bandit": 800, "ours-simlik-bandit": 800}
REPS, M = 400, 8


def one(seed, scenario, method, rho):
    first, _, t_method, n = ex.run(seed, scenario, method, 400, M, 0.05, rho)
    return first, t_method


def main():
    path = ROOT / "data" / "llm_audit" / "replies.json"
    ex.init_worker(path)
    rho = ex.fit_log_rho(ex.BASELINE_SUB)
    out = {"reps": REPS, "M": M, "prepared": PREPARED}
    for sc in SCENARIOS:
        for m in METHODS:
            seeds = [10**4 * list(ex.SCENARIOS).index(sc) + i for i in range(REPS)]
            with ProcessPoolExecutor(4, initializer=ex.init_worker, initargs=(path,)) as pool:
                res = list(pool.map(partial(one, scenario=sc, method=m, rho=rho), seeds))
            hit = [(f, s) for f, s in res if f is not None]
            r = {"reject": len(hit) / REPS}
            if hit:
                stops = np.array([f for f, _ in hit], float)
                secs = np.array([s for _, s in hit])
                r.update(median_stop=float(np.median(stops)),
                         median_compute_s=round(float(np.median(secs)), 3),
                         median_replies=float(PREPARED[m] + M * np.median(stops)))
            out.setdefault(sc, {})[m] = r
            print(sc, m, r, flush=True)
            (ROOT / "results" / "exp16_cost.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
