"""Computing time per query of each method in the language-model audit (exp16).

Each method is run sequentially, on one core, under the null for the whole
horizon (no early stop is possible after the rare false alarm, which ends a
run), and the time spent on the bets and the updates is divided by the number
of queries. The generation of the replies is excluded: it is the same for all
methods and takes 2 to 8 seconds per query (M = 8 replies) on four CPU cores.
The refit of Forre and Brenner revisits the whole history, so its cost per
query grows with the horizon; it is also timed over a longer horizon.

Output: results/exp16_timing.json
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT)]

import exp16_llm_audit as ex  # noqa: E402

METHODS = ["slice", "fb-pre", "fb-refit", "ours-uniform", "ours-simlik",
           "slice-bandit", "fb-bandit", "fb-refit-bandit", "combo-bandit",
           "ours-bandit", "ours-simlik-bandit"]
REPS, REPS_LONG = 20, 5


def time_method(method, horizon, rho, reps=REPS):
    spent, queries = 0.0, 0
    for i in range(reps):
        _, _, t_method, n = ex.run(90_000 + i, "null", method, horizon, 8, 0.05, rho)
        spent, queries = spent + t_method, queries + n
    return 1000 * spent / queries


def main():
    ex.init_worker(ROOT / "data" / "llm_audit" / "replies.json")
    rho = ex.fit_log_rho(ex.BASELINE_SUB)
    out = {"reps": REPS, "reps_1600": REPS_LONG, "ms_per_query_400": {}, "ms_per_query_1600": {}}
    for m in METHODS:
        out["ms_per_query_400"][m] = round(time_method(m, 400, rho), 2)
        print(f"{m:20s} {out['ms_per_query_400'][m]:6.2f} ms per query (horizon 400)", flush=True)
    for m in ["fb-refit", "ours-uniform", "ours-simlik"]:
        out["ms_per_query_1600"][m] = round(time_method(m, 1600, rho, REPS_LONG), 2)
        print(f"{m:20s} {out['ms_per_query_1600'][m]:6.2f} ms per query (horizon 1600)", flush=True)
    (ROOT / "results" / "exp16_timing.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
