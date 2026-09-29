"""Experiment 3: what the simulation budget M buys.

Same generative-drift problem as Experiment 1 (regularised learner), with the
number of simulator draws per step M varied. Proposition 2 is exact for every
M >= 1, so the type-I error should stay below alpha for all M; power should
grow with M towards the "M = infinity" limit T(X) / E_H0[T], at a simulation
cost linear in M.

Output: results/exp3.json
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

from exp1_generative_drift import B0, B1, run_one  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--horizon", type=int, default=500)
    ap.add_argument("--ms", type=int, nargs="+", default=[1, 2, 4, 8, 16, 64])
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    res = {"config": vars(args), "rows": []}
    for m in args.ms:
        row = {"m": m}
        for label, b_real, offset in [("null", B0, 0), ("alt", B1, 10**6)]:
            t0 = time.time()
            job = partial(run_one, b_real=b_real, horizon=args.horizon, m=m,
                          alpha=args.alpha, learner="regularised", ks=False)
            with ProcessPoolExecutor() as ex:
                runs = list(ex.map(job, [offset + i for i in range(args.reps)], chunksize=8))
            times = np.array([r["mc"] if r["mc"] is not None else np.inf for r in runs])
            row[f"{label}_reject"] = float(np.isfinite(times).mean())
            if label == "alt":
                md = np.median(times)
                row["alt_median_stop"] = float(md) if np.isfinite(md) else None
                row["alt_curve"] = [(times <= g).mean() for g in range(1, args.horizon + 1)]
            row[f"{label}_seconds"] = time.time() - t0
        print(f"M={m:3d}  type-I={row['null_reject']:.3f}  power={row['alt_reject']:.3f}  "
              f"median stop={row['alt_median_stop']}", flush=True)
        res["rows"].append(row)
    (ROOT / "results" / "exp3.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
