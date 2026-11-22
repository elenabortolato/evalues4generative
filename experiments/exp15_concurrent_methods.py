"""Experiment 15: comparison with the score choices of two concurrent papers.

Forre and Brenner (2026, arXiv:2610.08210) and Cortinovis, Battaglia and Caron
(2026, SLICE, arXiv:2610.13855) use the same normalised Monte Carlo bet as this
paper, for independent data. They differ from it in how the score is chosen.
All methods below use the same bet, the same simulations, the same mixture
over the betting fraction and the same data; only the score differs.

Methods
  ours-blr        Bayesian learning rule, logistic loss, probability-scale
                  predictive (Section 4.4)
  ours-simlik     Bayesian learning rule on the simulation likelihood,
                  posterior-predictive mixture bet, prior variance 0.03
                  (Theorem 3, Proposition 5)
  fb-online       Forre-Brenner, Section 4: score refitted every 10 steps by
                  maximising the penalised empirical growth on the past
                  (ridge equivalent to prior variance 0.03); plug-in bet
  fb-online-tau1  the same with a weaker penalty (variance 1)
  fb-pre-true     Forre-Brenner, Section 3: score pretrained on 2000
                  simulations of the TRUE alternative, then fixed (drift only)
  fb-pre-wrong    the same, pretrained on a wrong guess of the alternative
                  (drift only)
  slice-exact     SLICE-type plug-in of the parametric likelihood ratio
                  p_{theta_hat}/p_{theta_0}, with theta_hat the predictable
                  estimate and the ratio computed EXACTLY (SLICE uses a neural
                  estimate; exact is its best case). Needs the parametric family.
  slice-nre       SLICE as it would be used for this question: a single null
                  (G = 1), the score from an amortised neural ratio network
                  trained once, before monitoring, on simulations from a prior
                  over the parametric family (NRE, 2 x 10^5 pairs; SLICE uses
                  10^7 pairs and the NRE-C objective), evaluated at the same
                  predictable estimate as slice-exact. Its offline cost is
                  recorded.
  fb-online-iid   (bandit only) fb-online applied as an i.i.d. method: the twin
                  is run on uniformly random arms, not on the arm chosen

Tasks
  drift   the banana generator of Experiment 1, b = 0.5 under the null and
          b = 0.8 under the alternative (parametric family: b)
  bandit  the Thompson-sampling bandit of Experiment 2 (parametric family:
          a location shift per arm)

  drift-mis   the alternative is outside SLICE's family: the curvature stays at
              b = 0.5 and the first coordinate's scale grows to 1.2
  bandit-mis  the alternative is outside SLICE's family: arm 2, the most
              pulled, keeps its mean and its noise is scaled by 1.5

Output: results/exp15.json
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

from e4g import BLRRatio, ChoiceBLR, Monitor  # noqa: E402
from e4g.competitors import GrowthFit, NeuralRatio, ParametricPlugin  # noqa: E402
import exp1_generative_drift as d1  # noqa: E402
import exp2_adaptive_bandit as d2  # noqa: E402

DRIFT = ["ours-blr", "ours-simlik", "fb-online", "fb-online-tau1", "fb-pre-true",
         "fb-pre-wrong", "slice-exact", "slice-nre"]
DRIFT_MIS = ["ours-blr", "ours-simlik", "fb-online", "fb-pre-true", "slice-exact", "slice-nre"]
BANDIT_MIS = ["ours-blr", "ours-simlik", "fb-online", "slice-exact", "slice-nre"]
SCALE_X1, SCALE_ARM2 = 1.2, 1.5
BANDIT = ["ours-blr", "ours-simlik", "fb-online", "fb-online-tau1", "slice-exact",
          "slice-nre", "fb-online-iid"]
NETS = {}       # trained once per task in the parent process, inherited by the workers


# ------------------------------------------------- amortised ratio networks
def drift_sample(rng, n):
    b = rng.uniform(0.0, 1.5, n)
    z = rng.standard_normal((n, 2))
    y = np.column_stack([z[:, 0], z[:, 1] + b * (z[:, 0] ** 2 - 1)])
    return b[:, None], y


def drift_features(th, y):
    return np.column_stack([th[:, 0], y, y[:, 0] ** 2, th[:, 0] * (y[:, 0] ** 2 - 1)])


def bandit_sample(rng, n):
    delta = rng.uniform(-1.0, 1.0, n)
    arms = rng.integers(0, d2.K, n)
    y = d2.MU[arms] + delta + np.exp(rng.standard_normal(n) / 2) - d2.C
    return delta[:, None], np.column_stack([y, arms])


def bandit_features(th, rows):
    onehot = np.eye(d2.K)[rows[:, 1].astype(int)]
    return np.column_stack([th[:, 0], rows[:, 0], onehot, onehot * rows[:, :1]])


def train_net(task):
    if task == "drift":
        return NeuralRatio(drift_sample, drift_features, [d1.B0])
    return NeuralRatio(bandit_sample, bandit_features, [0.0])
B_WRONG = 0.2   # a wrong guess of the alternative: curvature reduced, not increased


# ---------------------------------------------------------------- drift task
def drift_log_ratio(x, b_hat):
    return d1.log_banana(x, b_hat) - d1.log_banana(x, d1.B0)


def drift_estimate(past):
    # MLE of b: x2 = u + b (x1^2 - 1) with u ~ N(0, 1) is least squares
    c = past[:, 0] ** 2 - 1
    return float((past[:, 1] * c).sum() / max((c * c).sum(), 1e-8))


def drift_learner(name, seed, m):
    rng = np.random.default_rng(10**7 + seed)
    sims = d1.banana(rng, 2000, d1.B0)
    if name == "ours-blr":
        return BLRRatio(degree=4, mode="predictive").fit_scaler(sims)
    if name == "ours-simlik":
        return ChoiceBLR(degree=4, mode="mixture", prior_var=0.03).fit_scaler(sims)
    if name.startswith("fb-online"):
        tau = 1.0 if name.endswith("tau1") else 0.03
        return GrowthFit(degree=4, tau=tau).fit_scaler(sims)
    if name.startswith("fb-pre"):
        b_alt = d1.B1 if name.endswith("true") else B_WRONG
        g = GrowthFit(degree=4, tau=0.03, mode="pretrained").fit_scaler(sims)
        return g.pretrain(d1.banana(rng, 2000, b_alt), d1.banana(rng, 2000 * m, d1.B0), m)
    if name == "slice-exact":
        return ParametricPlugin(drift_log_ratio, drift_estimate)
    if name == "slice-nre":
        net = NETS["drift"]
        return ParametricPlugin(lambda x, b: net.logit([b], x) - net.logit([d1.B0], x),
                                drift_estimate)
    raise ValueError(name)


def run_drift(seed, alt, horizon, m, alpha, name, mis=False):
    rng = np.random.default_rng(seed)
    b_real = d1.B1 if (alt and not mis) else d1.B0
    mon = Monitor(drift_learner(name, seed, m), alpha=alpha)
    real = d1.banana(rng, horizon, b_real)
    if alt and mis:
        # same curvature, wider first coordinate: outside the family in b
        real = np.column_stack([SCALE_X1 * real[:, 0],
                                real[:, 1] + d1.B0 * ((SCALE_X1 * real[:, 0]) ** 2 - real[:, 0] ** 2)])
    for t in range(horizon):
        x = real[t:t + 1]
        mon.step(x[0], d1.banana(rng, m, d1.B0))
    return mon.first_cross["mc"]


# --------------------------------------------------------------- bandit task
def bandit_log_ratio(rows, shifts):
    out = np.empty(len(rows))
    for i, (y, a) in enumerate(rows):
        a = int(a)
        out[i] = d2.log_density(y, a, shifts[a]) - d2.log_density(y, a, 0.0)
    return out


def bandit_estimate(past):
    # unbiased location estimate per arm: E[exp(eps/2)] - e^{1/8} = 0
    shifts = np.zeros(d2.K)
    for a in range(d2.K):
        ys = past[past[:, 1] == a, 0]
        if len(ys):
            shifts[a] = ys.mean() - d2.MU[a]
    return shifts


def bandit_learner(name, seed):
    r = np.random.default_rng(10**7 + seed)
    arms = r.integers(0, d2.K, 3000)
    rows = np.column_stack([d2.twin(r, arms), arms])
    fm = d2.arm_features
    if name == "ours-blr":
        return BLRRatio(feature_map=fm, mode="predictive").fit_scaler(rows)
    if name == "ours-simlik":
        return ChoiceBLR(feature_map=fm, mode="mixture", prior_var=0.03).fit_scaler(rows)
    if name.startswith("fb-online"):
        tau = 1.0 if name == "fb-online-tau1" else 0.03
        return GrowthFit(feature_map=fm, tau=tau).fit_scaler(rows)
    if name == "slice-exact":
        return ParametricPlugin(bandit_log_ratio, bandit_estimate)
    if name == "slice-nre":
        net = NETS["bandit"]

        def log_ratio(rows, shifts):
            per_row = shifts[rows[:, 1].astype(int)][:, None]
            return net.logit(per_row, rows) - net.logit([0.0], rows)
        return ParametricPlugin(log_ratio, bandit_estimate)
    raise ValueError(name)


def run_bandit(seed, alt, horizon, m, alpha, name, mis=False):
    rng = np.random.default_rng(seed)
    shift = d2.SHIFT if (alt and not mis) else np.zeros(d2.K)
    scale = np.array([1.0, 1.0, SCALE_ARM2]) if (alt and mis) else np.ones(d2.K)
    mon = Monitor(bandit_learner(name, seed), alpha=alpha)
    n, s = np.zeros(d2.K), np.zeros(d2.K)
    for t in range(horizon):
        a = int(np.argmax(rng.normal(s / (n + 1), 1 / np.sqrt(n + 1))))
        y = d2.MU[a] + shift[a] + scale[a] * (np.exp(rng.standard_normal() / 2) - d2.C)
        n[a] += 1
        s[a] += y
        if name == "fb-online-iid":
            arms = rng.integers(0, d2.K, m)
        else:
            arms = np.full(m, a)
        mon.step(np.array([y, a]), np.column_stack([d2.twin(rng, arms), arms]))
    return mon.first_cross["mc"]


# -------------------------------------------------------------------- driver
def summarise(times):
    t = np.array([np.inf if v is None else v for v in times], dtype=float)
    md = np.median(t)
    return dict(rate=float(np.isfinite(t).mean()), median=float(md) if np.isfinite(md) else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=400)
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--task", default="drift,bandit")
    ap.add_argument("--methods", default="")
    ap.add_argument("--seed-offset", type=int, default=0,
                    help="added to every seed; a nonzero offset gives replications "
                         "disjoint from those used to choose tuning constants")
    ap.add_argument("--out", default=str(ROOT / "results" / "exp15.json"))
    args = ap.parse_args()
    out_path = Path(args.out)
    res = json.loads(out_path.read_text()) if out_path.exists() else {}
    for task in args.task.split(","):
        default = {"drift": DRIFT, "bandit": BANDIT, "drift-mis": DRIFT_MIS,
                   "bandit-mis": BANDIT_MIS}[task]
        names = args.methods.split(",") if args.methods else default
        base = task.split("-")[0]
        run, horizon = (run_drift, 500) if base == "drift" else (run_bandit, 400)
        run = partial(run, mis=task.endswith("-mis"))
        for name in names:
            if task == "drift" and name == "fb-online-iid":
                continue
            row = {}
            if name == "slice-nre" and base not in NETS:
                NETS[base] = train_net(base)
                print(f"{task:6s} slice-nre network trained: {NETS[base].cost}", flush=True)
            if name == "slice-nre":
                row["offline_cost"] = NETS[base].cost
            t0 = time.time()
            for label, alt, offset in [("null", False, args.seed_offset),
                                       ("alt", True, args.seed_offset + 10**6)]:
                job = partial(run, alt=alt, horizon=horizon, m=args.m, alpha=args.alpha, name=name)
                with ProcessPoolExecutor() as ex:
                    times = list(ex.map(job, [offset + i for i in range(args.reps)], chunksize=8))
                row[label] = summarise(times)
            row["online_seconds"] = round(time.time() - t0, 1)
            res.setdefault(task, {})[name] = row
            print(f"{task:6s} {name:15s} false alarms {row['null']['rate']:.3f}  "
                  f"power {row['alt']['rate']:.3f}  median {row['alt']['median']}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
            out_path.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
