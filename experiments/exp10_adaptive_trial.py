"""Experiment 10: a response-adaptive trial checked against its in-silico twin.

Setting. A three-arm trial of insulin-dosing strategies for adults with type 1
diabetes. Participants enter one at a time and spend one in-clinic day
(06:00-18:00, standardised breakfast of 50 g at 07:00 and lunch of 70 g at
12:30) under the strategy assigned to them. Allocation is response-adaptive:
after a burn-in of 15 participants (5 per arm, permuted blocks), each new
participant is assigned by Thompson sampling on time in range (TIR, 70-180
mg/dL), with allocation probabilities floored at 0.1, as in Bayesian
response-adaptive trials.

Arms (dosing strategy run by the pump, closed loop within the day):
  A  standard      carbohydrate bolus, correction to 140 above 150, suspend < 80
  B  aggressive    bolus x1.5, correction to 100, suspend < 60
  C  conservative  bolus x0.8, correction to 160, suspend < 90

The twin is the in-silico trial: the UVA/Padova simulator (e4g.t1d) with the
participant's virtual-patient parameters (one of simglucose's 10 adults),
run M = 16 times under the arm actually assigned and the same meal protocol.
The UVA/Padova simulator is accepted by the FDA as a substitute for
preclinical animal trials (Kovatchev et al., 2009), so a trial designed and
justified in silico is a realistic use.

Scenarios for the real participants (the twin is always the nominal model):
  null        real participants are the simulator
  resistance  insulin sensitivity (Vmx, kp3) 25% lower than in the virtual
              cohort, in every arm
  rescue      participants self-treat hypoglycaemia: 15 g of carbohydrate when
              the CGM reads below 70 (at most every 30 minutes). The simulator
              omits this behaviour, and its effect depends on the arm: it
              matters most under the aggressive arm, which causes most lows

Monitors (alpha = 0.05, horizon 120 participants):
  mc_blr    Monte Carlo e-process, BLR predictive score, twin under the
            assigned arm (the paper's method)
  plugin    same score, bet directly
  pooled    Monte Carlo e-process with the twin run under an arm drawn
            uniformly for every simulated participant, i.e. the pre-trial
            in-silico prediction that ignores the allocation
  ttest     one-sample t-test of standardised TIR residuals, recomputed after
            every participant from the fifth

Which arm? One e-process per arm, betting only on the participants assigned to
that arm (factor 1 otherwise), each with its own BLR score. Whether a
participant goes to arm k is decided before the outcome, so each is a test
martingale for H0_k: "the twin is correct under arm k", whatever the
allocation (Theorem 1). Each arm's e-value is its wealth at the end of the
trial, or when it first reaches K / alpha (a bounded stopping time), and e-BH
(Wang & Ramdas 2022) is applied across the K = 3 arms. This is the bandit
multiple-testing construction of Xu, Wang & Ramdas (2021) with the twin as the
null. e-BH controls the FDR under arbitrary dependence between the arms.

All replications of one scenario are simulated together: participant i of
every replicate trial is stepped in the same batch (the patient parameters are
per trajectory), which is exact and much faster than one trial at a time.

Output: results/exp10.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.stats import ttest_1samp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from e4g import BLRRatio, LambdaMixture, mc_factor, plugin_factor, t1d  # noqa: E402

ADULTS = [f"adult#{i:03d}" for i in range(1, 11)]
ARMS = {"A": (1.0, 140.0, 80.0), "B": (1.5, 100.0, 60.0), "C": (0.8, 160.0, 90.0)}
ARM_NAMES = list(ARMS)
STEPS = 144                            # 12 hours of 5-minute intervals from 06:00
MEALS = {12: 50.0, 78: 70.0}           # 07:00 and 12:30
M, ALPHA, N_PART, BURN = 16, 0.05, 120, 15
THRESH = 1 / ALPHA
VAR = t1d.Variability(p_snack=0.0)     # in clinic: no unannounced snacks
COLS = [c for c in t1d.PARAMS.columns if c != "Name"]


def params(idx):
    """Namespace of per-trajectory parameter arrays for patients ADULTS[idx]."""
    rows = t1d.PARAMS.set_index("Name").loc[[ADULTS[i] for i in idx]]
    p = SimpleNamespace(**{c: rows[c].values.astype(float) for c in COLS})
    q = t1d.QUEST.set_index("Name").loc[[ADULTS[i] for i in idx]]
    ctl = SimpleNamespace(basal=p.u2ss * p.BW / 6000.0, cr=q.CR.values, cf=q.CF.values)
    x0 = rows.iloc[:, 1:14].values.astype(float)   # the 13 initial states
    return p, ctl, x0


def simulate(idx, arm, noisy, vmx, rescue, rng):
    """CGM every 5 minutes, (K, STEPS), for K trajectories run in one batch.

    idx: patient index; arm: 0/1/2; noisy: False for the noise-free path;
    vmx: insulin-sensitivity factor; rescue: self-treats hypoglycaemia.
    """
    p, ctl, x0 = params(idx)
    k = len(idx)
    b = t1d.Batch(x=x0.copy(), last_qsto=x0[:, 0] + x0[:, 1], last_food=np.zeros(k),
                  eating=np.zeros(k, bool), planned=np.zeros(k), last_cho=np.zeros(k))
    mult_b, target, suspend = (np.array([ARMS[a][j] for a in ARM_NAMES])[arm] for j in range(3))
    cgm = t1d.bg(b, p)
    since_rescue = np.full(k, 99)
    n_rescue = np.zeros(k, int)
    out = np.empty((k, STEPS))
    for s in range(STEPS):
        meal = MEALS.get(s, 0.0)
        rate = np.where(cgm < suspend, 0.0, ctl.basal)
        if meal > 0:
            bolus = mult_b * meal / ctl.cr + (cgm > 150) * (cgm - target) / ctl.cf
            rate = rate + np.maximum(bolus, 0.0) / 5.0
        take = rescue & (cgm < 70) & (since_rescue >= 6)
        since_rescue = np.where(take, 0, since_rescue + 1)
        n_rescue += take
        eta = rng.standard_normal(k) * noisy
        carbs = meal * np.exp(VAR.sigma_carb * rng.standard_normal(k) * noisy) + 15.0 * take
        m_is = np.exp(VAR.sigma_is * eta) * vmx
        for i in range(5):
            t1d.minute(b, p, carbs if i == 0 else 0.0, rate, vmx_mult=m_is, kp3_mult=m_is)
        g = t1d.bg(b, p)
        cgm = np.clip(g + noisy * t1d.johnson_noise(rng, k), 39.0, 600.0)
        out[:, s] = cgm
    return out, n_rescue


def rows(traces, g, arm, use_arm=True):
    r = traces - g
    tbr = (traces < 70).mean(1) * 10.0
    tar = (traces > 180).mean(1) * 5.0
    ab = (arm == 1).astype(float) * use_arm
    ac = (arm == 2).astype(float) * use_arm
    return np.column_stack([r.mean(1) / 20.0, r.std(1) / 20.0, tbr, tar, ab, ac])


def feature_map(z):
    rbar, rsd, tbr, tar, ab, ac = z.T
    return np.column_stack([rbar, rbar ** 2, rsd, tbr, tar, tbr * ab, tbr * ac,
                            rbar * ab, rbar * ac])


def tir(traces):
    return ((traces >= 70) & (traces <= 180)).mean(1)


def pilot(seed=10 ** 6, n=300):
    """Simulator-only rows for the feature scaler (fixed before any real data)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, 10, n)
    arm = rng.integers(0, 3, n)
    both, _ = simulate(np.r_[idx, idx], np.r_[arm, arm], np.r_[np.ones(n), np.zeros(n)],
                    np.ones(2 * n), np.zeros(2 * n, bool), rng)
    return rows(both[:n], both[n:], arm)


def ebh(e, alpha=ALPHA):
    """e-BH: reject the k* hypotheses with the largest e-values, where k* is the
    largest k such that the k-th largest e-value is at least K / (alpha k)."""
    e = np.asarray(e, float)
    order = np.argsort(-e)
    kstar = max([k for k in range(1, len(e) + 1) if e[order[k - 1]] >= len(e) / (alpha * k)],
                default=0)
    return sorted(order[:kstar].tolist())


class Allocator:
    """Thompson sampling on mean TIR per arm, with a burn-in and a 0.1 floor."""

    def __init__(self, rng, prior_mean=0.6, prior_sd=0.2, noise_sd=0.15):
        self.rng = rng
        self.burn = np.concatenate([rng.permutation(3) for _ in range(BURN // 3)])
        self.m0, self.v0, self.s2 = prior_mean, prior_sd ** 2, noise_sd ** 2
        self.y = [[] for _ in range(3)]

    def probs(self):
        prec = 1 / self.v0 + np.array([len(y) for y in self.y]) / self.s2
        mean = (self.m0 / self.v0 + np.array([sum(y) for y in self.y]) / self.s2) / prec
        draws = self.rng.normal(mean, 1 / np.sqrt(prec), (1000, 3))
        pb = np.bincount(draws.argmax(1), minlength=3) / 1000.0
        pb = np.maximum(pb, 0.1)
        return pb / pb.sum()

    def next(self, i):
        if i < BURN:
            return int(self.burn[i])
        return int(self.rng.choice(3, p=self.probs()))

    def observe(self, arm, y):
        self.y[arm].append(float(y))


class Monitor:
    def __init__(self, scaler_rows, use_arm=True):
        self.blr = BLRRatio(feature_map=feature_map).fit_scaler(scaler_rows)
        self.w, self.wp = LambdaMixture(), LambdaMixture()
        self.use_arm = use_arm

    def step(self, rx, ry):
        sc = self.blr.score(np.vstack([rx, ry]))
        wv = self.w.update(mc_factor(sc[0], sc[1:]))
        wp = self.wp.update(plugin_factor(sc[0]))
        self.blr.update(rx, ry)
        return wv, wp


def run_chunk(seeds, scenario):
    """Replicate trials `seeds` of one scenario, stepped together."""
    R = len(seeds)
    rngs = [np.random.default_rng(s) for s in seeds]
    sim_rng = np.random.default_rng(seeds[0] + 7)
    scaler = pilot()
    alloc = [Allocator(r) for r in rngs]
    cond = [Monitor(scaler) for _ in range(R)]
    pooled = [Monitor(scaler, use_arm=False) for _ in range(R)]
    per_arm = [[Monitor(scaler) for _ in range(3)] for _ in range(R)]
    e_arm = np.ones((R, 3))
    frozen = np.zeros((R, 3), bool)
    rescues = np.zeros((R, 3), int)
    z = [[] for _ in range(R)]
    first = [dict(mc_blr=None, plugin=None, pooled=None, ttest=None) for _ in range(R)]
    arms_log = np.zeros((R, N_PART), int)
    tir_real = np.zeros((R, N_PART))
    tir_twin = np.zeros((R, N_PART))
    vmx_real = 0.75 if scenario == "resistance" else 1.0
    rescue_real = scenario == "rescue"
    K = 2 * M + 3   # real, M conditioned, M pooled, noise-free under arm, under A
    for i in range(N_PART):
        pat = np.array([r.integers(0, 10) for r in rngs])
        arm = np.array([alloc[j].next(i) for j in range(R)])
        arms_log[:, i] = arm
        idx, arms, noisy, vmx, resc = [], [], [], [], []
        for j in range(R):
            pooled_arms = sim_rng.integers(0, 3, M)
            idx += [pat[j]] * K
            arms += [arm[j]] + [arm[j]] * M + list(pooled_arms) + [arm[j], 0]
            noisy += [1] * (2 * M + 1) + [0, 0]
            vmx += [vmx_real] + [1.0] * (2 * M + 2)
            resc += [rescue_real] + [False] * (2 * M + 2)
        tr, n_resc = simulate(np.array(idx), np.array(arms), np.array(noisy, float),
                              np.array(vmx), np.array(resc), sim_rng)
        tr, n_resc = tr.reshape(R, K, STEPS), n_resc.reshape(R, K)
        arms = np.array(arms).reshape(R, K)
        for j in range(R):
            x, yc, yp = tr[j, :1], tr[j, 1:M + 1], tr[j, M + 1:2 * M + 1]
            g_arm, g_a = tr[j, 2 * M + 1], tr[j, 2 * M + 2]
            a_real, a_cond, a_pool = arms[j, :1], arms[j, 1:M + 1], arms[j, M + 1:2 * M + 1]
            wv, wp = cond[j].step(rows(x, g_arm, a_real), rows(yc, g_arm, a_cond))
            wq, _ = pooled[j].step(rows(x, g_a, a_real, False), rows(yp, g_a, a_pool, False))
            k = arm[j]
            rescues[j, k] += n_resc[j, 0]
            if not frozen[j, k]:
                w_k, _ = per_arm[j][k].step(rows(x, g_arm, a_real), rows(yc, g_arm, a_cond))
                e_arm[j, k] = w_k
                frozen[j, k] = w_k >= 3 / ALPHA
            t_x, t_y = tir(x)[0], tir(yc)
            tir_real[j, i], tir_twin[j, i] = t_x, t_y.mean()
            z[j].append((t_x - t_y.mean()) / (t_y.std() + 0.02))
            p = ttest_1samp(z[j], 0.0).pvalue if len(z[j]) >= 5 else 1.0
            for key, hit in (("mc_blr", wv >= THRESH), ("plugin", wp >= THRESH),
                             ("pooled", wq >= THRESH), ("ttest", p < ALPHA)):
                if hit and first[j][key] is None:
                    first[j][key] = i + 1
            alloc[j].observe(arm[j], t_x)
    return [dict(first=first[j], arms=arms_log[j].tolist(), tir_real=tir_real[j].tolist(),
                 tir_twin=tir_twin[j].tolist(), e_arm=e_arm[j].tolist(),
                 ebh=ebh(e_arm[j]), rescues=rescues[j].tolist()) for j in range(R)]


def summarise(runs):
    keys = ["mc_blr", "plugin", "pooled", "ttest"]
    rate = {k: float(np.mean([r["first"][k] is not None for r in runs])) for k in keys}
    med = {}
    for k in keys:
        hits = [r["first"][k] for r in runs if r["first"][k] is not None]
        med[k] = float(np.median(hits)) if hits else None
    arms = np.array([r["arms"] for r in runs])
    share = {a: float((arms[:, BURN:] == j).mean()) for j, a in enumerate(ARM_NAMES)}
    tr, tt = np.array([r["tir_real"] for r in runs]), np.array([r["tir_twin"] for r in runs])
    by_arm = {a: {"real": float(tr[arms == j].mean()), "twin": float(tt[arms == j].mean())}
              for j, a in enumerate(ARM_NAMES)}
    rej = np.zeros((len(runs), 3), bool)
    for i, r in enumerate(runs):
        rej[i, r["ebh"]] = True
    resc = np.array([r["rescues"] for r in runs])
    return dict(reject_rate=rate, median_first=med, allocation_after_burn_in=share,
                tir_by_arm=by_arm,
                ebh_reject_rate={a: float(rej[:, j].mean()) for j, a in enumerate(ARM_NAMES)},
                ebh_any=float(rej.any(1).mean()),
                median_e={a: float(np.median([r["e_arm"][j] for r in runs]))
                          for j, a in enumerate(ARM_NAMES)},
                rescues_per_trial={a: float(resc[:, j].mean()) for j, a in enumerate(ARM_NAMES)},
                trials_with_rescue={a: float((resc[:, j] > 0).mean())
                                    for j, a in enumerate(ARM_NAMES)},
                _rej=rej.tolist(), _rescues=resc.tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--chunks", type=int, default=4)
    args = ap.parse_args()
    res = {"config": dict(M=M, alpha=ALPHA, participants=N_PART, burn_in=BURN,
                          arms=ARMS, reps=args.reps)}
    for sc, offset in (("null", 0), ("resistance", 10 ** 5), ("rescue", 2 * 10 ** 5)):
        t0 = time.time()
        seeds = np.arange(args.reps) + offset
        chunks = [list(c) for c in np.array_split(seeds, args.chunks)]
        with ProcessPoolExecutor(args.chunks) as ex:
            runs = [r for part in ex.map(partial(run_chunk, scenario=sc), chunks) for r in part]
        res[sc] = summarise(runs)
        show = {k: v for k, v in res[sc].items() if not k.startswith("_")}
        print(f"[{sc}] {time.time() - t0:.0f}s", json.dumps(show), flush=True)
    (ROOT / "results" / "exp10.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
