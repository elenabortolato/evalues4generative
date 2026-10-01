"""Experiment 11: a multi-arm multi-stage (MAMS) trial checked against its twin.

Design. Control (A) and four experimental insulin-dosing strategies (B-E), run
by the pump in closed loop, compared on time in range (TIR, 70-180 mg/dL) over
full days (06:00-06:00; meals of 50, 70 and 80 g at 07:00, 12:30 and 19:00):

  A  control       carbohydrate bolus, correction to 140 above 150, suspend < 80
  B  aggressive    bolus x1.5, correction to 100, suspend < 60
  C  conservative  bolus x0.8, correction to 160, suspend < 90
  D  intensified   bolus x1.2, correction to 120, suspend < 70
  E  tight corr.   bolus x1.0, correction to 110, suspend < 80

Three stages. In each stage, 50 new participants (virtual adults drawn from
simglucose's 10) are enrolled and allocated in permuted blocks among the
arms still active, and every participant already in the trial is observed for
one more day. Participants carry their physiological state from one stage to
the next. After stages 1 and 2 an interim analysis compares each active
experimental arm with control on all participant-days so far, and drops it for
futility (bounds below). Participants on a
dropped arm are RE-ASSIGNED to the active experimental arm with the highest
interim TIR (control if none is left); new participants are allocated only to
active arms. The trial has 150 participants and 300 participant-days.

Twin: the UVA/Padova simulator (e4g.t1d) with the participant's virtual-patient
parameters, RE-INITIALISED AT THE PARTICIPANT'S CURRENT STATE and run M = 16
times under the arm assigned for that day. Participant-days are monitored in a
fixed order within each stage.

Scenarios for the real participants:
  null        participants are the simulator
  resistance  insulin sensitivity 25% lower, in every arm
  snacking    participants told that their regimen is intensified (arms B and
              D) eat an unannounced 20 g snack at bedtime (22:00) against
              night-time lows. The simulator does not know this; arms A, C and
              E remain exactly as the twin predicts, so the FDR of e-BH across
              arms is meaningful.

Interim analysis: futility against control when z < -1.0 (interim 1) or
z < 0 (interim 2).

Monitors (alpha = 0.05):
  mc_blr     MC e-process, BLR predictive score, twin at the participant's
             state under the assigned arm (the paper's method)
  plugin     same score, bet directly
  stateless  twin under the assigned arm, but started from the resting state:
             ignores the participant's history (carry-over, re-assignment)
  pooled     twin started from rest under an arm drawn uniformly from the five:
             the pre-trial in-silico prediction, ignoring allocation and history
  ttest      t-test of standardised TIR residuals, recomputed every day
  per arm    one e-process per arm (bets only on that arm's participant-days),
             e-BH across the five arms at the end of the trial (Xu, Wang &
             Ramdas 2021), each e-value frozen when it reaches 5 / alpha

Output: results/exp11.json
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

from e4g import (BLRRatio, ChoiceBLR, LambdaMixture, MartingalePredictive, mc_factor,  # noqa: E402
                 plugin_factor, t1d)

ADULTS = [f"adult#{i:03d}" for i in range(1, 11)]
ARMS = {"A": (1.0, 140.0, 80.0), "B": (1.5, 100.0, 60.0), "C": (0.8, 160.0, 90.0),
        "D": (1.2, 120.0, 70.0), "E": (1.0, 110.0, 80.0)}
NAMES = list(ARMS)
NA = len(ARMS)
ARM_PAR = np.array([ARMS[a] for a in NAMES])
STEPS = 288                                   # 24 h of 5-minute intervals from 06:00
MEALS = {12: 50.0, 78: 70.0, 156: 80.0}
STAGES, NEW_PER_STAGE, FUTILITY = 3, 50, (-1.0, 0.0)
BEDTIME, SNACK = 192, 20.0                   # 22:00, grams (snacking scenario)
M, ALPHA = 16, 0.05
THRESH = 1 / ALPHA
VAR = t1d.Variability(p_snack=0.0)
COLS = [c for c in t1d.PARAMS.columns if c != "Name"]
_PT = t1d.PARAMS.set_index("Name").loc[ADULTS]
_QT = t1d.QUEST.set_index("Name").loc[ADULTS]
X_REST = _PT.iloc[:, 1:14].values.astype(float)      # (10, 13) resting states


def params(idx):
    p = SimpleNamespace(**{c: _PT[c].values.astype(float)[idx] for c in COLS})
    ctl = SimpleNamespace(basal=p.u2ss * p.BW / 6000.0, cr=_QT.CR.values[idx],
                          cf=_QT.CF.values[idx])
    return p, ctl


def simulate(x_start, idx, arm, noisy, vmx, rescue, rng, snack=None):
    """One 24-hour day for K trajectories. Returns CGM (K, STEPS), the final ODE
    state (K, 13) and the number of rescue treatments (K,). `snack` (K,) bool:
    an unannounced bedtime snack (no bolus)."""
    p, ctl = params(idx)
    k = len(idx)
    b = t1d.Batch(x=x_start.copy(), last_qsto=x_start[:, 0] + x_start[:, 1],
                  last_food=np.zeros(k), eating=np.zeros(k, bool), planned=np.zeros(k),
                  last_cho=np.zeros(k))
    mult_b, target, suspend = ARM_PAR[arm].T
    cgm = t1d.bg(b, p)
    since, n_rescue = np.full(k, 99), np.zeros(k, int)
    out = np.empty((k, STEPS))
    for s in range(STEPS):
        meal = MEALS.get(s, 0.0)
        rate = np.where(cgm < suspend, 0.0, ctl.basal)
        if meal > 0:
            bolus = mult_b * meal / ctl.cr + (cgm > 150) * (cgm - target) / ctl.cf
            rate = rate + np.maximum(bolus, 0.0) / 5.0
        take = rescue & (cgm < 70) & (since >= 6)
        since = np.where(take, 0, since + 1)
        n_rescue += take
        carbs = meal * np.exp(VAR.sigma_carb * rng.standard_normal(k) * noisy) + 15.0 * take
        if snack is not None and s == BEDTIME:
            carbs = carbs + SNACK * snack
        m_is = np.exp(VAR.sigma_is * rng.standard_normal(k) * noisy) * vmx
        for i in range(5):
            t1d.minute(b, p, carbs if i == 0 else 0.0, rate, vmx_mult=m_is, kp3_mult=m_is)
        cgm = np.clip(t1d.bg(b, p) + noisy * t1d.johnson_noise(rng, k), 39.0, 600.0)
        out[:, s] = cgm
    return out, b.x, n_rescue


def tir(traces):
    return ((traces >= 70) & (traces <= 180)).mean(1)


def rows(traces, g, arm, use_arm=True):
    r = traces - g
    onehot = (np.asarray(arm)[:, None] == np.arange(1, NA)[None, :]) * float(use_arm)
    onehot = np.broadcast_to(onehot, (len(traces), NA - 1))
    return np.column_stack([r.mean(1) / 20.0, r.std(1) / 20.0,
                            (traces < 70).mean(1) * 10.0, (traces > 180).mean(1) * 5.0,
                            onehot])


def feature_map(z):
    rbar, rsd, tbr, tar, oh = z[:, 0], z[:, 1], z[:, 2], z[:, 3], z[:, 4:]
    return np.column_stack([rbar, rbar ** 2, rsd, tbr, tar, rbar[:, None] * oh,
                            tbr[:, None] * oh])


def ebh(e, alpha=ALPHA):
    e = np.asarray(e, float)
    order = np.argsort(-e)
    kstar = max([k for k in range(1, len(e) + 1) if e[order[k - 1]] >= len(e) / (alpha * k)],
                default=0)
    return sorted(order[:kstar].tolist())


def pilot(seed=10 ** 6, n=200):
    """Simulator-only rows for the feature scaler, fixed before any real data."""
    rng = np.random.default_rng(seed)
    idx, arm = rng.integers(0, 10, n), rng.integers(0, NA, n)
    x = X_REST[np.r_[idx, idx]]
    tr, _, _ = simulate(x, np.r_[idx, idx], np.r_[arm, arm], np.r_[np.ones(n), np.zeros(n)],
                        np.ones(2 * n), np.zeros(2 * n, bool), rng)
    return rows(tr[:n], tr[n:], arm)


MP_LEARNERS = {"mp-logistic-plugin": dict(loss="logistic", mode="plugin"),
               "mp-logistic-resample": dict(loss="logistic", mode="resample"),
               "mp-choice-plugin": dict(loss="choice", mode="plugin", prior_var=0.03),
               "mp-choice-resample": dict(loss="choice", mode="resample", prior_var=0.03)}
LEARNER = "logistic"   # or "choice" (BLR on the simulation likelihood), or a key of MP_LEARNERS


class Monitor:
    def __init__(self, scaler):
        if LEARNER == "choice":
            self.blr = ChoiceBLR(feature_map=feature_map, prior_var=0.03).fit_scaler(scaler)
        elif LEARNER in MP_LEARNERS:
            self.blr = MartingalePredictive(feature_map=feature_map,
                                            **MP_LEARNERS[LEARNER]).fit_scaler(scaler)
        else:
            self.blr = BLRRatio(feature_map=feature_map).fit_scaler(scaler)
        self.w, self.wp = LambdaMixture(), LambdaMixture()

    def step(self, rx, ry):
        if LEARNER == "choice" or LEARNER in MP_LEARNERS:
            wv = self.w.update(self.blr.factor(rx, ry))
            wp = 0.0               # no plug-in bet for the conditional logit
        else:
            sc = self.blr.score(np.vstack([rx, ry]))
            wv = self.w.update(mc_factor(sc[0], sc[1:]))
            wp = self.wp.update(plugin_factor(sc[0]))
        self.blr.update(rx, ry)
        return wv, wp


def run_trial(seed, scenario, scaler):
    rng = np.random.default_rng(seed)           # design and participants
    sim = np.random.default_rng(seed + 10 ** 7)  # physiology and simulator
    vmx_real = 0.75 if scenario == "resistance" else 1.0
    rescue_real = False
    snack_arms = {1, 3} if scenario == "snacking" else set()
    active = list(range(NA))
    pat, arm, state = [], [], []                 # per participant
    days_arm, days_tir = [], []                  # per participant-day (real)
    mon = {k: Monitor(scaler) for k in ("mc_blr", "stateless", "pooled")}
    per_arm = [Monitor(scaler) for _ in range(NA)]
    e_arm, frozen = np.ones(NA), np.zeros(NA, bool)
    z, first, t = [], dict(mc_blr=None, plugin=None, stateless=None, pooled=None,
                           ttest=None), 0
    rescues = np.zeros(NA, int)
    drops, reassigned = [], 0
    K = 3 * M + 4
    for stage in range(STAGES):
        blocks = np.concatenate([rng.permutation(active)
                                 for _ in range(NEW_PER_STAGE // len(active) + 1)])
        for i in range(NEW_PER_STAGE):
            p = int(rng.integers(0, 10))
            pat.append(p)
            arm.append(int(blocks[i]))
            state.append(X_REST[p].copy())
        n = len(pat)
        # batch: real, M at state, M from rest, M pooled from rest, 3 noise-free paths
        xs, idx, arms, noisy, vmx, resc, snk = [], [], [], [], [], [], []
        for j in range(n):
            a, pj = arm[j], pat[j]
            pool = sim.integers(0, NA, M)
            xs += [state[j]] * (M + 1) + [X_REST[pj]] * (2 * M) + [state[j], X_REST[pj],
                                                                    X_REST[pj]]
            idx += [pj] * K
            arms += [a] * (2 * M + 1) + list(pool) + [a, a, 0]
            noisy += [1.0] * (3 * M + 1) + [0.0] * 3
            vmx += [vmx_real] + [1.0] * (3 * M + 3)
            resc += [rescue_real] + [False] * (3 * M + 3)
            snk += [a in snack_arms] + [False] * (3 * M + 3)
        tr, xend, nres = simulate(np.array(xs), np.array(idx), np.array(arms),
                                  np.array(noisy), np.array(vmx), np.array(resc), sim,
                                  snack=np.array(snk))
        tr, xend, nres = tr.reshape(n, K, STEPS), xend.reshape(n, K, 13), nres.reshape(n, K)
        arms_b = np.array(arms).reshape(n, K)
        for j in range(n):
            a = arm[j]
            x = tr[j, :1]
            ys, yr, yp = tr[j, 1:M + 1], tr[j, M + 1:2 * M + 1], tr[j, 2 * M + 1:3 * M + 1]
            gs, gr, gp = tr[j, 3 * M + 1], tr[j, 3 * M + 2], tr[j, 3 * M + 3]
            ar, aa = arms_b[j, :1], np.full(M, a)
            wv, wp = mon["mc_blr"].step(rows(x, gs, ar), rows(ys, gs, aa))
            ws, _ = mon["stateless"].step(rows(x, gr, ar), rows(yr, gr, aa))
            wq, _ = mon["pooled"].step(rows(x, gp, ar, False),
                                       rows(yp, gp, arms_b[j, 2 * M + 1:3 * M + 1], False))
            if not frozen[a]:
                e_arm[a], _ = per_arm[a].step(rows(x, gs, ar), rows(ys, gs, aa))
                frozen[a] = e_arm[a] >= NA / ALPHA
            t_x, t_y = tir(x)[0], tir(ys)
            z.append((t_x - t_y.mean()) / (t_y.std() + 0.02))
            pv = ttest_1samp(z, 0.0).pvalue if len(z) >= 5 else 1.0
            t += 1
            for key, hit in (("mc_blr", wv >= THRESH), ("plugin", wp >= THRESH),
                             ("stateless", ws >= THRESH), ("pooled", wq >= THRESH),
                             ("ttest", pv < ALPHA)):
                if hit and first[key] is None:
                    first[key] = t
            days_arm.append(a)
            days_tir.append(t_x)
            rescues[a] += nres[j, 0]
            state[j] = xend[j, 0].copy()
        if stage == STAGES - 1:
            break
        # interim analysis: futility against control on all participant-days so far
        da, dt = np.array(days_arm), np.array(days_tir)
        sd = dt.std() + 1e-9
        mean = {k: dt[da == k].mean() for k in active}
        dropped = []
        for k in active:
            if k == 0:
                continue
            nk, n0 = (da == k).sum(), (da == 0).sum()
            zk = (mean[k] - mean[0]) / (sd * np.sqrt(1 / nk + 1 / n0))
            if zk < FUTILITY[stage]:
                dropped.append(k)
        active = [k for k in active if k not in dropped]
        drops.append([NAMES[k] for k in dropped])
        exp_active = [k for k in active if k != 0]
        target = max(exp_active, key=lambda k: mean[k]) if exp_active else 0
        for j in range(len(arm)):
            if arm[j] in dropped:
                arm[j] = target
                reassigned += 1
    da, dt = np.array(days_arm), np.array(days_tir)
    return dict(first=first, e_arm=e_arm.tolist(), ebh=ebh(e_arm), drops=drops,
                active_final=[NAMES[k] for k in active], reassigned=reassigned,
                days_per_arm=np.bincount(da, minlength=NA).tolist(),
                tir_per_arm=[float(dt[da == k].mean()) if (da == k).any() else None
                             for k in range(NA)],
                rescues=rescues.tolist())


def run_many(seeds, scenario):
    scaler = pilot()
    return [run_trial(int(s), scenario, scaler) for s in seeds]


def summarise(runs):
    keys = ["mc_blr", "plugin", "stateless", "pooled", "ttest"]
    out = {"reject_rate": {k: float(np.mean([r["first"][k] is not None for r in runs]))
                           for k in keys}}
    out["median_first_day"] = {}
    for k in keys:
        hits = [r["first"][k] for r in runs if r["first"][k] is not None]
        out["median_first_day"][k] = float(np.median(hits)) if hits else None
    rej = np.zeros((len(runs), NA), bool)
    for i, r in enumerate(runs):
        rej[i, r["ebh"]] = True
    out["ebh_reject_rate"] = {a: float(rej[:, j].mean()) for j, a in enumerate(NAMES)}
    out["ebh_any"] = float(rej.any(1).mean())
    out["dropped_at_interim"] = [{a: float(np.mean([a in r["drops"][s] for r in runs]))
                                  for a in NAMES[1:]} for s in range(STAGES - 1)]
    out["active_at_end"] = {a: float(np.mean([a in r["active_final"] for r in runs]))
                            for a in NAMES}
    out["reassigned_mean"] = float(np.mean([r["reassigned"] for r in runs]))
    out["days_share"] = {a: float(np.mean([r["days_per_arm"][j] for r in runs]) / 300)
                         for j, a in enumerate(NAMES)}
    out["rescues_per_trial"] = {a: float(np.mean([r["rescues"][j] for r in runs]))
                                for j, a in enumerate(NAMES)}
    out["_ebh"] = rej.tolist()
    out["_rescues"] = [r["rescues"] for r in runs]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--learner", default="logistic", choices=["logistic", "choice", *MP_LEARNERS])
    ap.add_argument("--scenarios", default="null,resistance,snacking")
    args = ap.parse_args()
    global LEARNER
    LEARNER = args.learner
    wanted = args.scenarios.split(",")
    res = {"config": dict(M=M, alpha=ALPHA, stages=STAGES, new_per_stage=NEW_PER_STAGE,
                          futility=FUTILITY, arms=ARMS, reps=args.reps, learner=LEARNER)}
    for sc, offset in (("null", 0), ("resistance", 10 ** 5), ("snacking", 2 * 10 ** 5)):
        if sc not in wanted:
            continue
        t0 = time.time()
        seeds = np.arange(args.reps) + offset
        chunks = [list(c) for c in np.array_split(seeds, args.workers * 4)]
        with ProcessPoolExecutor(args.workers) as ex:
            runs = [r for part in ex.map(partial(run_many, scenario=sc), chunks) for r in part]
        res[sc] = summarise(runs)
        show = {k: v for k, v in res[sc].items() if not k.startswith("_")}
        print(f"[{sc}] {time.time() - t0:.0f}s", json.dumps(show), flush=True)
    name = "exp11.json" if LEARNER == "logistic" else f"exp11_{LEARNER}.json"
    if LEARNER in MP_LEARNERS:
        name = f"exp11_{LEARNER}_{'-'.join(wanted)}.json"
    (ROOT / "results" / name).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
