"""Experiment 7: monitoring an insulin-dosing controller against its digital twin.

Application (Section 6.6 of the paper). A virtual type 1 diabetes patient of the
UVA/Padova simulator (simglucose's adult#001, via the vectorised port e4g.t1d)
is treated by a feedback controller. Every 30 minutes (an "epoch") the
controller commits to a basal rate from the current CGM reading: basal is
suspended below 80 mg/dL and a correction is added above 180 mg/dL. Meal
boluses follow the announced carbohydrates. The digital twin is the same
simulator, with the patient's parameters, re-initialised at the state at the
start of each epoch and run M = 16 times under the committed dosing plan; the
monitor bets on the 30 one-minute CGM readings of the epoch.

Scenarios (48 hours; changes start at hour 12):
  null        the patient is the twin
  resistance  insulin sensitivity (Vmx, kp3) falls by 30%, as in illness
  sensor      CGM gain drifts linearly from 1 to 1.15 over the following 24 h
  snacking    unannounced snacks become six times more frequent

Methods (alpha = 0.05):
  mc_blr      Monte Carlo e-process, BLR predictive score (the paper's method)
  mc_fixed    Monte Carlo e-process, fixed score: the mean residual (no learning)
  nominal     as mc_blr, but the twin is run under the OPEN-LOOP plan (basal and
              meal boluses, no suspend or correction), i.e. not conditioned on
              the action actually taken
  ttest       t-test of standardised epoch residuals, recomputed every epoch

Output: results/exp7.json
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
from scipy.stats import ttest_1samp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from e4g import BLRRatio, LambdaMixture, mc_factor, t1d  # noqa: E402

NAME = "adult#001"
P = t1d.patient(NAME)
CTL = t1d.Controller(NAME)
VAR = t1d.Variability()
L = 6                       # intervals of 5 minutes per epoch (30 minutes)
HOURS, CHANGE_H = 48, 12
EPOCHS = HOURS * 12 // L
CHANGE = CHANGE_H * 12 // L  # first epoch after the change
M, ALPHA = 16, 0.05
METHODS = ["mc_blr", "mc_fixed", "nominal", "ttest"]


def committed_rate(cgm):
    if cgm < 80:
        return 0.0
    return CTL.basal + max(0.0, cgm - 180.0) / CTL.cf / (L * 5)


def block(b, rng, rate, meals, det=False, vmx=1.0, gain=1.0, p_snack=None):
    var = VAR if p_snack is None else t1d.Variability(p_snack=p_snack)
    out = []
    for meal in meals:
        ins = rate + (meal / CTL.cr) / 5 if meal > 0 else rate
        out.append(t1d.interval(b, P, rng, ins, meal, var, deterministic=det,
                                vmx_factor=vmx, kp3_factor=vmx, sensor_gain=gain))
    return np.concatenate(out, axis=1)


def rows(traces, g, rate):
    """Per-trace summary relative to the twin's noise-free prediction g."""
    r = traces - g
    rbar = r.mean(1) / 5.0
    rend = r[:, -5:].mean(1) / 10.0
    slope = (r[:, -10:].mean(1) - r[:, :10].mean(1)) / 10.0
    c = np.full(len(r), (g.mean() - 140.0) / 50.0)
    u = np.full(len(r), np.log1p(rate / CTL.basal))
    return np.column_stack([rbar, rend, slope, c, u])


def feature_map(z):
    rbar, rend, slope, c, u = z.T
    return np.column_stack([rbar, rbar ** 2, rend, slope, rbar * c, rbar * u, rend * u])


def scenario(name, epoch):
    """Real-system modifications at a given epoch."""
    if epoch < CHANGE or name == "null":
        return dict(vmx=1.0, gain=1.0, p_snack=None)
    if name == "resistance":
        return dict(vmx=0.7, gain=1.0, p_snack=None)
    if name == "sensor":
        frac = min(1.0, (epoch - CHANGE) / (24 * 12 / L))
        return dict(vmx=1.0, gain=1.0 + 0.15 * frac, p_snack=None)
    if name == "snacking":
        return dict(vmx=1.0, gain=1.0, p_snack=6 * VAR.p_snack)
    raise ValueError(name)


def pilot_rows(seed):
    """Scaler rows from the twin alone (independent of the monitored data)."""
    rng = np.random.default_rng(10**7 + seed)
    plan = t1d.meal_plan(rng, 1)
    b = t1d.Batch.initial(P)
    cgm, out = float(t1d.bg(b, P)[0]), []
    for e in range(24 * 12 // L):
        meals = plan[e * L:(e + 1) * L]
        rate = committed_rate(cgm)
        g = block(b.repeat(1), rng, rate, meals, det=True)[0]
        y = block(b.repeat(8), rng, rate, meals)
        out.append(rows(y, g, rate))
        block(b, rng, rate, meals)
        cgm = float(t1d.bg(b, P)[0])
    return np.vstack(out)


def run_one(seed, name):
    rng = np.random.default_rng(seed)
    plan = t1d.meal_plan(rng, HOURS // 24)
    real = t1d.Batch.initial(P)
    cgm = float(t1d.bg(real, P)[0])
    blr = BLRRatio(feature_map=feature_map).fit_scaler(pilot_rows(seed))
    blr_nom = BLRRatio(feature_map=feature_map).fit_scaler(pilot_rows(seed))
    proc = {k: LambdaMixture() for k in ("mc_blr", "mc_fixed", "nominal")}
    first = {k: None for k in METHODS}
    zs = []
    for e in range(EPOCHS):
        meals = plan[e * L:(e + 1) * L]
        rate = committed_rate(cgm)
        nominal_rate = CTL.basal
        # batched twin runs: noise-free predictions, then M draws per plan
        both = np.array([rate, nominal_rate])
        gg = block(real.repeat(2), rng, both, meals, det=True)
        g, g_nom = gg[0], gg[1]
        yy = block(real.repeat(2 * M), rng, np.repeat(both, M), meals)
        y, y_nom = yy[:M], yy[M:]
        x = block(real, rng, rate, meals, **scenario(name, e))[0][None, :]

        rx, ry = rows(x, g, rate), rows(y, g, rate)
        rxn, ryn = rows(x, g_nom, nominal_rate), rows(y_nom, g_nom, nominal_rate)
        z_all = np.vstack([rx, ry])
        e_blr = mc_factor(*(lambda s: (s[0], s[1:]))(blr.score(z_all)))
        e_fix = mc_factor(z_all[0, 0], z_all[1:, 0])
        zn = np.vstack([rxn, ryn])
        e_nom = mc_factor(*(lambda s: (s[0], s[1:]))(blr_nom.score(zn)))
        for k, v in (("mc_blr", e_blr), ("mc_fixed", e_fix), ("nominal", e_nom)):
            if proc[k].update(v) >= 1 / ALPHA and first[k] is None:
                first[k] = e + 1
        blr.update(rx, ry)
        blr_nom.update(rxn, ryn)

        # classical: standardised epoch residual, t-test recomputed every epoch
        zs.append((rx[0, 0] - ry[:, 0].mean()) / (ry[:, 0].std() + 1e-9))
        if first["ttest"] is None and len(zs) >= 5 and ttest_1samp(zs, 0.0).pvalue < ALPHA:
            first["ttest"] = e + 1
        cgm = float(x[0, -1])
    return first


def summarise(runs, name):
    out = {}
    for k in METHODS:
        t = np.array([r[k] if r[k] is not None else np.inf for r in runs])
        early = t <= CHANGE
        if name == "null":
            out[k] = {"false_alarm": float(np.isfinite(t).mean())}
        else:
            late = np.isfinite(t) & ~early
            delay = (t[late] - CHANGE) * L * 5 / 60.0  # hours after the change
            out[k] = {"alarm_before_change": float(early.mean()),
                      "detected": float(late.mean()),
                      "median_delay_h": float(np.median(delay)) if late.any() else None}
    return out


def main():
    reps = {"null": 200, "resistance": 80, "sensor": 80, "snacking": 80}
    res = {"config": dict(patient=NAME, epoch_min=L * 5, hours=HOURS, change_h=CHANGE_H,
                          m=M, alpha=ALPHA, reps=reps)}
    for name, n in reps.items():
        t0 = time.time()
        with ProcessPoolExecutor() as ex:
            runs = list(ex.map(partial(run_one, name=name), range(n), chunksize=2))
        res[name] = summarise(runs, name)
        print(f"[{name}] {time.time() - t0:.0f}s  {res[name]}", flush=True)
    (ROOT / "results" / "exp7.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
