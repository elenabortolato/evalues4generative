"""Experiment 9: real data with explicit actions. Italy's regional colour zones.

From November 2020 each of Italy's 21 regions was assigned, by government
ordinance, to a colour zone (white, yellow, orange, red) with specific
restrictions, based on monitoring indicators such as incidence and Rt. The
actions are therefore explicit, discrete, frequent (392 changes) and chosen
from the data.

Twin: e4g.epi_regional, a renewal model of daily new positives per region in
which transmission depends on the colour assigned 7 days earlier. Colour effects
(ordered, shared across regions), overdispersion and day-of-week factors are
estimated on 13 Nov - 20 Dec 2020; white zones, which start in March 2021,
are treated as yellow. Monitoring runs from 21 December 2020 to 31 December
2021.

Part A (real data)
  coverage   share of region-weeks whose observed total lies in the twin's 90%
             predictive interval, conditioning on the colour actually assigned
             versus assuming "always yellow" (first 12 weeks, frozen twin)
  regions    weekly monitor per region; after an alarm the region's baseline
             is recalibrated on its last 8 weeks and a new monitor starts
  national   average of the 21 regional wealth processes (no recalibration): a
             test martingale for "the twin holds in every region"
  e-BH       regions whose twin is rejected, with FDR control, from the stopped
             regional e-values (Wang & Ramdas 2022)

Part B (synthetic): 21-region series from the twin with colours set weekly by a
documented incidence rule (weekly cases per 100k > 250 red, > 150 orange, > 50
yellow, else white). Scenarios: null; variant (transmission x1.3 in every region
from week 8). Twins: "estimated" colour effects, and "strong" effects
(orange x0.85, red x0.70) as a sensitivity analysis.

Output: results/exp9.json, figures/exp9_italy_regions.png
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

from e4g import BLRRatio, LambdaMixture, mc_factor, plugin_factor  # noqa: E402
from e4g import epi_regional as ER  # noqa: E402
from e4g.epi import LAG  # noqa: E402

M, ALPHA, BLOCK, RECAL = 16, 0.05, 7, 56
THRESH = 1 / ALPHA
YELLOW = ER.CODE["yellow"]
DATA = ROOT / "data" / "italy_regions_daily.csv"


CAL_START, MON_START, MON_END = "2020-11-13", "2020-12-21", "2021-12-31"


def setup():
    regions, dates, y, c, wd = ER.load(DATA)
    last = int(np.where(dates == np.datetime64(MON_END))[0][0]) + 1
    dates, y, c, wd = dates[:last], y[:, :last], c[:, :last], wd[:last]
    start = int(np.where(dates == np.datetime64(CAL_START))[0][0])
    end = int(np.where(dates == np.datetime64(MON_START))[0][0])
    twin, _ = ER.fit(y, c, wd, start, end)
    twin, _ = ER.calibrate_sigma(twin, y, c, wd, start, end)
    return regions, dates, y, c, wd, start, end, twin


def strong(twin):
    beta = np.array([0.0, 0.0, np.log(0.85), np.log(0.70)])
    return ER.Twin(a=twin.a, beta=beta, k=twin.k, dow=twin.dow, sigma=twin.sigma)


_SETUP = None


def setup_cached():
    global _SETUP
    if _SETUP is None:
        _SETUP = setup()
    return _SETUP


def rows(traces, g, beta_eff):
    r = np.log1p(traces) - np.log1p(g)
    rbar = r.mean(1) / 0.1
    slope = (r[:, -3:].mean(1) - r[:, :3].mean(1)) / 0.1
    lev = np.full(len(r), np.log1p(g.mean()) - 6.0)
    u = np.full(len(r), beta_eff / 0.1)
    return np.column_stack([rbar, slope, lev, u])


def feature_map(z):
    rbar, slope, lev, u = z.T
    return np.column_stack([rbar, rbar ** 2, slope, rbar * u, rbar * lev])


class RegionMonitor:
    def __init__(self, twin, r, y, c, wd, t0, rng, colour=None):
        self.twin, self.r, self.colour = twin, r, colour
        sims = ER.simulate_block(twin, r, y[r], c[r], wd, t0, BLOCK, 200, rng, colour=colour)
        g = ER.simulate_block(twin, r, y[r], c[r], wd, t0, BLOCK, 1, rng, True, colour)[0]
        self.blr = BLRRatio(feature_map=feature_map).fit_scaler(
            rows(sims, g, self._beta(c, t0)))
        self.w, self.wp = LambdaMixture(), LambdaMixture()
        self.z = []

    def _beta(self, c, t0):
        col = c[self.r, t0 - LAG] if self.colour is None else self.colour
        return self.twin.beta[col]

    def week(self, y, c, wd, t0, rng):
        tw, r, col = self.twin, self.r, self.colour
        sims = ER.simulate_block(tw, r, y[r], c[r], wd, t0, BLOCK, M, rng, colour=col)
        g = ER.simulate_block(tw, r, y[r], c[r], wd, t0, BLOCK, 1, rng, True, col)[0]
        b = self._beta(c, t0)
        rx, ry = rows(y[r, None, t0:t0 + BLOCK], g, b), rows(sims, g, b)
        sc = self.blr.score(np.vstack([rx, ry]))
        wv = self.w.update(mc_factor(sc[0], sc[1:]))
        wp = self.wp.update(plugin_factor(sc[0]))
        self.blr.update(rx, ry)
        tot = sims.sum(1)
        self.z.append((y[r, t0:t0 + BLOCK].sum() - tot.mean()) / (tot.std() + 1e-9))
        p = ttest_1samp(self.z, 0.0).pvalue if len(self.z) >= 3 else 1.0
        return wv, wp, p


def ebh(e, alpha):
    """e-BH (Wang & Ramdas 2022): reject the k* largest e-values."""
    e = np.asarray(e)
    order = np.argsort(-e)
    n = len(e)
    kstar = 0
    for k in range(1, n + 1):
        if e[order[k - 1]] >= n / (alpha * k):
            kstar = k
    return sorted(order[:kstar].tolist())


# ------------------------------------------------------------ Part A: real data

def real_data(seed=0):
    regions, dates, y, c, wd, start, end, twin = setup_cached()
    R, T = y.shape
    rng = np.random.default_rng(seed)
    n_weeks = (T - end) // BLOCK

    # coverage of the weekly total, frozen twin, first 12 weeks
    cov = {"conditioned": [], "always_yellow": []}
    for r in range(R):
        for wk in range(12):
            t0 = end + wk * BLOCK
            obs = y[r, t0:t0 + BLOCK].sum()
            for key, col in (("conditioned", None), ("always_yellow", YELLOW)):
                sims = ER.simulate_block(twin, r, y[r], c[r], wd, t0, BLOCK, 400, rng,
                                         colour=col).sum(1)
                lo, hi = np.percentile(sims, [5, 95])
                cov[key].append(bool(lo <= obs <= hi))
    coverage = {k: float(np.mean(v)) for k, v in cov.items()}

    # per-region monitors with recalibration, and the national no-recal monitor
    mons = [RegionMonitor(twin, r, y, c, wd, end, rng) for r in range(R)]
    frozen = [RegionMonitor(twin, r, y, c, wd, end, rng) for r in range(R)]
    twins = [twin] * R
    alarms = {regions[r]: [] for r in range(R)}
    stopped = [None] * R
    national, national_alarm = [], None
    for wk in range(n_weeks):
        t0 = end + wk * BLOCK
        wf = []
        for r in range(R):
            wv, _, _ = mons[r].week(y, c, wd, t0, rng)
            if wv >= THRESH and t0 + 2 * BLOCK <= T:
                alarms[regions[r]].append(str(dates[t0 + BLOCK - 1].date()))
                twins[r] = ER.refit_region(twins[r], y, c, wd, r, t0 + BLOCK - RECAL,
                                           t0 + BLOCK)
                mons[r] = RegionMonitor(twins[r], r, y, c, wd, t0 + BLOCK, rng)
            if stopped[r] is None:
                w_frozen, _, _ = frozen[r].week(y, c, wd, t0, rng)
                if w_frozen >= THRESH:
                    stopped[r] = w_frozen
                wf.append(w_frozen)
            else:
                wf.append(stopped[r])
        national.append(float(np.mean(wf)))
        if national_alarm is None and national[-1] >= THRESH:
            national_alarm = str(dates[t0 + BLOCK - 1].date())
    final_e = [stopped[r] if stopped[r] is not None else float(frozen[r].w.value)
               for r in range(R)]
    rejected = [regions[i] for i in ebh(final_e, ALPHA)]
    return {"coverage_first12w": coverage, "alarms": alarms,
            "n_alarms": int(sum(len(v) for v in alarms.values())),
            "national_logw": [float(np.log(v)) for v in national],
            "national_first_alarm": national_alarm, "ebh_rejected": rejected,
            "twin": {"beta": twin.beta.tolist(), "k": twin.k, "sigma_week": twin.sigma,
                     "R_multiplier": np.exp(twin.beta).tolist()},
            "monitoring": [str(dates[end].date()), str(dates[end + n_weeks * BLOCK - 1].date())],
            "regions": regions}


# -------------------------------------------------------- Part B: synthetic data

def tier(weekly_cases, pop_millions):
    inc = weekly_cases / (pop_millions * 10.0)  # per 100k
    return (ER.CODE["red"] if inc > 250 else ER.CODE["orange"] if inc > 150
            else YELLOW if inc > 50 else ER.CODE["white"])


def synthetic_one(seed, scenario, effects):
    regions, dates, y_real, c_real, wd, start, end, twin = setup_cached()
    if effects == "strong":
        twin = strong(twin)  # synthetic data are generated from the twin monitored
    rng = np.random.default_rng(seed)
    R, T = y_real.shape
    n_weeks = (T - end) // BLOCK
    y, c = y_real.copy(), c_real.copy()
    mons = {k: [RegionMonitor(twin, r, y, c, wd, end, rng, colour=col) for r in range(R)]
            for k, col in (("cond", None), ("yellow", YELLOW))}
    first = {k: [None] * R for k in ("mc_blr", "plugin", "always_yellow", "ttest")}
    nat = {"mc_blr": None, "always_yellow": None}
    wealth = {"mc_blr": np.ones(R), "always_yellow": np.ones(R)}
    for wk in range(n_weeks):
        t0 = end + wk * BLOCK
        shift = np.log(1.3) if (scenario == "variant" and wk >= 8) else 0.0
        for r in range(R):
            c[r, t0:t0 + BLOCK] = tier(y[r, t0 - BLOCK:t0].sum(), ER.POP[regions[r]])
            y[r, t0:t0 + BLOCK] = ER.simulate_block(twin, r, y[r], c[r], wd, t0, BLOCK, 1,
                                                    rng, a_shift=shift)[0]
        for r in range(R):
            wv, wp, p = mons["cond"][r].week(y, c, wd, t0, rng)
            wy, _, _ = mons["yellow"][r].week(y, c, wd, t0, rng)
            wealth["mc_blr"][r], wealth["always_yellow"][r] = wv, wy
            for k, hit in (("mc_blr", wv >= THRESH), ("plugin", wp >= THRESH),
                           ("always_yellow", wy >= THRESH), ("ttest", p < ALPHA)):
                if hit and first[k][r] is None:
                    first[k][r] = wk + 1
        for k in nat:
            if nat[k] is None and wealth[k].mean() >= THRESH:
                nat[k] = wk + 1
    return {"first": first, "national": nat}


def summarise(runs, scenario):
    out = {}
    for k in runs[0]["first"]:
        t = np.array([[x if x is not None else np.inf for x in r["first"][k]] for r in runs])
        if scenario == "null":
            out[k] = {"regional_false_alarm": float(np.isfinite(t).mean())}
        else:
            late = np.isfinite(t) & (t > 8)
            out[k] = {"regional_alarm_before_change": float((t <= 8).mean()),
                      "regional_detected": float(late.mean()),
                      "median_delay_weeks": float(np.median(t[late] - 8)) if late.any() else None}
    for k in runs[0]["national"]:
        t = np.array([r["national"][k] if r["national"][k] is not None else np.inf for r in runs])
        key = f"national_{k}"
        if scenario == "null":
            out[key] = {"false_alarm": float(np.isfinite(t).mean())}
        else:
            late = np.isfinite(t) & (t > 8)
            out[key] = {"alarm_before_change": float((t <= 8).mean()),
                        "detected": float(late.mean()),
                        "median_delay_weeks": float(np.median(t[late] - 8)) if late.any() else None}
    return out


def main():
    t_start = time.time()
    res = {"config": dict(m=M, alpha=ALPHA, block_days=BLOCK, recal_days=RECAL, lag=LAG)}
    res["real"] = real_data()
    print("coverage:", res["real"]["coverage_first12w"])
    print("alarms:", res["real"]["n_alarms"], "national first alarm:",
          res["real"]["national_first_alarm"], "e-BH rejected:", res["real"]["ebh_rejected"],
          flush=True)
    res["synthetic"] = {}
    for effects in ("estimated", "strong"):
        for sc, n in (("null", 100), ("variant", 60)):
            with ProcessPoolExecutor() as ex:
                runs = list(ex.map(partial(synthetic_one, scenario=sc, effects=effects),
                                   range(n), chunksize=2))
            res["synthetic"][f"{effects}/{sc}"] = summarise(runs, sc)
            print(f"{effects}/{sc}:", res["synthetic"][f"{effects}/{sc}"], flush=True)
    print(f"{time.time() - t_start:.0f}s")
    (ROOT / "results" / "exp9.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
