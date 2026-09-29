"""Experiment 8b: Italy 2020-2021, daily COVID-19 deaths and national stringency.

Same design as Experiment 8, moved to the core period of the pandemic and to an
outcome available from the start: daily deaths (difference of the Protezione
Civile cumulative count; one negative reporting correction set to zero).
Restrictions act on deaths with a lag of about three weeks, so the renewal
twin uses the stringency 21 days earlier (still longer than the 7-day block, so
conditioning on the policy in force is exact).

  calibration  1 April - 26 May 2020 (8 weeks, lockdown and reopening)
  monitoring   27 May 2020 - 31 December 2021, recalibration after each alarm
  synthetic    series from the calibrated twin with a tier policy on weekly
               deaths (stringency 80 / 60 / 40 above 1500 / 500 / otherwise);
               null, and a variant multiplying transmission by 1.4 from week 10
  sensitivity  policy effect b in {-0.5, -1, -2}; free estimates on different
               2020 windows range from -0.4 to -3 (the bound)

Output: results/exp8b.json, figures/exp8b_italy_deaths.png
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from e4g import epi  # noqa: E402
import exp8_italy_covid as E8  # noqa: E402

epi.LAG = 21
CAL_START, MON_START, MON_END = "2020-04-01", "2020-05-27", "2021-12-31"


def load():
    d = pd.read_csv(ROOT / "data" / "italy_covid_daily.csv", parse_dates=["date"])
    d["deaths"] = d.deaths_cumulative.diff().clip(lower=0)
    d = d[(d.date >= "2020-02-24") & (d.date <= MON_END)].reset_index(drop=True)
    y = d.deaths.fillna(0).values.astype(float)
    s = d.stringency.ffill().bfill().values.astype(float)
    wd = d.date.dt.weekday.values
    return d, y, s, wd


def index(d, day):
    return int(np.where(d.date == pd.Timestamp(day))[0][0])


def real_data(b=-1.0, seed=0):
    d, y, s, wd = load()
    rng = np.random.default_rng(seed)
    t_cal, t0 = index(d, CAL_START), index(d, MON_START)
    twin = epi.fit(y, s, wd, t_cal, t0, b_fixed=b)
    mon = E8.Monitor(twin, y, s, wd, t0, rng)
    log = {"date": [], "logw": [], "lo": [], "hi": [], "segment": []}
    alarms, seg = [], 0
    while t0 + E8.BLOCK <= len(y):
        wv, _, _, _ = mon.week(y, s, wd, t0, rng)
        band = epi.simulate_block(mon.twin, y, s, wd, t0, E8.BLOCK, 400, rng)
        for j in range(E8.BLOCK):
            log["date"].append(str(d.date[t0 + j].date()))
            log["lo"].append(float(np.percentile(band[:, j], 5)))
            log["hi"].append(float(np.percentile(band[:, j], 95)))
            log["segment"].append(seg)
        log["logw"].append(float(np.log(wv)))
        t0 += E8.BLOCK
        if wv >= E8.THRESH and t0 + E8.BLOCK <= len(y):
            alarms.append({"date": str(d.date[t0 - 1].date()), "weeks_monitored": len(mon.z)})
            twin = epi.fit(y, s, wd, t0 - E8.CALIB, t0, b_fixed=b)
            mon = E8.Monitor(twin, y, s, wd, t0, rng)
            seg += 1
    return {"alarms": alarms, "log": log, "b": b,
            "first_alarm": alarms[0]["date"] if alarms else None}


def tier(weekly_deaths):
    return 80.0 if weekly_deaths > 1500 else 60.0 if weekly_deaths > 500 else 40.0


def synthetic_one(seed, scenario, b=-1.0):
    d, y_real, s_real, wd = load()
    rng = np.random.default_rng(seed)
    t_cal, t_start = index(d, CAL_START), index(d, MON_START)
    twin = epi.fit(y_real, s_real, wd, t_cal, t_start, b_fixed=b)
    n_weeks = (len(y_real) - t_start) // E8.BLOCK
    y, s = y_real.copy(), s_real.copy()
    nominal = float(s_real[t_cal - epi.LAG:t_start - epi.LAG].mean())
    mons = {"mc": E8.Monitor(twin, y, s, wd, t_start, rng),
            "nominal": E8.Monitor(twin, y, s, wd, t_start, rng, policy=nominal)}
    first = {k: None for k in ("mc_blr", "plugin", "nominal", "ttest")}
    for wk in range(n_weeks):
        t0 = t_start + wk * E8.BLOCK
        s[t0:t0 + E8.BLOCK] = tier(y[t0 - E8.BLOCK:t0].sum())
        mult = 1.4 if (scenario == "variant" and wk >= 10) else 1.0
        truth = epi.Twin(a=twin.a + np.log(mult), b=twin.b, k=twin.k, dow=twin.dow)
        y[t0:t0 + E8.BLOCK] = epi.simulate_block(truth, y, s, wd, t0, E8.BLOCK, 1, rng)[0]
        wv, wp, p, _ = mons["mc"].week(y, s, wd, t0, rng)
        wn, _, _, _ = mons["nominal"].week(y, s, wd, t0, rng)
        for k, hit in (("mc_blr", wv >= E8.THRESH), ("plugin", wp >= E8.THRESH),
                       ("nominal", wn >= E8.THRESH), ("ttest", p < E8.ALPHA)):
            if hit and first[k] is None:
                first[k] = wk + 1
    return first


def main():
    t_start = time.time()
    res = {"config": dict(calibration=[CAL_START, MON_START], monitoring=[MON_START, MON_END],
                          lag=epi.LAG, m=E8.M, alpha=E8.ALPHA)}
    res["real"] = real_data()
    print("first alarm:", res["real"]["first_alarm"], "alarms:",
          [a["date"] for a in res["real"]["alarms"]], flush=True)
    res["sensitivity_b"] = {}
    for b in (-0.5, -1.0, -2.0):
        r = real_data(b=b)
        gaps = [a["weeks_monitored"] for a in r["alarms"]]
        res["sensitivity_b"][str(b)] = {"first_alarm": r["first_alarm"], "n_alarms": len(gaps),
                                        "median_weeks_between": float(np.median(gaps)) if gaps else None}
    print("sensitivity:", res["sensitivity_b"], flush=True)
    res["synthetic"] = {}
    for sc, n in (("null", 200), ("variant", 100)):
        with ProcessPoolExecutor() as ex:
            runs = list(ex.map(partial(synthetic_one, scenario=sc), range(n), chunksize=4))
        res["synthetic"][sc] = E8.summarise(runs, sc)
        print(f"synthetic {sc}:", res["synthetic"][sc], flush=True)
    print(f"{time.time() - t_start:.0f}s")
    (ROOT / "results" / "exp8b.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
