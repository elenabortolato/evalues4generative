"""Direction of each real-data alarm in Experiments 8b and 9.

Replays the real-data monitors of exp8b (national deaths) and exp9 (regional
zones) with the same seeds, so the alarms are identical, and records for each
alarm whether the data were above or below the twin: the standardised
deviation z of the weekly total from the twin's simulations, averaged over the
last three weeks of the segment (national) or taken in the alarm week
(regional).

Output: results/alarm_directions.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

# exp9 and epi_regional bind the 7-day lag at import; exp8b then sets epi.LAG = 21
# for the national deaths twin, so the order of these imports matters
import exp9_italy_regions as G  # noqa: E402,I001
from e4g import epi_regional as ER  # noqa: E402
import exp8b_italy_deaths_2020_2021 as N  # noqa: E402


def national(b=-1.0, seed=0):
    """Mirrors N.real_data, including its RNG calls."""
    E8, epi = N.E8, N.epi
    d, y, s, wd = N.load()
    rng = np.random.default_rng(seed)
    t_cal, t0 = N.index(d, N.CAL_START), N.index(d, N.MON_START)
    mon = E8.Monitor(epi.fit(y, s, wd, t_cal, t0, b_fixed=b), y, s, wd, t0, rng)
    out = []
    while t0 + E8.BLOCK <= len(y):
        wv, _, _, _ = mon.week(y, s, wd, t0, rng)
        epi.simulate_block(mon.twin, y, s, wd, t0, E8.BLOCK, 400, rng)
        t0 += E8.BLOCK
        if wv >= E8.THRESH and t0 + E8.BLOCK <= len(y):
            z3 = float(np.mean(mon.z[-3:]))
            out.append(dict(date=str(d.date[t0 - 1].date()), weeks=len(mon.z), z_last3=z3,
                            direction="above" if z3 > 0 else "below"))
            twin = epi.fit(y, s, wd, t0 - E8.CALIB, t0, b_fixed=b)
            mon = E8.Monitor(twin, y, s, wd, t0, rng)
    return out


def regional(seed=0):
    """Mirrors G.real_data, including its RNG calls."""
    regions, dates, y, c, wd, start, end, twin = G.setup_cached()
    R, T = y.shape
    rng = np.random.default_rng(seed)
    for r in range(R):
        for wk in range(12):
            for col in (None, G.YELLOW):
                ER.simulate_block(twin, r, y[r], c[r], wd, end + wk * G.BLOCK, G.BLOCK, 400,
                                  rng, colour=col)
    mons = [G.RegionMonitor(twin, r, y, c, wd, end, rng) for r in range(R)]
    frozen = [G.RegionMonitor(twin, r, y, c, wd, end, rng) for r in range(R)]
    twins, stopped, out = [twin] * R, [None] * R, []
    names = {v: k for k, v in ER.CODE.items()}
    for wk in range((T - end) // G.BLOCK):
        t0 = end + wk * G.BLOCK
        for r in range(R):
            wv, _, _ = mons[r].week(y, c, wd, t0, rng)
            if wv >= G.THRESH and t0 + 2 * G.BLOCK <= T:
                z = float(mons[r].z[-1])
                week = y[r, t0:t0 + G.BLOCK].sum()
                out.append(dict(region=regions[r], date=str(dates[t0 + G.BLOCK - 1].date()),
                                weeks=len(mons[r].z), z_last=z,
                                direction="above" if z > 0 else "below",
                                zone=names[int(c[r, t0 + G.BLOCK - 1])],
                                cases_per_100k=float(week / (ER.POP[regions[r]] * 1e6) * 1e5)))
                twins[r] = ER.refit_region(twins[r], y, c, wd, r, t0 + G.BLOCK - G.RECAL,
                                           t0 + G.BLOCK)
                mons[r] = G.RegionMonitor(twins[r], r, y, c, wd, t0 + G.BLOCK, rng)
            if stopped[r] is None:
                w, _, _ = frozen[r].week(y, c, wd, t0, rng)
                if w >= G.THRESH:
                    stopped[r] = w
    return out


def main():
    res = {"national": national(), "regional": regional()}
    reg = res["regional"]
    phases = [("2021-01-01", "2021-04-11"), ("2021-04-12", "2021-07-10"),
              ("2021-07-11", "2021-08-15"), ("2021-08-16", "2021-10-31"),
              ("2021-11-01", "2021-12-31")]
    res["regional_phases"] = []
    for a, b in phases:
        sel = [x for x in reg if a <= x["date"] <= b]
        res["regional_phases"].append(dict(
            start=a, end=b, n=len(sel), above=sum(x["direction"] == "above" for x in sel),
            median_cases_per_100k=float(np.median([x["cases_per_100k"] for x in sel])),
            white=sum(x["zone"] == "white" for x in sel)))
    for x in res["national"]:
        print(x)
    for p in res["regional_phases"]:
        print(p)
    print("regional alarms:", len(reg), "in white zones:", sum(x["zone"] == "white" for x in reg))
    (ROOT / "results" / "alarm_directions.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
