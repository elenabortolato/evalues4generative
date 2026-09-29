"""Figure for the real-data application (reads results/exp8.json)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from make_figures import INK, INK2, SLOT  # noqa: E402


def main():
    r = json.loads((ROOT / "results" / "exp8.json").read_text())["real"]
    d = pd.read_csv(ROOT / "data" / "italy_covid_daily.csv", parse_dates=["date"])
    log = r["log"]
    dates = pd.to_datetime(log["date"])
    obs = d.set_index("date").loc[dates, "icu_admissions"].values
    fig, ax = plt.subplots(2, 1, figsize=(11, 5.8), sharex=True,
                           gridspec_kw={"height_ratios": [2.2, 1]})
    ax[0].fill_between(dates, log["lo"], log["hi"], color=SLOT[0], alpha=0.3, lw=0,
                       step="mid")
    ax[0].plot(dates, obs, color=INK, lw=0.8)
    for a in r["alarms"]:
        ax[0].axvline(pd.Timestamp(a["date"]), color=SLOT[1], lw=1.2)
    ax[0].set_ylabel("daily ICU admissions")
    ax[0].set_title("Observed admissions (black), twin 90% band for each coming week (blue), "
                    "alarms and recalibrations (orange)", color=INK, fontsize=10)
    # wealth per segment
    week_end = dates[6::7]
    seg = np.array(log["segment"])[6::7]
    lw = np.array(log["logw"])
    for s in np.unique(seg):
        m = seg == s
        ax[1].plot(week_end[m], lw[m], color=SLOT[0], lw=2)
    ax[1].axhline(np.log(20), color=INK2, lw=1, ls=":")
    ax[1].set_ylabel("log wealth")
    ax[1].set_title("Monitor wealth, restarted at each recalibration; dotted: log 20",
                    color=INK, fontsize=10)
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.suptitle("Exp. 8: Italy, February 2021 – December 2022: when does a calibrated "
                 "transmission twin stop describing the epidemic?",
                 x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(ROOT / "figures" / "exp8_italy_covid.png", dpi=150)


if __name__ == "__main__":
    main()
