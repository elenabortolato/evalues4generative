"""Figure for the regional application (reads results/exp9.json)."""

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
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from make_figures import INK, INK2, SLOT  # noqa: E402

# the four official zone colours, softened so the alarm marks stay legible
ZONE = {"white": "#f2f2ef", "yellow": "#f3d36b", "orange": "#ee9a4d", "red": "#d9534f"}


def main():
    res = json.loads((ROOT / "results" / "exp9.json").read_text())["real"]
    d = pd.read_csv(ROOT / "data" / "italy_regions_daily.csv", parse_dates=["date"])
    start, stop = pd.Timestamp(res["monitoring"][0]), pd.Timestamp(res["monitoring"][1])
    d = d[(d.date >= start) & (d.date <= stop)]
    regions = res["regions"]
    code = {"white": 0, "yellow": 1, "orange": 2, "red": 3}
    grid = d.pivot(index="region", columns="date", values="colour").loc[regions]
    grid = grid.apply(lambda s: s.map(code)).values.astype(float)
    dates = pd.DatetimeIndex(sorted(d.date.unique()))

    fig, ax = plt.subplots(2, 1, figsize=(11.5, 7.6), sharex=True,
                           gridspec_kw={"height_ratios": [3.2, 1]})
    cmap = ListedColormap([ZONE[k] for k in ("white", "yellow", "orange", "red")])
    x0, x1 = mdates.date2num(dates[0]), mdates.date2num(dates[-1])
    ax[0].imshow(grid, aspect="auto", cmap=cmap, vmin=-0.5, vmax=3.5, interpolation="nearest",
                 extent=[x0, x1, len(regions) - 0.5, -0.5])
    for i, reg in enumerate(regions):
        for a in res["alarms"][reg]:
            ax[0].plot(mdates.date2num(pd.Timestamp(a)), i, marker="v", ms=6, color=INK,
                       markeredgecolor="white", markeredgewidth=0.6)
    ax[0].set_yticks(range(len(regions)), regions, fontsize=8)
    ax[0].xaxis_date()
    ax[0].set_title("Colour zone assigned to each region (cells) and monitor alarms (triangles)",
                    color=INK, fontsize=10)
    handles = [Patch(facecolor=ZONE[k], edgecolor=INK2, lw=0.5, label=k) for k in ZONE]
    handles.append(plt.Line2D([], [], marker="v", ls="", color=INK, label="alarm"))
    ax[0].legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False,
                 fontsize=8)
    weeks = dates[6::7][:len(res["national_logw"])]
    ax[1].plot(weeks, res["national_logw"], color=SLOT[0], lw=2)
    ax[1].axhline(np.log(20), color=INK2, lw=1, ls=":")
    ax[1].set_ylabel("log wealth")
    ax[1].set_title("National monitor: average of the 21 regional wealths (frozen twin); "
                    "dotted: log 20", color=INK, fontsize=10)
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.suptitle("Exp. 9: Italy's regional colour zones, March 2021 – May 2022",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(ROOT / "figures" / "exp9_italy_regions.png", dpi=150)


if __name__ == "__main__":
    main()
