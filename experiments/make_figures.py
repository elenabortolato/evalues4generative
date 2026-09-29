"""Figures from results/exp{1,2,3}.json -> figures/*.png."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
REF = "#52514e"  # oracle reference line (neutral, dashed)
# categorical slots in fixed order (reference palette, light mode)
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def panel(ax, curves, styles, alpha, title, null):
    t = np.arange(1, len(next(iter(curves.values()))) + 1)
    ends = []
    for key, (label, color, ls) in styles.items():
        y = np.asarray(curves[key])
        ax.plot(t, y, color=color, lw=2, ls=ls, solid_capstyle="round")
        ends.append([y[-1], label, color])
    if null:
        ax.axhline(alpha, color=INK2, lw=1, ls=":")
        ax.text(t[0], alpha, f" α = {alpha}", color=INK2, va="bottom", fontsize=8)
    # direct labels at line ends, nudged apart
    ends.sort(key=lambda e: e[0])
    gap = 0.055
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + gap)
    for y, label, color in ends:
        ax.annotate(label, (t[-1], y), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK)
        ax.plot([t[-1]], [y], marker="o", ms=4, color=color, clip_on=False)
    ax.set_title(title, color=INK)
    ax.set_xlim(1, t[-1])
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("time step t")


def fig_exp1():
    r = json.loads((RES / "exp1.json").read_text())
    a = r["config"]["alpha"]
    styles = {"mc": ("Monte Carlo e-value", SLOT[0], "-"),
              "plugin": ("plug-in ratio", SLOT[1], "-"),
              "ks": ("KS, peeking", SLOT[2], "-"),
              "oracle": ("oracle LR", REF, "--")}
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7), sharey=True)
    for i, learner in enumerate(["regularised", "flexible"]):
        for j, (lab, name) in enumerate([("null", "H0 true: false alarms"),
                                         ("alt", "H1 true: detections")]):
            panel(axes[i, j], r[learner][lab]["curves"], styles, a,
                  f"{name}  ·  {learner} learner", null=(lab == "null"))
        axes[i, 0].set_ylabel("P(rejected by time t)")
    fig.suptitle("Exp. 1: monitoring a black-box generative model for drift",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 0.93, 0.97))
    fig.savefig(FIG / "exp1_generative_drift.png", dpi=150)


def fig_exp2():
    r = json.loads((RES / "exp2.json").read_text())
    a = r["config"]["alpha"]
    styles = {"mc": ("MC e-value (arm-conditional twin)", SLOT[0], "-"),
              "plugin": ("plug-in ratio", SLOT[1], "-"),
              "ttest": ("t-test, peeking", SLOT[2], "-"),
              "mc_uncond": ("MC e-value, twin NOT conditioned on arm", SLOT[3], "-"),
              "oracle": ("oracle LR", REF, "--")}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4), sharey=True)
    panel(axes[0], r["null"]["curves"], styles, a, "H0 true: false alarms", null=True)
    panel(axes[1], r["alt"]["curves"], styles, a, "H1 true: detections", null=False)
    axes[0].set_ylabel("P(rejected by time t)")
    fig.suptitle("Exp. 2: Thompson-sampling bandit checked against a digital twin",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 0.86, 0.95))
    fig.savefig(FIG / "exp2_adaptive_bandit.png", dpi=150)


def fig_exp3():
    r = json.loads((RES / "exp3.json").read_text())
    a = r["config"]["alpha"]
    rows = r["rows"]
    ms = [row["m"] for row in rows]
    x = np.arange(len(ms))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    typ = [row["null_reject"] for row in rows]
    powr = [row["alt_reject"] for row in rows]
    stop = [row["alt_median_stop"] or np.nan for row in rows]
    for ax, y, title, ylab in [
        (axes[0], typ, "Type-I error (H0 true)", "P(reject by T)"),
        (axes[1], powr, "Power (H1 true)", "P(reject by T)"),
        (axes[2], stop, "Median stopping time (H1)", "steps"),
    ]:
        ax.bar(x, y, width=0.6, color=SLOT[0])
        for xi, yi in zip(x, y):
            if np.isfinite(yi):
                ax.annotate(f"{yi:.2f}" if yi < 2 else f"{yi:.0f}", (xi, yi),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", fontsize=8, color=INK)
        ax.set_xticks(x, [str(m) for m in ms])
        ax.set_xlabel("simulator draws per step, M")
        ax.set_title(title, color=INK)
        ax.set_ylabel(ylab)
        ax.grid(axis="x", visible=False)
    axes[0].axhline(a, color=INK2, lw=1, ls=":")
    axes[0].set_ylim(0, max(0.12, max(typ) * 1.3))
    axes[1].set_ylim(0, 1.08)
    fig.suptitle("Exp. 3: the simulation budget buys power, never validity",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG / "exp3_simulation_budget.png", dpi=150)


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    for f, name in [(fig_exp1, "exp1"), (fig_exp2, "exp2"), (fig_exp3, "exp3")]:
        if (RES / f"{name}.json").exists():
            f()
            print("wrote", name)
