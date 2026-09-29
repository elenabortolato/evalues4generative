"""Figure for the application: one monitored patient under insulin resistance."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import exp7_artificial_pancreas as E  # noqa: E402
from e4g import BLRRatio, LambdaMixture, mc_factor, t1d  # noqa: E402
from make_figures import GRID, INK, INK2, REF, SLOT, legend  # noqa: E402


def record(seed=3, name="resistance"):
    rng = np.random.default_rng(seed)
    plan = t1d.meal_plan(rng, E.HOURS // 24)
    real = t1d.Batch.initial(E.P)
    cgm = float(t1d.bg(real, E.P)[0])
    blr = BLRRatio(feature_map=E.feature_map).fit_scaler(E.pilot_rows(seed))
    blr_nom = BLRRatio(feature_map=E.feature_map).fit_scaler(E.pilot_rows(seed))
    w, wn = LambdaMixture(), LambdaMixture()
    rec = {k: [] for k in ("x", "lo", "hi", "rate", "logw", "logw_nom", "meal")}
    for e in range(E.EPOCHS):
        meals = plan[e * E.L:(e + 1) * E.L]
        rate = E.committed_rate(cgm)
        both = np.array([rate, E.CTL.basal])
        gg = E.block(real.repeat(2), rng, both, meals, det=True)
        yy = E.block(real.repeat(2 * E.M), rng, np.repeat(both, E.M), meals)
        y, y_nom = yy[:E.M], yy[E.M:]
        x = E.block(real, rng, rate, meals, **E.scenario(name, e))[0][None, :]
        rx, ry = E.rows(x, gg[0], rate), E.rows(y, gg[0], rate)
        rxn, ryn = E.rows(x, gg[1], E.CTL.basal), E.rows(y_nom, gg[1], E.CTL.basal)
        s = blr.score(np.vstack([rx, ry]))
        sn = blr_nom.score(np.vstack([rxn, ryn]))
        rec["logw"].append(np.log(w.update(mc_factor(s[0], s[1:]))))
        rec["logw_nom"].append(np.log(wn.update(mc_factor(sn[0], sn[1:]))))
        blr.update(rx, ry)
        blr_nom.update(rxn, ryn)
        rec["x"].append(x[0])
        rec["lo"].append(np.percentile(y, 5, axis=0))
        rec["hi"].append(np.percentile(y, 95, axis=0))
        rec["rate"].append(rate * 60)
        rec["meal"].append(meals.sum())
        cgm = float(x[0, -1])
    return {k: np.array(v) for k, v in rec.items()}


def main():
    r = record()
    minutes = np.arange(r["x"].size) / 60.0
    epoch_h = (np.arange(E.EPOCHS) + 1) * E.L * 5 / 60.0
    fig, ax = plt.subplots(3, 1, figsize=(11, 7.2), sharex=True,
                           gridspec_kw={"height_ratios": [2.2, 1, 1.4]})
    ax[0].fill_between(minutes, r["lo"].ravel(), r["hi"].ravel(), color=SLOT[0], alpha=0.25,
                       lw=0, label="twin, 5–95% band")
    ax[0].plot(minutes, r["x"].ravel(), color=INK, lw=0.8)
    ax[0].axvline(E.CHANGE_H, color=INK2, lw=1, ls=":")
    ax[0].text(E.CHANGE_H + 0.3, ax[0].get_ylim()[1] * 0.95, "insulin sensitivity −30%",
               color=INK2, fontsize=8, va="top")
    ax[0].set_ylabel("CGM (mg/dL)")
    ax[0].set_title("Patient (black) against the twin's predictive band, epoch by epoch",
                    color=INK)
    ax[1].step(epoch_h - E.L * 5 / 60, r["rate"], where="post", color=SLOT[1], lw=1.5)
    ax[1].set_ylabel("basal (U/h)")
    ax[1].set_title("Committed basal rate: suspended below 80, corrected above 180 mg/dL",
                    color=INK)
    ax[2].plot(epoch_h, r["logw"], color=SLOT[0], lw=2)
    ax[2].plot(epoch_h, r["logw_nom"], color=SLOT[3], lw=2)
    ax[2].axhline(np.log(1 / E.ALPHA), color=INK2, lw=1, ls=":")
    ax[2].axvline(E.CHANGE_H, color=INK2, lw=1, ls=":")
    ax[2].set_ylabel("log wealth")
    ax[2].set_xlabel("hours")
    ax[2].set_title("Monitor wealth; dotted: rejection threshold log 20", color=INK)
    styles = {"a": ("twin conditioned on the committed dose", SLOT[0], "-"),
              "b": ("twin under the open-loop plan", SLOT[3], "-")}
    fig.suptitle("Exp. 7: an artificial pancreas monitored against its digital twin",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    legend(fig, styles, 0.945)
    fig.savefig(ROOT / "figures" / "exp7_artificial_pancreas.png", dpi=150)
    print("crossing (h):", epoch_h[np.argmax(r["logw"] >= np.log(20))]
          if (r["logw"] >= np.log(20)).any() else None)


if __name__ == "__main__":
    main()
