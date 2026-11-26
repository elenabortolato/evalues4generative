"""LaTeX of Tables 6 and 7 (language-model audit) from the results of exp16.

Reads results/exp16.json (queries to the alarm and false alarms),
results/exp16_cost.json (computing time to the alarm, our design) and
results/exp16_timing.json (milliseconds per query), and prints the two
tables in the journal form used by paper/stco (\\sbox{\\widetab}{...}).
Medians and percentages are rounded half up.
"""

from __future__ import annotations

import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = json.loads((ROOT / "results" / "exp16.json").read_text())
COST = json.loads((ROOT / "results" / "exp16_cost.json").read_text())
TIME = json.loads((ROOT / "results" / "exp16_timing.json").read_text())

PREPARED = {"slice": 6000, "fb-pre": 6000, "fb-refit": 800, "ours-uniform": 800,
            "ours-simlik": 800, "slice-bandit": 6000, "fb-bandit": 6000,
            "fb-refit-bandit": 800, "combo-bandit": 6800, "ours-bandit": 800,
            "ours-simlik-bandit": 800}
LABEL = {"slice": "SLICE", "fb-pre": r"F\&B, fitted in advance", "fb-refit": r"F\&B, refitted",
         "ours-uniform": "ours, logistic BLR", "ours-simlik": "ours, simulation likelihood",
         "combo-bandit": r"average$^\ddagger$"}
for _m in ["slice", "fb-pre", "fb-refit"]:
    LABEL[_m + "-bandit"] = LABEL[_m]
LABEL["fb-bandit"] = LABEL["fb-pre"]
LABEL["ours-bandit"], LABEL["ours-simlik-bandit"] = LABEL["ours-uniform"], LABEL["ours-simlik"]
GROUPS = [("prompts drawn uniformly (as published)",
           ["slice", "fb-pre", "fb-refit", "ours-uniform", "ours-simlik"]),
          ("prompts chosen by UCB (our design)",
           ["slice-bandit", "fb-bandit", "fb-refit-bandit", "combo-bandit", "ours-bandit",
            "ours-simlik-bandit"])]
ROUTING = {"all": ["all-q4", "all-q3", "all-05b", "sharp"],
           "two": ["some-q4", None, "some-05b", None]}
COST_COLS = ["all-q4", "some-q4", "all-q3", "all-05b", "some-05b", "sharp"]


def hu(x, nd=0):
    q = Decimal(1).scaleb(-nd)
    return str(Decimal(str(x)).quantize(q, rounding=ROUND_HALF_UP))


def rejects(r):
    return r.get("median_stop") is not None and r["reject"] >= 0.5


def table6():
    best = {sc: min(int(hu(RES[sc][m]["median_stop"])) for g in GROUPS for m in g[1]
                    if rejects(RES[sc][m]))
            for sc in ROUTING["all"] + [s for s in ROUTING["two"] if s]}

    def cell(m, sc):
        if sc is None:
            return ""
        r = RES[sc][m]
        if not rejects(r):
            return "never"
        q = int(hu(r["median_stop"]))
        s = rf"\textbf{{{q}}}" if q == best[sc] else str(q)
        return s + (f" ({hu(r['reject'], 2)})" if r["reject"] < 0.995 else "")

    out = [r"\sbox{\widetab}{\begin{tabular}{llrrrrrl}", r"\toprule",
           r" & & \multicolumn{4}{c}{substitute} & false & growth \\", r"\cmidrule(lr){3-6}",
           r"score & routing & 4-bit & 3-bit & 0.5B & sharpened $P$ & alarms & guarantee \\",
           r"\midrule"]
    for gi, (title, methods) in enumerate(GROUPS):
        if gi:
            out.append(r"\midrule")
        out.append(rf"\multicolumn{{8}}{{l}}{{\emph{{{title}}}}} \\")
        for j, m in enumerate(methods):
            if j:
                out.append(r"\addlinespace[3pt]")
            fa = hu(100 * RES["null"][m]["reject"], 1) + r"\%"
            g = r"Thm~\ref{thm:simlik}$^*$" if "simlik" in m else "--"
            out.append(f"{LABEL[m]} & all & " + " & ".join(cell(m, s) for s in ROUTING["all"])
                       + f" & {fa} & {g} \\\\")
            out.append(" & two & " + " & ".join(cell(m, s) for s in ROUTING["two"]) + r" & & \\")
    out += [r"\bottomrule", r"\end{tabular}}%"]
    return "\n".join(out)


def table7():
    methods = GROUPS[1][1]
    reps = {sc: {m: PREPARED[m] + 8 * COST[sc][m]["median_stop"]
                 for m in methods if rejects(COST[sc][m])} for sc in COST_COLS}
    secs = {sc: {m: COST[sc][m]["median_compute_s"] for m in methods if rejects(COST[sc][m])}
            for sc in COST_COLS}

    def fmt(val, col, nd):
        s = hu(val, nd)
        return rf"\textbf{{{s}}}" if val == min(col.values()) else s

    out = [r"\sbox{\widetab}{\begin{tabular}{llrrrrrrrrr}", r"\toprule",
           r" & & \multicolumn{2}{c}{4-bit} & 3-bit & \multicolumn{2}{c}{0.5B} & sharpened"
           r" & prepared & \multicolumn{2}{c}{ms per query} \\",
           r"\cmidrule(lr){3-4}\cmidrule(lr){5-5}\cmidrule(lr){6-7}\cmidrule(lr){8-8}"
           r"\cmidrule(lr){10-11}",
           r"score & to the alarm & all & two & all & all & two & $P$, all & ($10^3$)"
           r" & 400 & 1600 \\", r"\midrule"]
    for j, m in enumerate(methods):
        if j:
            out.append(r"\addlinespace[3pt]")
        r1 = [fmt(reps[sc][m] / 1000, {k: v / 1000 for k, v in reps[sc].items()}, 1)
              if m in reps[sc] else "never" for sc in COST_COLS]
        r2 = [fmt(secs[sc][m], secs[sc], 2) if m in secs[sc] else "never" for sc in COST_COLS]
        t400 = hu(TIME["ms_per_query_400"][m], 1)
        t1600 = hu(TIME["ms_per_query_1600"][m], 1)
        out.append(f"{LABEL[m]} & replies ($10^3$) & " + " & ".join(r1)
                   + f" & {hu(PREPARED[m] / 1000, 1)} & {t400} & {t1600} \\\\")
        out.append(" & computing (s) & " + " & ".join(r2) + r" & & & \\")
    out += [r"\bottomrule", r"\end{tabular}}%"]
    return "\n".join(out)


if __name__ == "__main__":
    print("% Table 6\n" + table6() + "\n\n% Table 7\n" + table7())
