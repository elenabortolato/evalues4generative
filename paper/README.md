# Paper draft

`main.tex` + `references.bib`: *Betting against the simulator: anytime-valid
criticism of generative models and digital twins under adaptive control.*
Standard LaTeX (natbib, amsthm). Figures are read from `../figures/`.

```bash
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

`main.pdf` (17 pages) was **not** produced by pdflatex: no TeX distribution
could be installed in the drafting container. `build_pdf.py` renders it
instead, LaTeX → HTML → Chromium print-to-PDF, with KaTeX setting the
mathematics:

```bash
pip install playwright && python paper/build_pdf.py   # -> paper/main.pdf
```

The converter is adapted from `PosteriorOperator/paper/build_pdf.py`, with
additions for this paper:
- author-year citations from `references.bib`;
- theorem environments and proofs;
- `\[…\]` display maths;
- labelled items;
- subsection references;
- a lettered appendix.

It handles only the LaTeX this paper uses and reports anything it cannot
convert. The current build reports no unconverted commands, no missing
citations and no unresolved references. Prefer pdflatex once TeX is available.

## Where the numbers come from

| Paper | Source |
|---|---|
| Table 1, Figure 1 | `results/exp1.json` (`experiments/exp1_generative_drift.py`) |
| Table 2, Figure 2 | `results/exp2.json` (`experiments/exp2_adaptive_bandit.py`) |
| Table 3 | `results/exp3.json` (`experiments/exp3_simulation_budget.py`) |
| Figure 3 and Section 6.3 bullets | `results/exp4.json` (`experiments/exp4_growth_rate.py`) |
| Table 4, Figure 4 | `results/exp5.json`, `results/exp5b.json` (`experiments/exp5_bayesian_learning_rule.py`, `exp5b_predictive_scales.py`) |
| Table 5 | `results/exp6.json` (`experiments/exp6_compare_besag_clifford.py`) |

## Before submission

- Read Dombowsky, Engelhardt & Ramdas (2026) and Shaer, Maman & Romano (2023)
  in full. Section 7 states the differences from them provisionally.
- Check the 7 bibliography entries marked `VERIFY`.
- Proof of Theorem 2(iv): the O(ε) error terms are argued informally, under
  bounded g and h.
