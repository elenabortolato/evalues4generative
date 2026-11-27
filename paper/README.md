# Paper draft

`main.tex` + `references.bib`: *Betting against the simulator: Monte Carlo
e-processes under adaptive actions.*
Standard LaTeX (natbib, amsthm). Figures are read from `../figures/`.

```bash
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Journal versions

Both are generated from `main.tex`, so edits belong there; each folder is
self-contained and can be uploaded to Overleaf as it is.

- `stco/`: *Statistics and Computing* (Springer Nature template `sn-jnl`).
  `cd paper/stco && python make_stco.py`.
- `jcgs/`: *Journal of Computational and Graphical Statistics* (ASA template:
  12pt, 1-inch margins, line spacing 1.45, Chicago author-year references).
  `cd paper/jcgs && python make_jcgs.py`. Set `\blind` to 1 in
  `manuscript.tex` for the anonymised version.

Then `pdflatex manuscript && bibtex manuscript && pdflatex manuscript && pdflatex manuscript`.

## arXiv

`cd paper && python make_arxiv.py` writes `arxiv.zip`: `main.tex` without
comment lines, `main.bbl`, the figures, and the compiled supplement as an
ancillary file (`anc/supplement.pdf`). The script also checks that the zip
builds on its own without BibTeX, as arXiv builds it.

`main.pdf` (25 pages) was **not** produced by pdflatex: no TeX distribution
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
| Section 7, Figure 5 | `results/exp7.json` (`experiments/exp7_artificial_pancreas.py`, `exp7_figure.py`) |
| Section 8.1, regional colour zones 2020–21 | `results/exp9.json` (`experiments/exp9_italy_regions.py`, `exp9_figure.py`) |
| Section 8.2, national deaths 2020–21 | `results/exp8b.json` (`experiments/exp8b_italy_deaths_2020_2021.py`, `exp8_figure.py`) |
| National ICU admissions 2021–22 (mentioned in 8.2) | `results/exp8.json` (`experiments/exp8_italy_covid.py`) |

## Before submission

- Read Dombowsky, Engelhardt & Ramdas (2026) and Shaer, Maman & Romano (2023)
  in full. Section 7 states the differences from them provisionally.
- Check the 7 bibliography entries marked `VERIFY`.
- Proof of Theorem 2(iv): the O(ε) error terms are argued informally, under
  bounded g and h.
