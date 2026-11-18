# Reproducing the paper

This file explains how to reproduce every table and figure in
*Betting against the simulator: Monte Carlo e-processes under adaptive
actions* (`paper/main.pdf`) and its supplement
(`paper/supplement.pdf`) from the code in this repository.

## 1. Setup

The results were produced with Python 3.11 on Linux (4 cores). Library versions:

| package | version |
|---|---|
| numpy | 2.4.6 |
| scipy | 1.17.1 |
| scikit-learn | 1.9.1 |
| pandas | 3.0.6 |
| matplotlib | 3.11.2 |
| simglucose | 0.2.11 |

```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy scipy scikit-learn pandas matplotlib pytest
pip install --no-deps simglucose==0.2.11   # only its patient parameters are used
python -m pytest tests                     # unit tests, including exact validity checks
```

All commands below are run from the repository root. Each script writes its
numbers to `results/` (JSON, with a `.log` of what was printed) and, where
relevant, a figure to `figures/`. The JSON files used in the paper are already
in `results/`, so any table can be checked against them without re-running.

**Randomness.** Every replication draws from `numpy.random.default_rng` with a
seed fixed by the replication index, and no script uses an unseeded generator.
With the library versions above, re-running a script reproduces the stored
numbers exactly. Seeds depend on the replication index, not on the worker,
so parallel runs (`--workers`) give the same numbers as serial runs.

**Running times** are approximate, for a 4-core machine. "Minutes" means under
about 10 minutes.

## 2. Code layout

| path | content |
|---|---|
| `e4g/monitor.py` | the Monte Carlo e-process (eq. 3) and the hedged wealth (eq. 4) |
| `e4g/learners.py` | score learners: ridge logistic (`OnlineRatio`), Gaussian Bayesian learning rule on the logistic loss (`BLRRatio`) and on the simulation likelihood (`ChoiceBLR`), predictive-first updates (`MartingalePredictive`), Bayesian-bootstrap forest (`ForestMP`) |
| `experiments/` | one script per experiment (below) |
| `data/` | the processed public data, with sources in `data/README.md` |
| `tests/` | unit tests |
| `paper/` | LaTeX source, bibliography, and `build_pdf.py` (HTML-based PDF preview) |

## 3. Main paper

| item | content | command | output | time |
|---|---|---|---|---|
| Table 1 | objects of the Bayes factor and their replacements | none (text) | | |
| Figure 1 | workflow scheme | `python experiments/make_workflow_figure.py` (needs Chromium via Playwright) | `figures/workflow.png` | seconds |
| Table 2 | generative-model drift | `python experiments/exp1_generative_drift.py` | `results/exp1.json` | minutes |
| Table 3 | Thompson-sampling bandit against a twin | `python experiments/exp2_adaptive_bandit.py` | `results/exp2.json` | minutes |
| Section 5.3 (budget) | simulation budget M | `python experiments/exp3_simulation_budget.py` | `results/exp3.json` | minutes |
| Figure 2 | growth rate vs M (Theorem 2) | `python experiments/exp4_growth_rate.py` | `results/exp4.json`, `figures/exp4_growth_rate.png` | minutes |
| Section 5.3 (regret) | regret of the posterior-predictive e-process (Theorem 3) | `python experiments/exp12_regret.py` | `results/exp12.json` | ~10–20 min |
| Table 4 | update rules (selected rows; all rows in Table S4) | as for Table S4 below | | |
| Table 5 | comparison with concurrent methods (Forré & Brenner; SLICE) | `python experiments/exp15_concurrent_methods.py --task drift,bandit,drift-mis,bandit-mis` | `results/exp15.json` | ~1 h in all |
| Table 6 | auditing a language-model API with adaptive prompts | replies: `python experiments/exp16_llm_generate.py --models DIR` (needs `llama-cpp-python` and the GGUF files of Qwen2.5-1.5B-Instruct at q8_0, q4_k_m, q3_k_m and Qwen2.5-0.5B-Instruct at q4_k_m; the replies used are in `data/llm_audit/replies.json`); audit: `python experiments/exp16_llm_audit.py` | `results/exp16.json` | ~3 h for the replies on 4 CPU cores; ~30 min for the audit |
| Table 7 | MAMS trial, default learner | `python experiments/exp11_mams_trial.py` | `results/exp11.json` | ~30 min |
| Table 8 | notation | none (text) | | |

The experiment numbers in the file names follow the order in which the
experiments were written, not the order of the paper.

## 4. Supplement

| item | content | command | output | time |
|---|---|---|---|---|
| Section S1: table and figure | artificial pancreas, one patient against its twin (UVA/Padova parameters via `simglucose`) | `python experiments/exp7_artificial_pancreas.py` then `python experiments/exp7_figure.py` | `results/exp7.json`, `figures/exp7_artificial_pancreas.png` | ~10–30 min |
| Section S2: table | three-arm response-adaptive trial | `python experiments/exp10_adaptive_trial.py` | `results/exp10.json` | ~10–30 min |
| Section S3: table | comparison with the Besag–Clifford e-process | `python experiments/exp6_compare_besag_clifford.py` | `results/exp6.json` | ~10–30 min |
| Figure S2 | drift experiment, rejection curves | `python experiments/exp1_generative_drift.py` then `python experiments/make_figures.py` | `figures/exp1_generative_drift.png` | minutes |
| Table S4, BLR rows | Bayesian learning rule, logistic loss | `python experiments/exp5_bayesian_learning_rule.py` and `python experiments/exp5b_predictive_scales.py` | `results/exp5.json`, `results/exp5b.json` | minutes each |
| Table S4, simulation-likelihood rows | BLR on the simulation likelihood | `python experiments/exp13_choice_blr.py` | `results/exp13.json` | ~10–30 min |
| Table S4, predictive-update rows | martingale-posterior predictive updates | `python experiments/exp14_predictive_updates.py` | `results/exp14.json` | ~1–17 min per learner |
| Table S4, forest rows | Bayesian-bootstrap forest | `python experiments/exp14_predictive_updates.py --drift forest-mixture --bandit forest-mixture` | `results/exp14.json` | ~15–20 min |
| Table S5 | regret of the posterior-predictive e-process (Theorem 3) | `python experiments/exp12_regret.py` | `results/exp12.json` | ~10–20 min |
| Table S6 | MAMS trial, each learner | `python experiments/exp11_mams_trial.py --learner L --scenarios null` and `... --scenarios snacking`, for L in `choice`, `mp-logistic-plugin`, `mp-logistic-resample`, `mp-choice-plugin`, `mp-choice-resample`, `forest-mixture-early` | `results/exp11_<L>_<scenario>.json` (`results/exp11_choice.json` for `choice`) | ~15–40 min per run |
| Table S7 | comparison with concurrent methods, all variants | as for Table 5 above | `results/exp15.json` | |

The COVID-19 analyses of Italy's regions and national deaths are not in the
current version of the paper; their text is kept in `paper/removed_covid.tex`
and their scripts (`exp8_italy_covid.py`, `exp8b_italy_deaths_2020_2021.py`,
`exp9_italy_regions.py`, `exp9_figure.py`, `alarm_directions.py`) and results
remain in the repository.

## 5. Data

The paper uses no external data. The replies of the language models
(Section 5.6) are generated by `experiments/exp16_llm_generate.py` and stored in
`data/llm_audit/replies.json`. The COVID-19 analyses kept outside the paper use
only public data, under CC-BY-4.0:
- Dipartimento della Protezione Civile, national and regional COVID-19 series;
- the Oxford COVID-19 Government Response Tracker, national stringency index;
- a public compilation of the regional restriction tiers.

`data/README.md` gives the source files, column definitions and download dates.
The processed files are `data/italy_covid_daily.csv` and
`data/italy_regions_daily.csv`. The scripts read them directly, with no network
access.

## 6. Building the paper

`paper/main.tex` and `paper/supplement.tex` are standard LaTeX (natbib,
`references.bib`):

```bash
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

`paper/build_pdf.py` produces a preview PDF without a TeX installation, through
HTML and headless Chromium. It is a convenience for editing and does not
replace a LaTeX build:

```bash
python paper/build_pdf.py main
python paper/build_pdf.py supplement
```
