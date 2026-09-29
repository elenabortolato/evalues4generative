# Related work: where this method sits

This review was done on 2026-09-29 from search-engine summaries of abstracts;
arXiv full texts were not reachable. The two closest papers are marked ★.
Read them in full before claiming novelty.

Each paper is checked against five questions:
- **Stream:** do new real observations arrive over time?
- **Sim-only:** is the null available only as a simulator?
- **Actions:** are actions chosen adaptively from the data?
- **Learned:** is the test statistic learned?
- **Anytime:** is the test anytime-valid?

| Work | Stream | Sim-only | Actions | Learned | Anytime | Relation |
|---|---|---|---|---|---|---|
| Besag & Clifford (1989), Monte Carlo tests | no | yes | no | no | no | single-step validity argument |
| Fischer & Ramdas (JRSSB 2025), sequential MC testing by betting | no (fixed data) | yes | no | no | in the simulations | sequential in the Monte Carlo draws, not in the data |
| ★ Dombowsky, Engelhardt & Ramdas (arXiv 2603.15845v2, 2026), Besag–Clifford e-values | **yes (§4, i.i.d. null)** | via MCMC or exact sampling | no | likelihood ratio, or plug-in MLE (§3.2, §6.3) | **yes** | same one-step e-value (their Prop. 1) and sequential product (their Prop. 6); log-optimality (Thm 1, 2); non-asymptotic bound (Thm 3) |
| Barber & Ramdas (arXiv 2607.23010, 2026), MC testing without joint exchangeability | no | yes | no | – | no | extension route for approximate or MCMC simulators |
| ★ Shaer, Maman & Romano (AISTATS 2023), model-X sequential CI testing | yes | known conditional law | no | yes | yes | same mechanism with M = 1, for conditional independence |
| Vovk et al. (2003); Vovk (Stat. Sci. 2021), conformal test martingales | yes | no reference model | no | yes | yes | tests exchangeability of the stream itself |
| Prinster et al. (ICML 2025), WATCH | yes | no | no | yes | yes | weighted conformal martingales for monitoring AI deployments |
| Podkopaev & Ramdas (NeurIPS 2023); Shekhar & Ramdas (IEEE TIT 2023) | yes | no (two real streams) | no | yes (classifier) | yes | natural baseline: feed it a simulated stream |
| Martinez-Taboada & Ramdas (AISTATS 2025), sequential KSD | yes | needs ∇log p | no | kernel | yes | goodness of fit for unnormalised densities |
| Nair & Janson (arXiv 2301.05365, 2023), randomisation tests for adaptive data | no | no | yes | – | no | validity from the known assignment algorithm |
| Zampieri (arXiv 2512.04366, 2025), randomisation e-process for trials | yes | no | yes | – | yes | validity from randomisation |
| Waudby-Smith et al. (2024), off-policy confidence sequences | yes | no | yes | – | yes | targets policy values |
| Zhou et al. (arXiv 2605.07002, 2026), adaptive auditing of AI | yes | no | yes (test cases) | – | yes | adaptive choice of test cases |
| Ghasemloo, Eckman & Li (arXiv 2607.17088, 2026), digital-twin validation | no | yes | conditions on the observed trace | no | no | same conditioning idea, fixed-sample tests |
| **This work** | yes | yes | yes | yes | yes | combination, plus the growth theory (Theorem 1) |

## Claims, pending the full-text check

- **C1.** Streaming criticism of simulator-only nulls with fresh draws for each
  context, exactly valid for any learner.
- **C2.** Validity under adaptive control if and only if the simulator is run
  on the action taken. The "only if" part is a counterexample: a
  Thompson-sampling bandit where the unconditioned version rejects a true null
  in 100% of runs.
- **C3.** Growth-rate theory with learned scores:
  - the decomposition KL(Q‖P) − KL(Q‖Q̂);
  - a finite-M bound;
  - local efficiency M/(M+1);
  - tempering of overconfident scores by a finite budget.
- **C4.** Evidence that the natural shortcuts fail: using a learned ratio
  directly as the bet, or re-running fixed-sample tests.

**Not new:** exchangeability validity (Besag–Clifford), testing by betting,
classifier-based ratio estimation, and per-step conditional null sampling
(Shaer et al.).

## Questions for the full texts

1. Dombowsky et al.: is "sequential testing" a data stream with fresh null
   samples at each step? Do they condition on actions or covariates? Does their
   log-optimality constant equal M/(M+1) for i.i.d. simulations?
2. Shaer et al.: do they analyse the number of dummy draws per step and the
   power as a function of it?
3. Fischer & Ramdas, Barber & Ramdas: is there any data-stream version?

## Update after reading Dombowsky, Engelhardt & Ramdas (v2, 27 April 2026) in full

The full text answers the first question above, and it narrows our claims.

**Already in their paper:**
- The one-step e-value (their Prop. 1). They credit it as the "soft-rank"
  e-variable of Wang & Ramdas (2022, JRSSB).
- The **sequential** Besag–Clifford e-process (their §4, Prop. 6). There are
  fresh null draws at every step and a predictable betting fraction, but the
  null is i.i.d., simulations are independent across time, and there are no
  actions.
- The limit M → ∞ with the likelihood ratio as score: log-optimality (Thm 1,
  Props. 7–8).
- The **non-asymptotic** bound g_M ≥ g_∞ − log(1 + (ρ−1)/(M+1)) (their Thm 3).
  Our Theorem 2(iii) is this bound, in a weaker form. The paper now states it
  in their tighter form, with attribution. It holds in all 56 cases of
  experiment 4.
- Plug-in scores for a composite alternative, in a sequential example (§6.3).
- The improvement over under-powered exact e-values (their Cor. 1), which is
  related to our tempering remark.

**Still ours:**
- Actions chosen adaptively from the data, conditional simulation given
  history and action, and the necessity result (Proposition 2 of the paper).
- Arbitrary learned scores analysed through the tilted null: the
  decomposition KL(Q‖P) − KL(Q‖Q̂), and the explicit local efficiency M/(M+1).
- Posterior predictive scores: Bayes optimality, the Bayes-factor limit, the
  online Bayesian learning rule, and the comparison of probability-scale and
  ratio-scale predictives.
- The generative-drift and bandit experiments with learned scores.

The paper's abstract, introduction, attributions of Proposition 1 and
Theorem 2, and related-work section have been revised accordingly.
