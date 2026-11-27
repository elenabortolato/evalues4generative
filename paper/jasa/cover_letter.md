[Place, date]

To the Editors, *Journal of the American Statistical Association*, Theory and Methods

**Submission: "Betting against the simulator: Monte Carlo e-processes under adaptive actions"**

Dear Editors,

I submit the above manuscript for consideration in the Theory and Methods section of JASA.

**The problem.** Clinical trials with interim analyses, bandit experiments and digital twins all collect data adaptively: what is observed next depends on what has been seen. More and more often, the model of the system is available only as a simulator, such as the UVA/Padova simulator used to design insulin-dosing trials, a simulator of user behaviour, or a language model run locally. The question is whether the real system still behaves like its simulator. Answering it requires a test that can be checked after every observation, that stays valid when the actions depend on the past, and that needs no likelihood.

**What the manuscript provides.** A score learned by classifying real against simulated outcomes becomes an exact bet when it is normalised over fresh simulations run under the action actually taken and from the current state of the system. The resulting e-process is valid for every allocation rule and every score, under interim analyses, re-assignment and optional stopping. The main results are the following.

1. *Validity under adaptive actions, and why the conditioning is needed.* Without conditioning on the action, the expected bet tends to $1 + \chi^2$ and false alarms grow exponentially, at a rate of at most $\log K$ per step with $K$ arms.
2. *Exact Bayesian updating of the score.* The bet is a conditional-logit likelihood in the score's parameter, which the simulations normalise. Posterior-predictive bets are updated in one step per observation and stay within $d \log t$ of the best score in hindsight on every sample path; approximate posteriors keep validity exact.
3. *The power of an arbitrary score.* The growth rate splits into a simulation cost, $M/(M+1)$ for local alternatives, and a learning cost, the divergence between the truth and the score's implied forecast.
4. *Adaptive actions as a source of power.* In an audit of a language-model API with real model outputs, choosing the prompts adaptively makes every score faster, including those of two concurrent methods, by up to 61% in the number of queries.

**Application.** A three-stage multi-arm trial of insulin-dosing strategies, with arms dropped for futility and participants re-assigned, is checked against its in-silico twin. Only the monitor conditioned on each participant's state and arm controls false alarms (2% of trials, against 95% for a twin that ignores the participants' history), and per-arm e-processes combined by e-BH identify the arm on which the twin fails.

**Why JASA.** The method addresses a problem common to clinical trials, adaptive experimentation and the validation of AI systems, and its guarantees hold for any allocation rule and any learned score. I believe it is of interest to the broad readership of the Theory and Methods section.

**Concurrent work.** Two related preprints appeared on arXiv while the manuscript was in preparation: Forré and Brenner (arXiv:2610.08210, 6 October 2026) and Cortinovis, Battaglia and Caron (arXiv:2610.13855, October 2026). Both use the normalised bet for independent data. The manuscript cites both, attributes the shared results to them where they are used, and compares with both methods in Sections 5.5 and 5.6. Section S9 of the supplement records the timing.

**Format and supporting material.** The manuscript is 35 pages in the required format, including the title page, abstract, tables, figures and references. The notation, all proofs, the experimental details, and further results and related work are in the supplement, which is anonymised like the manuscript. Code, data and the workflow that reproduces every table and figure are ready, and I will submit them with the Author Contributions Checklist if the manuscript is invited for revision.

**Declarations.** The manuscript is not under consideration elsewhere. A preprint is posted on arXiv ([arXiv identifier]), so I am aware that anonymity cannot be fully guaranteed. I have no competing interests to declare.

[Optional: suggested associate editors or reviewers, with affiliation and e-mail, and any reviewers to exclude.]

Yours sincerely,

Elena Bortolato
CUNEF Universidad, Madrid
