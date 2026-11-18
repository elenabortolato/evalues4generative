[Place, date]

To the Editors, *Statistics and Computing*

**Submission: "Betting against the simulator: Monte Carlo e-processes under adaptive actions"**

Dear Editors,

I submit the above manuscript for consideration as an original research article in *Statistics and Computing*.

**Summary.** Multi-arm multi-stage trials, bandits and digital twins collect data adaptively, and the models they rely on are often available only as simulators. The manuscript gives an anytime-valid test of whether a simulator reproduces such a system. A score learned by classifying real against simulated outcomes becomes an exact bet when it is normalised over fresh simulations run under the action actually taken and from the current state. The resulting e-process is valid for every allocation rule and every score, so it remains valid under interim analyses, arm dropping and optional stopping.

**Contributions.**

1. *Adaptive actions: validity and its necessity.* Validity requires the simulator to be conditioned on the action and on its state. Without this conditioning, the expected bet tends to 1 + χ² and false alarms grow exponentially, at a rate of at most log K per step with K arms.
2. *Prequential Bayesian updating of the score.* The bet is a conditional-logit likelihood that the simulations normalise. Treating it as a likelihood gives exact Bayesian updating with posterior-predictive bets, in one step per observation, and a regret bound that holds on every sample path.
3. *Adaptive actions as a source of power.* In an audit of a language-model API with real model outputs, choosing the prompts adaptively makes every score faster, including those of two concurrent methods, by up to 61% in the number of queries.
4. *Power of an arbitrary score.* The growth rate splits into a simulation cost, exactly M/(M+1) for local alternatives, and a learning cost.
5. *Application.* A multi-arm multi-stage trial of insulin-dosing strategies is checked against its in-silico twin, the UVA/Padova simulator. Only the monitor conditioned on each participant's state and arm controls false alarms, and per-arm e-processes combined by e-BH identify the arm on which the twin fails.

**Fit with the journal.** The manuscript combines Monte Carlo methodology, simulation-based inference and sequential computation, with attention to computational cost (Section 5.6). I believe it suits the readership of *Statistics and Computing*.

**Concurrent work.** Two related preprints appeared on arXiv while the manuscript was in preparation: Forré and Brenner (arXiv:2610.08210, 6 October 2026) and Cortinovis, Battaglia and Caron (arXiv:2610.13855, October 2026). Both develop the normalised bet for independent data. The manuscript cites both papers, attributes the shared results to them where they are used, and compares with both methods in Sections 5.5 and 5.6. Appendix D records the timing, based on the version history of the code repository.

**Reproducibility.** All code, results and the generated language-model data are available at https://github.com/elenabortolato/evalues4generative. The file REPRODUCIBILITY.md maps every table and figure to the script that produces it, and all runs are seeded.

**Declarations.** The manuscript is not under consideration elsewhere. A preprint is available on arXiv [arXiv identifier to be added]. I have no competing interests to declare.

[Optional: suggested reviewers, with affiliation and e-mail, and any reviewers to exclude.]

Yours sincerely,

Elena Bortolato
CUNEF Universidad, Madrid
