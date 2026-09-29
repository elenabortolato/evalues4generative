# evalues4generative

A proof of concept for the research proposal *Simulation-Based E-Values and
Mixtures for Valid Inference in Dynamic and Adaptive Systems*.

**Question.** Can we monitor a stream of real data against a generative model
or simulator we can only *sample* from, with guarantees that hold at every
time step (anytime-valid) and under data-dependent (adaptive) allocation?

**Method (Proposition 2 of the proposal).** At step t, draw M samples
Y_1..Y_M from the simulator, conditional on the past and on the action taken.
Score them and the real observation X_t with a ratio estimator trained on the
past only. Then bet

    E_t = (M+1) T(X_t) / (T(X_t) + sum_j T(Y_j)),      T = exp(learned log-ratio).

Under H0, X_t and the Y_j are exchangeable, so E[E_t | past] = 1 *exactly*,
whatever the learner. The running product is therefore a test martingale:
reject when it reaches 1/alpha, at any time, with type-I error at most alpha
(Ville's inequality). No density is ever evaluated. To hedge the betting
fraction, the monitor averages the processes prod(1 - lam + lam E_t) over a grid
of lam values; that average is itself a test martingale.

The design follows the proposal's central principle: **the denominator carries
validity, the numerator carries power**. The learner only affects power.

## Results

alpha = 0.05. Rates are P(reject by the horizon). With 400 replications the
Monte Carlo standard error at 0.05 is about 0.011; with 200 it is about 0.015.

### Experiment 1: criticising a black-box generative model

The null simulator is a 2-D banana distribution (curvature b = 0.5). Under H1
the real data drift to b = 0.8. The drift changes neither the means nor the
marginal of x1, so the detector has to learn the shape of the discrepancy.
Horizon 500, M = 16, 400 replications.

| learner | method | false alarms (H0) | power (H1) | median stop (H1) |
|---|---|---|---|---|
| regularised (deg. 4, ridge 1) | **Monte Carlo e-value** | **0.030** | 0.990 | 198 |
| | plug-in learned ratio | 0.102 | 0.995 | 158 |
| | KS test, peeking every 10 steps | 0.323 | 0.750 | 250 |
| | oracle likelihood ratio | 0.030 | 1.000 | 35 |
| flexible (deg. 6, ridge 1e-3) | **Monte Carlo e-value** | **0.028** | 0.838 | 320 |
| | plug-in learned ratio | **0.588** | 0.998 | 66 |

![exp1](figures/exp1_generative_drift.png)

With the same learner and the same data, using the learned ratio directly as
the bet gives 10% false alarms with a well-regularised learner and **59%** with
an overfitting one. Calibrating it against simulator draws keeps false alarms
at about 3% in both cases. The worse learner costs power, but never validity.

### Experiment 2: an adaptive bandit checked against a digital twin

Three arms are allocated by Thompson sampling on the real outcomes, with
skewed noise. The twin simulates any arm. Under H1, arm 0 is 0.5 better in
reality than the twin predicts. Horizon 400, M = 16, 400 replications.
Thompson sampling pulls the best arm 78% of the time under H0; under H1 it
pulls arm 0 65% of the time.

| method | false alarms (H0) | power (H1) | median stop (H1) |
|---|---|---|---|
| **Monte Carlo e-value, twin run on the chosen arm** | **0.028** | 1.000 | 49 |
| Monte Carlo e-value, twin run on random arms | **1.000** | 1.000 | 40 |
| plug-in learned ratio | 0.055 | 1.000 | 44 |
| per-arm Welch t-test, peeking | 0.388 | 1.000 | 30 |
| oracle likelihood ratio | 0.043 | 1.000 | 15 |

![exp2](figures/exp2_adaptive_bandit.png)

Adaptive allocation does not break the construction, provided the simulator
is run *conditionally on the action actually taken*. If the simulator ignores
the action, exchangeability fails and the test rejects a true null every time.
This is exactly the requirement stated in the proposal (Risk (i)).

### Experiment 3: what the simulation budget buys

Setup as in Experiment 1 (regularised learner), with M varied; 200
replications.

| M | 1 | 2 | 4 | 8 | 16 | 64 |
|---|---|---|---|---|---|---|
| false alarms (H0) | 0.030 | 0.030 | 0.035 | 0.030 | 0.020 | 0.015 |
| power (H1) | 0.780 | 0.960 | 0.990 | 0.990 | 0.995 | 1.000 |
| median stop (H1) | 371 | 282 | 227 | 212 | 202 | 207 |

![exp3](figures/exp3_simulation_budget.png)

The method is valid for every M, including M = 1. Power saturates quickly: M = 4
already comes close to M = 64.

## What this does and does not show

- **Shown.**
  - Exact validity with a simulator-only null, for arbitrary and even badly
    overfitting learners, under optional stopping and adaptive allocation.
  - The failure of the two naive alternatives: using a learned ratio directly
    as the bet, and repeated fixed-sample tests.
  - The necessity of simulating conditionally on the action.
- **Price of validity.**
  - The Monte Carlo e-value is well behind the oracle likelihood ratio (median
    stopping time about 200 vs 35 in Experiment 1). The cost comes from
    learning the ratio online: no bets are placed during the first 20 steps,
    and the learner is refitted every 10 steps.
  - When the plug-in happens to be nearly calibrated (Experiment 2), it is
    slightly faster, but it has no guarantee.
  - Closing the gap to the oracle is RQ2 in the proposal.
- **Not covered yet.**
  - Composite or nuisance-parameter nulls (RQ3).
  - High-dimensional generative models, such as images or text embeddings.
  - Adaptive choice of M, and reuse of simulations across steps (RQ1).
  - Symmetry-based speed-ups (RQ4).

## Layout and reproduction

```
e4g/evalues.py      Monte Carlo factor, plug-in factor, lambda-mixture, product process
e4g/learners.py     online balanced logistic ratio estimator (polynomial features)
e4g/monitor.py      runs the competing e-processes side by side
experiments/        exp1_generative_drift.py, exp2_adaptive_bandit.py,
                    exp3_simulation_budget.py, make_figures.py
tests/              exactness tests (E[E_t] = 1 for arbitrary scores, etc.)
results/            JSON results and run logs; figures/  PNG figures
```

```bash
pip install -r requirements.txt
python -m pytest -q tests                        # ~30 s
python experiments/exp1_generative_drift.py      # ~10 min on 4 cores
python experiments/exp2_adaptive_bandit.py       # ~2.5 min
python experiments/exp3_simulation_budget.py     # ~4 min
python experiments/make_figures.py
```

All runs are seeded (replication i uses seed i under H0 and 10^6 + i under H1),
so reruns reproduce the tables exactly.
