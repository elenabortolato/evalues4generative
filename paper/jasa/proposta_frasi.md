# Venti frasi da accorciare (versione JASA)

Ogni frase nuova è più corta di almeno il 20% (parole contate come nel sorgente, formule comprese). In tutto si passa da 1139 a 852 parole (−287, circa 0,9 pagine). Per ogni frase: dove si trova, la versione attuale, quella proposta e, se c'è, che cosa si perde.

## 1. Sezione 1 (Introduzione): 44 → 34 parole (−23%)

**Ora:**

```latex
In these settings, as we choose actions based on past observations, can we continuously test whether the resulting stream of outcomes is still consistent with a simulator, and sound an alarm as soon as it is not, while controlling false alarms under optional stopping?
```

**Proposta:**

```latex
With actions chosen from past observations, can we test continuously whether the outcomes remain consistent with a simulator, raising an alarm as soon as they do not, with false alarms controlled under optional stopping?
```

## 2. Sezione 1 (Introduzione): 64 → 47 parole (−27%)

**Ora:**

```latex
They also learn the score differently: \citet{forre2026anytime} refit it by maximising its empirical growth, and \citet{cortinovis2026slice} train a neural ratio estimator before the data arrive, over the simulator's own parametric family, so that their score can forecast only departures that the family contains; our scores reweight the simulator by features of the outcome, and detect departures that its designers did not build into it.
```

**Proposta:**

```latex
They also learn the score differently: \citet{forre2026anytime} refit it by maximising its empirical growth, and \citet{cortinovis2026slice} train a neural ratio estimator in advance on the simulator's parametric family, so that their score forecasts only the departures it contains; ours reweight the simulator by features of the outcome.
```

*Si perde: «e rilevano deviazioni che i progettisti del simulatore non hanno previsto» (lo dice già il paragrafo dopo i contributi).*

## 3. Sezione 1 (Introduzione): 51 → 41 parole (−20%)

**Ora:**

```latex
We first recall why the Bayes factor is a product of forecasts and show that the Monte Carlo e-process has the same structure, with the score in place of the model (Section~\ref{sec:scores}); we then derive an update rule for the score with a pathwise guarantee (Section~\ref{sec:simlik}), and its practical version (Section~\ref{sec:blr}).
```

**Proposta:**

```latex
We show that the Monte Carlo e-process has the structure of the Bayes factor, with the score in place of the model (Section~\ref{sec:scores}), and derive an update rule for the score with a pathwise guarantee (Section~\ref{sec:simlik}) and its practical version (Section~\ref{sec:blr}).
```

## 4. Sezione 2.1: 50 → 36 parole (−28%)

**Ora:**

```latex
The real system then produces an outcome $X_t$ with conditional law $Q_t(\cdot \mid \F_{t-1}, A_t)$, and the analyst can draw samples from a simulator with conditional law $P_t(\cdot \mid \F_{t-1}, A_t)$; for a digital twin this means re-initialising the twin at the current state and running it under action $A_t$.
```

**Proposta:**

```latex
The real system then produces $X_t$ with law $Q_t(\cdot \mid \F_{t-1}, A_t)$, and a simulator can be sampled from $P_t(\cdot \mid \F_{t-1}, A_t)$; a digital twin is re-initialised at the current state and run under $A_t$.
```

## 5. Sezione 2.2: 48 → 33 parole (−31%)

**Ora:**

```latex
If $\log T_t = \log(q_t/p_t) + \varepsilon$ with $\E[\varepsilon] = 0$, then $\E[Z_t] = \E[e^{\varepsilon}] > 1$ by Jensen's inequality even when $q_t = p_t$, so estimation noise alone makes the plug-in bets favourable under the null; this is why an overfitted learner gives many false alarms (Section~\ref{sec:exp-drift}).
```

**Proposta:**

```latex
If $\log T_t = \log(q_t/p_t) + \varepsilon$ with $\E[\varepsilon] = 0$, then $\E[Z_t] = \E[e^{\varepsilon}] > 1$ even when $q_t = p_t$, so noise alone makes plug-in bets favourable under the null (Section~\ref{sec:exp-drift}).
```

*Si perde: «an overfitted learner gives many false alarms», che la Sezione 5.1 mostra comunque con i numeri.*

## 6. Sezione 2.2: 64 → 50 parole (−22%)

**Ora:**

```latex
The theorem involves no density, no property of the score and no restriction on the policy, which may depend on the data and on the wealth itself; in a MAMS trial the action $A_t$ is the arm assigned to a participant for a period, chosen after interim analyses and re-assignments, and the conditional law $P_t$ is that of the participant's outcome given their history.
```

**Proposta:**

```latex
The theorem needs no density, no property of the score and no restriction on the policy, which may depend on the data and the wealth; in a MAMS trial $A_t$ is the arm assigned after interim analyses and re-assignments, and $P_t$ the law of the outcome given the participant's history.
```

## 7. Sezione 4.1: 74 → 54 parole (−27%)

**Ora:**

```latex
A score can only forecast within the family of its tilted nulls, so the features determine which departures can be detected: for a score built from the simulator's own parameters, as in SLICE, the tilted nulls belong to that parametric family, and a departure outside it is invisible however accurate the score (Section~\ref{sec:exp-concurrent}), whereas a score learned on features of the outcome reweights the simulator in directions that need not correspond to any parameter.
```

**Proposta:**

```latex
A score can only forecast within the family of its tilted nulls, so the features determine which departures are detectable: a score built from the simulator's parameters, as in SLICE, misses any departure outside that family (Section~\ref{sec:exp-concurrent}), whereas features of the outcome reweight the simulator in directions that need not correspond to any parameter.
```

## 8. Sezione 5.1: 53 → 37 parole (−30%)

**Ora:**

```latex
The simulator is the two-dimensional ``banana'' $x = (z_1, z_2 + b(z_1^2 - 1))$, $z \sim \mathcal N(0, I)$, with $b = 0.5$; under the alternative the real data drift to $b = 0.8$, which changes neither the means nor the marginal of $x_1$, so the shape of the discrepancy must be learned.
```

**Proposta:**

```latex
The simulator is the ``banana'' $x = (z_1, z_2 + b(z_1^2 - 1))$, $z \sim \mathcal N(0, I)$, $b = 0.5$; under the alternative $b = 0.8$, which changes neither the means nor the marginal of $x_1$.
```

*Si perde: «so the shape of the discrepancy must be learned».*

## 9. Sezione 5.2: 45 → 35 parole (−22%)

**Ora:**

```latex
The conditioned e-process stays valid, with 2.8\% false alarms (Table~\ref{tab:bandit}), whereas the same e-process with the twin run on uniformly random arms, the setting of Proposition~\ref{prop:necessary}, rejects the true null in every run, and a per-arm $t$-test recomputed every ten steps in 39\% of runs.
```

**Proposta:**

```latex
The conditioned e-process stays valid, with 2.8\% false alarms (Table~\ref{tab:bandit}); with the twin on random arms (Proposition~\ref{prop:necessary}) it rejects the true null in every run, and a per-arm $t$-test recomputed every ten steps in 39\%.
```

## 10. Sezione 5.5: 62 → 48 parole (−23%)

**Ora:**

```latex
Scores that know where the departure lies are fastest inside the family, SLICE after 67 steps on the drift and a score fitted on the true alternative after 58, against about 210 for scores learned from the data; but outside the family their power falls to 0.01 on the drift (Remark~\ref{rem:family}), while every score learned from real against simulated outcomes keeps it.
```

**Proposta:**

```latex
Scores that know where the departure lies are fastest inside the family (SLICE after 67 steps on the drift, a score fitted on the true alternative after 58, learned scores after about 210), but outside it their power falls to 0.01 (Remark~\ref{rem:family}), while every learned score keeps it.
```

## 11. Sezione 5.6: 48 → 33 parole (−31%)

**Ora:**

```latex
As in the routing audit of \citet{cortinovis2026slice}, the prompts ask for random output (eight random digits, an integer between 1 and 100, a colour, six coin flips), so that a substitute shows up as a change in the frequencies of the replies rather than as a wrong answer.
```

**Proposta:**

```latex
As in \citet{cortinovis2026slice}, the prompts ask for random output (eight digits, an integer from 1 to 100, a colour, six coin flips), so a substitute shows up in the frequencies of the replies.
```

*Si perde: «rather than as a wrong answer» e il nome «routing audit».*

## 12. Sezione 5.6: 66 → 45 parole (−32%)

**Ora:**

```latex
For each prompt the auditor computes the average log growth of its past bets, adds a bonus that shrinks as the prompt is asked more often, and asks the prompt with the largest sum, so that it tries prompts it knows little about as well as those that have paid; this is the upper confidence bound (UCB) rule of bandit problems \citep{lattimore2020bandit}, used here without tuning.
```

**Proposta:**

```latex
For each prompt the auditor adds to the average log growth of its past bets a bonus that shrinks as the prompt is asked more often, and asks the prompt with the largest sum: the upper confidence bound (UCB) rule of bandit problems \citep{lattimore2020bandit}, untuned.
```

*Si perde la spiegazione intuitiva del bonus («tries prompts it knows little about as well as those that have paid»).*

## 13. Sezione 5.6: 61 → 49 parole (−20%)

**Ora:**

```latex
The score fitted in advance and SLICE learn $R$ before the audit from 500 replies per prompt of the 4-bit model, so they guess wrong for the 3-bit and 0.5B substitutes; routing that depends on the prompt lies outside SLICE's family, and the sharpened $P$ lies outside it in a stronger sense, since it is closer to $P$ than to $R$.
```

**Proposta:**

```latex
The score fitted in advance and SLICE learn $R$ from 500 replies per prompt of the 4-bit model, so they guess wrong for the other substitutes; routing that depends on the prompt lies outside SLICE's family, and the sharpened $P$, closer to $P$ than to $R$, lies further outside.
```

## 14. Sezione 5.6: 60 → 44 parole (−27%)

**Ora:**

```latex
Choosing the prompts helps every score (Table~\ref{tab:llm}): with our adaptive design every score needs fewer queries than with uniformly drawn prompts, in every scenario in which it rejects and up to 61\% fewer, and our scores gain up to 0.24 in power; the auditor learns to ask for a colour more often, because that is where the substitutes differ most.
```

**Proposta:**

```latex
Choosing the prompts helps every score (Table~\ref{tab:llm}): every score needs fewer queries than with uniform prompts wherever it rejects, up to 61\% fewer, and our scores gain up to 0.24 in power; the auditor learns to ask for colours, where the substitutes differ most.
```

## 15. Sezione 5.6: 47 → 33 parole (−30%)

**Ora:**

```latex
When the API sharpens $P$, however, a score fitted to $R$ has a tilted null close to $R$, hence a growth rate close to $\KL(Q\|P) - \KL(Q\|R) < 0$ by Theorem~\ref{thm:growth}: it never rejects, nor does SLICE, which estimates a routing fraction of zero and stops betting.
```

**Proposta:**

```latex
When the API sharpens $P$, a score fitted to $R$ has a tilted null near $R$ and a growth rate near $\KL(Q\|P) - \KL(Q\|R) < 0$ (Theorem~\ref{thm:growth}): it never rejects, nor does SLICE.
```

*Si perde: «which estimates a routing fraction of zero and stops betting», cioè il perché SLICE non rifiuta.*

## 16. Sezione 5.6: 50 → 37 parole (−26%)

**Ora:**

```latex
In exchange our rule carries the guarantee of Theorem~\ref{thm:simlik}, through Proposition~\ref{prop:approx} for the one-step update, and its computing time per query does not grow with the history: 2.1 and 2.2~ms over 400 and 1600 queries, against 8.0 and 53~ms for the refit, which revisits the whole history at every refit.
```

**Proposta:**

```latex
In exchange our rule carries the guarantee of Theorem~\ref{thm:simlik}, through Proposition~\ref{prop:approx}, and its cost per query does not grow with the history: 2.1 and 2.2~ms over 400 and 1600 queries, against 8.0 and 53~ms for the refit.
```

## 17. Sezione 5.6: 61 → 46 parole (−25%)

**Ora:**

```latex
The cost of an alarm is therefore counted in generated replies: SLICE and the score fitted in advance prepare 6000 replies before the audit and reach the alarm after 6200 to 6900, whereas the learned scores prepare 800, only to standardise the features, and reach it after 1100 to 2500 (Supplement, Section~S4, which also reports the computing time to the alarm).
```

**Proposta:**

```latex
The cost of an alarm is therefore counted in generated replies: SLICE and the score fitted in advance prepare 6000 and reach the alarm after 6200 to 6900, the learned scores prepare 800, to standardise the features, and reach it after 1100 to 2500 (Supplement, Section~S4).
```

## 18. Sezione 7: 71 → 56 parole (−21%)

**Ora:**

```latex
The monitor rejects when its forecasts, the tilted nulls of its scores, have predicted reality better than the simulator; as the posterior concentrates, the tilted null approaches the reweighting of the simulator closest to reality, $\arg\min_\theta \KL(Q\|Q_\theta)$, and its log ratio to the simulator, $\theta^\top\phi - \log Z(\theta)$, shows where reality puts more or less mass than the simulator: a shift in the mean, an excess of variance, a particular arm.
```

**Proposta:**

```latex
The monitor rejects when its forecasts have predicted reality better than the simulator; as the posterior concentrates, the tilted null approaches the closest reweighting of the simulator, $\arg\min_\theta \KL(Q\|Q_\theta)$, whose log ratio $\theta^\top\phi - \log Z(\theta)$ shows where reality puts more or less mass: a shift in the mean, an excess of variance, a particular arm.
```

## 19. Sezione 7: 63 → 49 parole (−22%)

**Ora:**

```latex
Relative to them, ours treats actions chosen from the data and simulators with state, shows that the conditioning is necessary and quantifies the failure without it, reads arbitrary scores through the tilted null with the local efficiency $M/(M+1)$, and updates the score by Bayes' rule on the simulation likelihood, with a pathwise regret bound, where the concurrent papers bet at a point estimate.
```

**Proposta:**

```latex
Ours adds actions chosen from the data and simulators with state, the necessity of the conditioning, the reading of any score through its tilted null with local efficiency $M/(M+1)$, and Bayesian updating of the score with a pathwise regret bound, where the concurrent papers bet at a point estimate.
```

## 20. Sezione 7: 57 → 45 parole (−21%)

**Ora:**

```latex
Once the simulator is run under the action taken, the design may be chosen to gather evidence where it accumulates fastest: in the audit this makes every score faster, ours and the concurrent ones alike, by 8\% to 61\% in the number of queries, while without the conditioning on the action the concurrent methods lose validity altogether.
```

**Proposta:**

```latex
Once the simulator is run under the action taken, the design may gather evidence where it accumulates fastest: in the audit this saves every score, ours and the concurrent ones, 8\% to 61\% of the queries, whereas without the conditioning the concurrent methods lose validity.
```
