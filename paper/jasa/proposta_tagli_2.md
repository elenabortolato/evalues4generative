# Seconda proposta: frasi da assottigliare

Frasi che ripetono cose già dette altrove, riassumono il paragrafo appena
letto o fanno da slogan. Nessun risultato si perde. Per ogni punto: dove si
trova (inizio della frase), cosa togliere o il testo nuovo, parole risparmiate.
Risparmio totale stimato: circa 650 parole, cioè 2 pagine nel formato JASA.

---

## Introduzione

**1. Esempio delle piattaforme** (paragrafo "The same question comes up with
multi-armed bandits"). L'esempio è lungo e ha sei citazioni. Sostituire da
"For example, a content-ranking" fino a "\citep{zhao2020sim}." con:

```latex
For example, a ranking or moderation system changes what users post and hence
what the platform observes next \citep{chaney2018algorithmic,jiang2019degenerate},
and such systems are tuned on simulators of user behaviour whose fidelity is
exactly what is in doubt \citep{ie2019recsim,shi2019virtual,zhao2020sim}.
```
Circa 25 parole.

**2. "Validity and power are separate questions"** (paragrafo dopo l'elenco
dei contributi). Lo stesso concetto apre la Discussione. Sostituire il
paragrafo con:

```latex
Validity comes from how the simulator is run, not from the score, so any score
fixed before the new outcome is seen, including a network trained in advance,
gives a valid test. Our update rules are meant for the usual situation in which
nothing is known about the alternative: a twin tends to fail in ways its
designers did not anticipate.
```
Circa 35 parole.

## Sezioni 3 e 4

**3. Commento al Corollario 1** (paragrafo "Part (i) needs no assumption on
the scores"). La prima frase ripete l'enunciato. Cancellare da "Part (i)
needs no assumption" fino a "whatever the learner." e iniziare il paragrafo
con "Since $1-\lambda+\lambda E$ is a weighted average". Circa 30 parole.

**4. Fine della Sezione 4.1.** Due frasi riassumono quanto appena detto o
ripetono l'Introduzione. Cancellare:
- "A learned score therefore pays, at every step, for its error as a
  forecaster, and only in power." (fine del paragrafo "Indeed, by the limit
  of the bet above");
- "And since a forecast need only be issued before the outcome it is scored
  on, any learner, exact or approximate, trained online or in advance, keeps
  the validity of Theorem~\ref{thm:valid}." (fine del paragrafo "Reading the
  score as a forecast").

Circa 50 parole.

**5. Commento alla parte (i) del Teorema 3** (paragrafo "Part (i) says that
the score can be updated by Bayes' rule exactly"). Sostituire il paragrafo
con:

```latex
Part (i) says that the score can be updated by Bayes' rule exactly, although
the normalising constant $Z(\theta)$ of the tilt is unknown. The simulation
likelihood is not the density of the real outcome under the tilt; it answers
the question of Section~\ref{sec:method}, which of the $M+1$ outcomes at hand
is the real one, and needs only their scores. Its posterior predictive is the
bet itself, so the guarantees of Bayesian mixtures hold for the wealth, and as
$M \to \infty$ it recovers the Bayes factor of Proposition~\ref{prop:bayes}.
```
Circa 50 parole.

## Sezione 5

**6. Apertura della Sezione 5** ("The experiments ask whether a learned score
can break validity"). L'elenco ripete i titoli delle sottosezioni. Sostituire
la prima frase (fino a "(Section~\ref{sec:exp-llm})."), tenendo il resto del
paragrafo, con:

```latex
The experiments check validity with learned scores and adaptive allocation,
the theory of Sections~\ref{sec:power} and~\ref{sec:predictive}, the update
rules and the concurrent methods, and the gain from adaptive actions in an
audit with real model outputs.
```
Circa 35 parole.

**7. Sezione 5.6, apertura del paragrafo "Knowing the substitute pays when
the guess is right and fails when it is not."** È uno slogan; il paragrafo lo
dimostra subito dopo. Cancellare la frase e iniziare con "With our design the
score fitted in advance". Circa 15 parole.

## Discussione

**8. Primo paragrafo** ("The Monte Carlo e-process separates validity").
Riassume tutto l'articolo. Sostituire con:

```latex
The Monte Carlo e-process separates validity, the control of false alarms,
from power. Validity comes only from how the simulator is run, through
exchangeability with fresh simulations under the action taken and from the
current state, and holds for any score and any policy, including interim
analyses and re-assignments. Power comes only from the score, through the
distance between its tilted null and the truth (Theorem~\ref{thm:growth});
approximations in the score therefore cost power but never validity.
```
Circa 70 parole.

**9. Esempio dei pasti non annunciati** (paragrafo "The tilted null also makes
a rejection informative"). L'esempio occupa otto righe. Sostituire da "For a
single patient with unannounced meals" fino a "gives no evidence that its part
of the twin is right." con:

```latex
For a single patient, for instance, monitors on fasting windows and on windows
after declared meals test the twin's basal physiology and its model of meal
absorption separately; a monitor that does not reject gives no evidence that
its part of the twin is right.
```
Circa 60 parole.

**10. Limiti: il posterior temperato** (paragrafo "Several limitations
remain", da "It holds for the exact posterior" a "the most useful next step").
Accorciare a:

```latex
It holds for the exact posterior, whereas the one-step Gaussian update used in
practice is covered only through Proposition~\ref{prop:approx}; and the
logistic learner, which has no regret guarantee, is faster than the
simulation-likelihood learner, by about 6\% on the drift task and 60\% in the
trial, because the latter bets cautiously. A tempered posterior
$\pi_t \propto \pi_0 L_t^{\gamma}$ with $\gamma > 1$
\citep{sinharoy2026prequential} would bet more and stays valid; a regret bound
for it is the most useful next step.
```
Circa 50 parole.

**11. Limiti: lo stato del sistema fisico** (stesso paragrafo, da "Exact
validity requires" a "long monitoring horizons."). Accorciare a:

```latex
Exact validity requires the simulator to be run from the state of the real
system, which is known in silico but must be inferred for a physical twin; an
approximate filter makes exchangeability only approximate. A persistent error
of divergence $\delta$ per step can then raise a false alarm, but not much
before $\log(1/\alpha)/\delta$ steps (Corollary~\ref{cor:falsealarm}(ii)).
```
Circa 60 parole. Si perde la precisazione che campionare lo stato da un filtro
esatto mantiene la validità esatta; se ti serve, è una riga in più.

---

Cosa non toccherei: le spiegazioni dopo la Proposizione 1 ("The bet pays when
the score picks out the real outcome") e dopo il Teorema 2, perché servono a
chi non conosce gli e-value, e la frase sulla tilted null per il design, che è
un'idea nuova.
