# Proposta di tagli per la versione JASA

Obiettivo: portare il testo da 37 a circa 28–30 pagine (formato JASA, 12pt,
interlinea 1.45, circa 330 parole per pagina) senza perdere risultati. Tutto
quello che esce dal testo va nel supplemento (Sezione S4) o vi è già. I tagli
valgono solo per la versione JASA; `main.tex` (arXiv) resta com'è.

Per ogni punto: cosa cambia, il testo nuovo da incollare, e il risparmio stimato.
Il testo da sostituire si trova cercando l'inizio indicato.

| # | dove | risparmio stimato |
|---|---|---|
| 1 | 5.6: Tabella 7 nel supplemento, paragrafo sui costi accorciato | ~1,5 pagine |
| 2 | 5.3: Figura 2 nel supplemento, testo accorciato | ~0,7 pagine |
| 3 | 5.4: Tabella 4 nel supplemento, testo accorciato | ~0,8 pagine |
| 4 | Discussione: confronto con la letteratura in un paragrafo | ~1 pagina |
| 5 | Discussione: forme classiche della fattorizzazione in una frase | ~0,3 pagine |
| 6 | Algoritmo 1: tolti i commenti dopo ogni passo | ~0,6 pagine |
| 7 | Remark 1 accorciato | ~0,3 pagine |
| | **totale** | **~5 pagine** |

Con questi sette punti si arriva a circa 32 pagine di testo. Per scendere
ancora (punto 8, facoltativo) si può spostare nel supplemento la Sezione 4.3
dalla Proposizione 4 in poi, tenendo nel testo solo la regola di default:
circa 2–3 pagine in meno, ma è il taglio che pesa di più sul contenuto.

---

## 1. Sezione 5.6: Tabella 7 nel supplemento

**Cosa cambia.** La Tabella 7 (costo fino all'allarme, ms per query) va nella
Sezione S4. Nel testo restano i numeri che contano: 2,1 contro 53 ms per query;
6200–6900 contro 1100–2500 risposte generate.

**a)** Nel paragrafo che inizia con "Our simulation-likelihood rule and the
refit", sostituire la parte finale da "and its computing time per query" a
"at every refit." con:

```latex
and its computing time per query does not grow with the history: 2.1 and
2.2~ms over 400 and 1600 queries, against 8.0 and 53~ms for the refit, which
revisits the whole history at every refit.
```

**b)** Sostituire tutto il paragrafo che inizia con "Almost all the running
time of the audit" con:

```latex
Almost all the running time of the audit goes into the simulator: generating
the $M = 8$ replies of $P$ takes 2 to 8 seconds per query on four CPU cores,
whereas the bet and the update take at most 8~ms. The cost of an alarm is
therefore counted in generated replies: SLICE and the score fitted in advance
prepare 6000 replies before the audit and reach the alarm after 6200 to 6900,
whereas the learned scores prepare 800, only to standardise the features, and
reach it after 1100 to 2500 (Supplement, Section~S4, which also reports the
computing time to the alarm).
```

**c)** La Tabella 7 (`tab:llmcost`) esce dal testo e va nella Sezione S4 del
supplemento.

Risparmio: la tabella (~0,9 pagine) e circa 60 parole.

---

## 2. Sezione 5.3: Figura 2 nel supplemento

**Cosa cambia.** La Figura 2 va nella Sezione S4; il testo tiene i numeri.

Sostituire il testo della Sezione 5.3 ("Empirical growth rates") con:

```latex
On the drift task, validity holds at every budget, including $M = 1$ (false
alarms 1.5--3.5\%), and the median stopping time at $M = 1$ is 1.8 times that
at $M = 64$, close to the factor 2 of Theorem~\ref{thm:growth}(iv). With
$P = \mathcal N(0,1)$, $Q = \mathcal N(\delta, 1)$ and $T(x) = e^{\theta x}$,
where everything is explicit, the bounds of Theorem~\ref{thm:growth} hold in
all 56 configurations, and $g_M/g_\infty$ follows $M/(M+1)$ closely for
perfect, under- and overconfident scores alike. The regret bound of
Theorem~\ref{thm:simlik} holds on every path, with largest regret 7.5 against
a bound of 21.5 at $t = 2000$ (Supplement, Section~S4, with the figure).
```

Risparmio: la figura (~0,5 pagine) e circa 50 parole.

---

## 3. Sezione 5.4: Tabella 4 nel supplemento

**Cosa cambia.** La Tabella 4 va nella Sezione S4, dove ci sono già tutte le
varianti; il testo tiene il confronto essenziale.

Sostituire il testo della Sezione 5.4 ("Update rules") con:

```latex
All the update rules of Section~\ref{sec:predictive}, updated at every step
with no burn-in, give valid tests, with 1.3 to 3\% false alarms, whereas using
the logistic posterior odds directly as the bet gives 14 to 74\%. The logistic
Bayesian learning rule is the fastest and makes the method insensitive to the
flexibility of the features (power 0.97 with degree-6 features, against 0.84
for the refit). The simulation-likelihood learner, the one with the guarantee
of Theorem~\ref{thm:simlik}, is about 6\% slower on the drift task: the
Bayesian mixture pays for its pathwise bound with cautious early bets. All
variants are in the Supplement (Section~S4), and a comparison with the
Besag--Clifford e-process of \citet{dombowsky2026besag} in Section~S3.
```

Risparmio: la tabella (~0,4 pagine) e circa 90 parole.

Attenzione: la Sezione 5.5 cita i tempi mediani della Tabella 4 (214.5,
228.5); restano corretti, ma il lettore li troverà nella Tabella 5 e nel
supplemento.

---

## 4. Discussione: il confronto con la letteratura in un paragrafo

**Cosa cambia.** I due paragrafi che iniziano con "Monte Carlo tests compare a
fixed dataset" e "Two concurrent papers use a similar betting strategy"
ripetono in parte l'Introduzione (ultimo paragrafo prima dell'elenco delle
sezioni). Li unirei in uno, tenendo le attribuzioni.

Sostituire i due paragrafi con:

```latex
Monte Carlo tests compare a fixed dataset with simulations from the null
\citep{besag1989generalized}, and \citet{fischer2024sequential} make them
anytime-valid in the number of simulations; here the data themselves arrive
over time, with fresh simulations for each observation. The closest work is
\citet{dombowsky2026besag}, whose results on the likelihood-ratio score are
used, with attribution, in Section~\ref{sec:power}; \citet{forre2026anytime}
show that the likelihood ratio is the optimal score for every $M$, and
\citet{cortinovis2026slice} invert the e-process into confidence sequences for
the parameters of a simulator. All three treat independent data. Relative to
them, ours treats actions chosen from the data and simulators with state,
shows that the conditioning is necessary and quantifies the failure without
it, reads arbitrary scores through the tilted null with the local efficiency
$M/(M+1)$, and updates the score by Bayes' rule on the simulation likelihood,
with a pathwise regret bound, where the concurrent papers bet at a point
estimate.
```

Risparmio: circa 300 parole (~1 pagina).

---

## 5. Discussione: le forme classiche della fattorizzazione

**Cosa cambia.** Il paragrafo che inizia con "Computing a marginal likelihood
as a product of one-step predictives" riassume la Sezione S5. Lo ridurrei a
una frase, aggiunta in fondo al paragrafo precedente.

Cancellare il paragrafo e aggiungere alla fine del paragrafo che inizia con
"Once the simulator is run under the action taken":

```latex
The product of one-step predictives behind Section~\ref{sec:predictive} has
classical forms in prequential analysis, minimum description length and
sequential Monte Carlo (Supplement, Section~S5); prequential posteriors
\citep{sinharoy2026prequential} use simulations to calibrate a simulator,
where we use them to test it.
```

Risparmio: circa 70 parole (~0,3 pagine). Le citazioni di Rissanen, Kong et
al., Chopin, Fong e Holmes restano nel supplemento.

---

## 6. Algoritmo 1: senza i commenti dopo ogni passo

**Cosa cambia.** I commenti dopo i trattini ("--- It defines the null law",
"--- Exchangeable with $X_t$", ...) ripetono il testo delle Sezioni 2–4. Li
toglierei tutti tranne il clipping al passo 3, che serve per il Corollario
1(ii).

Sostituire l'elenco dei passi con:

```latex
\begin{enumerate}
  \item \textbf{Action.} Observe the action $A_t$ chosen by the design from the
        past (arm, prompt, dose).
  \item \textbf{Simulate.} Draw $Y_{t,1}, \ldots, Y_{t,M}$ from the simulator
        under $A_t$ from the current state, and observe $X_t$.
  \item \textbf{Score.} For each of the $M+1$ outcomes $x$, compute
        $\bar\eta = z(x)^\top \bar\theta_{t-1}$, $v = z(x)^\top S_{t-1}^{-1} z(x)$,
        $\kappa = (1 + \pi v/8)^{-1/2}$ and
        $s_{t-1}(x) = \mathrm{clip}(\kappa\,\bar\eta,\, -B,\, B)$, the predictive
        log-odds of ``real'' against ``simulated''. Clipping at $B$ (we use
        $B = 30$) keeps validity and makes Corollary~\ref{cor:cumulative}(ii)
        apply.
  \item \textbf{Bet.}
        $E_t = (M+1)\, e^{s_{t-1}(X_t)}/\{e^{s_{t-1}(X_t)} + \sum_{j=1}^{M} e^{s_{t-1}(Y_{t,j})}\}$.
  \item \textbf{Wealth.} $W_t = |\Lambda|^{-1}\sum_{\lambda\in\Lambda}
        \prod_{i\le t}(1-\lambda+\lambda E_i)$; reject $H_0$ when
        $W_t \ge 1/\alpha$.
  \item \textbf{Learn.} With labels $y_0 = 1$ for $X_t$ and $y_j = 0$ for
        $Y_{t,j}$, weights $w_0 = 1$ and $w_j = 1/M$, and
        $p_k = \sigma(s_{t-1}(U_{t,k}))$, update
        \[
        \begin{gathered}
        \nabla_t = \sum_{k=0}^{M} w_k (p_k - y_k)\, z_k, \qquad
        J_t = \sum_{k=0}^{M} w_k \kappa_k\, p_k (1 - p_k)\, z_k z_k^\top, \\
        S_t = S_{t-1} + J_t, \qquad \bar\theta_t = \bar\theta_{t-1} - S_t^{-1} \nabla_t ,
        \end{gathered}
        \]
        one step of the Bayesian learning rule \eqref{eq:blr}, with $\kappa_k$
        as in step 3 for $z_k$.
\end{enumerate}
```

Risparmio: circa 140 parole e due formule in display (~0,6 pagine). La
spiegazione di $\kappa_k p_k(1-p_k)$ come derivata dell'approssimazione probit
si perde; se la vuoi tenere, sono due righe in più.

---

## 7. Remark 1 accorciato

Sostituire il contenuto del Remark "Inside and outside the family" con:

```latex
\begin{remark}[Inside and outside the family]\label{rem:family}
SLICE bets with $s_t = \log (p_{\hat\theta_{t-1}}/p_{\theta_0})$ within a
parametrised family $\{P_\theta\}$ with null $\theta_0$, so its tilted null is
$P_{\hat\theta_{t-1}}$. Outside the family, $\hat\theta_t$ converges to
$\theta^* = \arg\min_\theta \KL(Q\|P_\theta)$, and by
Theorem~\ref{thm:growth}(ii) the growth tends to
$\KL(Q\|P_{\theta_0}) - \KL(Q\|P_{\theta^*})$, which is zero when
$\theta^* = \theta_0$, however large the departure. Both out-of-family
alternatives are of this kind, whereas our features contain $x_1^2$ and the
squared outcome of each arm, so a change of scale moves the tilted null towards
$Q$. Details, and the choice between the NRE-A and NRE-C estimators
\citep{miller2022contrastive}, are in the Supplement.
\end{remark}
```

Risparmio: circa 50 parole.

---

## 8. (Facoltativo) Sezione 4.3 nel supplemento dalla Proposizione 4 in poi

Il taglio più grande e il più delicato. Nel testo resterebbero: la Gaussiana
aggiornata in un passo \eqref{eq:blr}, la regola logistica di default e
l'Algoritmo 1. Andrebbero nel supplemento la Proposizione 4 con la sua
dimostrazione, la regola predictive-first \eqref{eq:mp} e il paragrafo sui
martingale posteriors, con una frase di rimando. Risparmio: 2–3 pagine. Lo
sconsiglio se pensi di rafforzare la teoria sull'aggiornamento gaussiano
(il lavoro sul regret), perché la Proposizione 4 ne è il punto di partenza.
