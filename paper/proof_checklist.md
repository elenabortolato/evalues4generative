# Controllo delle dimostrazioni: un pomeriggio (circa 4 ore)

Le dimostrazioni sono nell'Appendice B. Proposizione 1, Teorema 1 e
Proposizione 4 sono nel testo. L'ordine qui sotto segue le dipendenze: ogni
risultato usa solo quelli prima.

Due fatti tornano ovunque. Conviene fissarli subito:

- **(F1)** sotto lo stesso $P$, $\mathbb E_{P\otimes P^M}[E^{(M)}] = 1$ (Proposizione 1);
- **(F2)** cambio di misura: $\mathbb E_{Q\otimes P^M}[f/(dQ/dP)(X)] \le \mathbb E_{P\otimes P^M}[f]$ per $f\ge 0$.

---

## 1. Proposizione 1 e Teorema 1: validità (15 min)

- [ ] **Prop. 1.** I pesi $T(U_k)/\sum T$ sommano a 1 e per scambiabilità hanno la stessa media, quindi ciascuno ha media $1/(M+1)$.
- [ ] **Teo. 1.** Condizionatamente a $\mathcal G_t$, $(X_t, Y_{t,1..M})$ sono i.i.d. sotto $H_0$. Serve che le $Y$ siano indipendenti da $X_t$ e che lo score non usi né $X_t$ né le $Y$ del passo $t$.
- [ ] Il passaggio da $\mathcal G_t$ a $\mathcal F_{t-1}$ è per la proprietà della torre. La media su $\Lambda$ di martingale è una martingala.

**Attenzione:** nessuna.

## 2. Teorema 2: tasso di crescita (60 min, la parte più lunga)

- [ ] **(i)** È Jensen più (F2) più (F1). Controlla che serva $\mathrm{KL}(Q\|P)<\infty$ per sottrarlo.
- [ ] **(ii)** Per la legge dei grandi numeri $V/M\to Z$. Con $|s|\le B$ si ha $|\log E^{(M)}|\le 2B$, e la convergenza dominata dà il limite. L'identità $g_\infty = \mathrm{KL}(Q\|P)-\mathrm{KL}(Q\|\tilde T)$ viene da $\log(d\tilde T/dP)=\log(dQ/dP)-\log(dQ/d\tilde T)$.
- [ ] **(iii)** Due Jensen nell'ordine giusto: prima sulle $Y$ ($\mathbb E V = MZ$), poi su $X$ ($\mathbb E_Q T/Z=\rho$). Il caso $g_\infty=-\infty$ è banale.
- [ ] **(iv)** Sviluppo al secondo ordine di $\log\frac{1}{M+1}\sum_k e^{\varepsilon h(U_k)}$. Controlla tre cose:
  - $\mathbb E[h(X)-\bar h] = \frac{M}{M+1}\varepsilon\,\mathbb E_P[\tilde h\psi]$;
  - la varianza empirica (normalizzata con $M+1$) ha media $\frac{M}{M+1}\mathrm{Var}_P(h)$;
  - $2\mathbb E[\tilde h\psi]-\mathbb E[\tilde h^2] = \mathbb E[\psi^2]-\mathbb E[(\psi-\tilde h)^2]$.
- [ ] Le costanti dell'enunciato e della dimostrazione coincidono: c'è $\varepsilon^2/2$ sia in $g_M$ sia in $\mathrm{KL}(Q\|P)$.

**Attenzione: manca una riga.** La formula dell'efficienza "$\approx \frac{M}{M+1}\times(1-\mathrm{KL}(Q\|\tilde T)/\mathrm{KL}(Q\|P))$" dopo il teorema usa $\mathrm{KL}(Q\|\tilde T) = \frac{\varepsilon^2}{2}\mathbb E_P[(\psi-\tilde h)^2]+O(\varepsilon^3)$, che la dimostrazione non scrive. Il conto è giusto (l'ho rifatto). La riga è già aggiunta nel repository, in fondo alla (iv); in Overleaf va aggiunta così:

> Similarly, $\log(d\tilde T/dP) = \varepsilon\tilde h - \frac{\varepsilon^2}{2}\mathrm{Var}_P(h) + O(\varepsilon^3)$, so $\mathrm{KL}(Q\|\tilde T) = \mathrm{KL}(Q\|P) - \mathbb E_Q[\log(d\tilde T/dP)] = \frac{\varepsilon^2}{2}\mathbb E_P[(\psi-\tilde h)^2] + O(\varepsilon^3)$, which gives the efficiency display.

## 3. Proposizione 2: il condizionamento è necessario (40 min)

- [ ] **(i)** Il limite quasi certo è facile. Per le medie, l'evento $G_M=\{\bar T_M\ge Z/2\}$ dà su $G_M$ il limite $E_t\le 4B/Z$ (convergenza dominata). Su $G_M^c$, $E_t\le M+1$ e Hoeffding dà $P(G_M^c)\le e^{-MZ^2/(2B^2)}$. Controlla l'esponente: $2M(Z/2)^2/B^2$.
- [ ] **(ii)** L'insieme $A$ con $P_a(A)>\bar P(A)$, poi $\chi^2 = \int R_a^2 d\bar P - 1$. Lo score troncato $T_c=\min(R_a,c)+1/c$ è limitato e lontano da zero, e per convergenza monotona il limite tende a $1+\chi^2$, anche quando $\chi^2=\infty$.
- [ ] **(iii)** La catena delle costanti:
  - $\mu=(\rho-1)/2$;
  - $b/B\le E_t\le B/b$, quindi $\sigma^2=(B/b)^2$;
  - $\log(1+x)\ge x-2x^2$ per $x\ge-1/2$;
  - $\lambda\mu-2\lambda^2\sigma^2\ge\lambda\mu/2$ se $\lambda\le\mu/(4\sigma^2)=(\rho-1)b^2/(8B^2)$;
  - la conclusione è $\liminf\ge\lambda\mu/2=\lambda(\rho-1)/4$.

  Le costanti coincidono con l'enunciato (l'ho rifatto).
- [ ] Azuma–Hoeffding con incrementi limitati da $C$, poi Borel–Cantelli. Sull'evento $\Omega_0$ il tempo $t_0$ è finito, quindi i primi termini sono trascurabili.

**Attenzione:** $M_0$ esiste perché $\rho = 1+2\mu > 1+\mu$. Basta verificare che sia scritto così.

## 4. Corollari 1 e 2 (30 min)

- [ ] **Cor. 1.**
  - Si ha $W_t\ge|\Lambda|^{-1}\times$ ogni prodotto.
  - Gli incrementi stanno in $[\log(1-\lambda),\log(1+\lambda M)]$ per $\lambda<1$, senza ipotesi sullo score.
  - Per $\lambda=1$ serve $|s|\le B$.
  - La legge forte delle martingale fa il resto.
- [ ] **Cor. 2(i)** usa Teo. 2(ii)–(iii) con $Q=P_a$, $P=\bar P$ e frequenze delle azioni convergenti.
- [ ] **Cor. 2(ii)** è come Teo. 2(i), applicato a $e_\lambda=1-\lambda+\lambda E$. Poi $\mathrm{KL}\le\log(1+\chi^2)$ per Jensen.
- [ ] **Cor. 2(iii)**: $\bar P\ge u_aP_a$, quindi $R_a\le1/u_a$.

**Attenzione:** nessuna.

## 5. Proposizione 3: il benchmark del Bayes factor (15 min)

- [ ] $\mathbb E_P[q_\theta]=1$, quindi $Z=1$. Il prodotto dei rapporti predittivi telescopa nella miscela.
- [ ] Lo sviluppo è quello di Clarke e Barron. Il limite di $t^{-1}\log$ richiede una legge dei grandi numeri uniforme, coperta da "usual regularity conditions".

**Attenzione:** se un revisore chiede di più, la citazione da aggiungere è Clarke e Barron (1990, Theorem 1) per lo sviluppo.

## 6. Teorema 3: aggiornamento predittivo (60 min)

- [ ] **(i)** $L_t=L_{t-1}E_t$ e $\pi_{t-1}\propto L_{t-1}\pi_0$, quindi $\int L_t d\pi_0=\prod\bar E_i$. La validità viene da Teo. 1 più Fubini.
- [ ] **(ii), primo limite.**
  - $F=\log L_t$ è concava, con gradiente di norma $\le 2G$ per fattore, quindi è $2Gt$-Lipschitz.
  - L'insieme $S_r=\theta^*+\frac{r}{2D}(\Theta-\theta^*)$ ha $\pi_0(S_r)=(r/2D)^d$.
  - Con $r=d/(2Gt)$: $-d+d\log\frac{d}{4GDt} = -d\log t-d\log\frac{4eGD}{d}$.
  - La condizione $r\le2D$ equivale a $t\ge d/(4GD)$, come nell'enunciato.
- [ ] **(ii), secondo limite.**
  - L'Hessiano è $\succeq -tG^2I$ e il massimo è interno.
  - Con $r=\sqrt{d/(tG^2)}$: $-\frac d2\log t-\frac d2\log\frac{eG^2(2D)^2}{d}$.
  - Che $B_r\subset\Theta$ implica $r\le D$, e $S_r\subset B_r$.
- [ ] **(iii)** $E_i(\theta)\in[(M+1)/(1+Me^{2GB_\Theta}),M+1]$, quindi la legge forte vale per $\theta$ fisso. Un sottoinsieme numerabile denso più la continuità di $g_M$ danno il sup.

**Attenzione:** il teorema vale con prior uniforme su un compatto e posterior esatto. La regola pratica (prior gaussiano, un passo) è coperta solo dalla Prop. 4, come già detto nel testo e nella tabella delle regole. Va bene così, basta non dire di più nel testo.

## 7. Proposizione 4: posterior approssimati (15 min)

- [ ] La validità viene da Fubini. Poi Jensen. L'Hessiano $-\mathrm{Cov}_w(\phi)\succeq -G^2I$ dà lo sviluppo con resto. Il termine lineare si annulla perché $m$ è la media di $\nu$.
- [ ] Il corollario nel testo: se la precisione cresce almeno linearmente, $\sum_i\mathrm{tr}\,\Sigma_{i-1}\le\frac{d}{c_0}\log t+O(1)$.

---

## Controlli numerici già nel repository (se vuoi una conferma rapida)

| Risultato | Controllo | Dove |
|---|---|---|
| Prop. 1, Teo. 1 | $\mathbb E[E_t]=1$ per score arbitrari | `tests/test_validity.py` |
| Teo. 2 (i)–(iv) | limiti verificati in 56 configurazioni gaussiane | Figura 2, `experiments/exp4_growth_rate.py` |
| Teo. 3 (ii) | regret calcolato esattamente su griglia, 4 scenari | Supplemento S4 |
| Prop. 4 | gradiente della simulation likelihood alle differenze finite | `tests/test_choice_blr.py` |

`python -m pytest tests` (circa 30 secondi).

## Esito del mio controllo

Non ho trovato errori. L'unica aggiunta, già fatta nel repository, è la riga sulla
$\mathrm{KL}(Q\|\tilde T)$ nella dimostrazione del Teo. 2(iv).
