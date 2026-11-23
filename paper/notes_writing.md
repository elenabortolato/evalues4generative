# Writing notes (from the read-through of the Introduction, 23 Nov 2026)

## Length
- The current Introduction is about 5 journal pages; aim for 2.5–3.5.
- The Bayes-factor material (three consequences, coin, Clarke–Barron) is a
  small tutorial: keep 3 sentences in the Introduction, move the rest to the
  start of Section 4.1.
- Halve the attribution paragraph; the detailed list belongs in Section 6
  (related work).

## Introducing new concepts
Explain in plain words *before* using the technical name, at first use:
- **tilted null**: "the simulator reweighted by the score", the law the score
  implicitly bets reality looks like.
- **conditional logit**: one item chosen from a set with probability
  proportional to its score; the set is the real outcome plus its M
  simulations, the chosen item is the real one.
- **prequential**: forecasts issued before each observation, scored by log
  loss; the Bayes factor rewards the better forecaster.
- **regret**: the gap between our wealth and that of the best score chosen in
  hindsight.
- **e-process / Ville**: the running wealth of fair bets; it exceeds 1/alpha
  with probability at most alpha, whenever we stop.

## Style (make it less recognisably machine-written)
- Fewer colons used as "X: because Y" hinges; use two sentences instead.
- Fewer lists of three; vary sentence length; allow short sentences.
- Avoid recurring formulas: "whatever the score", "at a bounded cost",
  "the one that…", "this is what…".
- Start from a concrete case (the insulin trial) before the abstraction.
- Rewrite in your own voice at least: the opening, the sentence on why the
  problem matters to you, and the closing of the contributions.

## Proposed alternative
- `paper/intro_v2.tex`: shorter Introduction (Part A) + Bayes-factor block
  for Section 4.1 (Part B). Not yet input in `main.tex`.
