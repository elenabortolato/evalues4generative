"""Sequential monitor running the competing e-processes side by side.

At each step the caller supplies one real observation and M simulator draws for
the same context (the same action, in an adaptive design). The monitor

* scores them with a ratio estimator fitted on the past only,
* updates the Monte Carlo e-process (exactly valid, Proposition 2),
* updates the naive plug-in process (learned ratio used as the bet),
* optionally updates the oracle likelihood-ratio process,
* and refits the estimator every ``refit_every`` steps.

Inputs are "rows": whatever the learner's feature map expects, typically the
observation concatenated with a one-hot encoding of the action.
"""

from __future__ import annotations

import numpy as np

from .evalues import LambdaMixture, ProductProcess, mc_factor, plugin_factor
from .learners import OnlineRatio


class Monitor:
    def __init__(self, learner: OnlineRatio, alpha: float = 0.05, burn_in: int = 20,
                 refit_every: int = 10, max_sim_pool: int = 4000):
        self.learner = learner
        self.threshold = 1.0 / alpha
        self.burn_in = burn_in
        self.refit_every = refit_every
        self.max_sim_pool = max_sim_pool
        self.real, self.sims = [], []
        self.proc = {"mc": LambdaMixture(), "plugin": LambdaMixture(),
                     "oracle": ProductProcess()}
        self.first_cross = {k: None for k in self.proc}
        self.t = 0

    def step(self, row_real: np.ndarray, rows_sim: np.ndarray, oracle_lr: float | None = None):
        self.t += 1
        s = self.learner.score(np.vstack([row_real[None, :], rows_sim]))
        e = {"mc": mc_factor(s[0], s[1:]), "plugin": plugin_factor(s[0])}
        if oracle_lr is not None:
            e["oracle"] = oracle_lr
        for k, v in e.items():
            val = self.proc[k].update(v)
            if self.first_cross[k] is None and val >= self.threshold:
                self.first_cross[k] = self.t
        # the new data become part of the past only now
        self.real.append(row_real)
        self.sims.extend(rows_sim)
        if self.t >= self.burn_in and self.t % self.refit_every == 0:
            sims = np.asarray(self.sims[-self.max_sim_pool:])
            self.learner.fit(np.asarray(self.real), sims)
        return e
