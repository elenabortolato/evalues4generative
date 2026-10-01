"""MartingalePredictive: the predictive update is a martingale under the
predictive (coherence), and both bets are fair under the null."""

import numpy as np
import pytest

from e4g import MartingalePredictive


@pytest.mark.parametrize("loss", ["logistic", "choice"])
def test_update_is_a_martingale_under_the_predictive(loss):
    rng = np.random.default_rng(0)
    lr = MartingalePredictive(loss=loss, degree=2).fit_scaler(rng.normal(size=(500, 2)))
    lr.theta = rng.normal(size=lr.theta.shape) * 0.5
    z = lr._design(rng.normal(size=(9, 2)))
    theta = np.repeat(lr.theta[None, :], 200000, 0)
    eta = theta @ z.T
    if loss == "logistic":
        p = 1 / (1 + np.exp(-eta))
        y = (rng.random(p.shape) < p).astype(float)
    else:
        e = np.exp(eta - eta.max(1, keepdims=True))
        c = np.cumsum(e / e.sum(1, keepdims=True), 1)
        y = (rng.random((len(theta), 1)) > c).sum(1)
    s, J = lr._score_info(theta, z, y)
    step = np.linalg.solve(lr.info + J[0], s.mean(0))
    assert np.abs(step).max() < 3e-3


@pytest.mark.parametrize("mode", ["plugin", "resample"])
def test_bets_fair_under_null(mode):
    rng = np.random.default_rng(1)
    lr = MartingalePredictive(loss="logistic", mode=mode, degree=2, n_paths=8, h_max=20,
                              seed=1).fit_scaler(rng.normal(size=(500, 2)))
    for _ in range(30):                           # give it a past and a nonzero state
        lr.update(rng.normal(0.5, 1, size=(1, 2)), rng.normal(size=(8, 2)))
    lr.refresh = 10 ** 9                          # freeze the paths: a fixed predictable bet
    lr._resample() if mode == "resample" else None
    vals = [lr.factor(rng.normal(size=(1, 2)), rng.normal(size=(8, 2))) for _ in range(30000)]
    assert abs(np.mean(vals) - 1.0) < 0.03
