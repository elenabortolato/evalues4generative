"""Exactness checks for the e-value constructions (run: python -m pytest tests)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from e4g import LambdaMixture, OnlineRatio, mc_factor  # noqa: E402


def _mc_mean(score, draw, m, n=40000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        z = draw(rng, m + 1)
        s = score(z)
        vals.append(mc_factor(s[0], s[1:]))
    return np.mean(vals), np.std(vals) / np.sqrt(n)


def test_mc_factor_mean_one_for_arbitrary_score():
    # a deliberately wild, badly "miscalibrated" score: validity must not care
    score = lambda z: 5 * np.sin(3 * z[:, 0]) + z[:, 0] ** 3
    draw = lambda rng, n: rng.standard_normal((n, 1))
    for m in (1, 4, 32):
        mean, se = _mc_mean(score, draw, m)
        assert abs(mean - 1) < 4 * se + 1e-3, (m, mean, se)


def test_mc_factor_handles_degenerate_scores():
    # all scores equal -> E = 1 exactly
    assert mc_factor(0.0, np.zeros(9)) == 1.0
    # huge scores do not overflow
    assert np.isfinite(mc_factor(700.0, np.full(5, -700.0)))


def test_plugin_ratio_is_not_calibrated():
    # a learned ratio fitted on null-vs-null data has null mean > 1 when it overfits
    rng = np.random.default_rng(1)
    r = OnlineRatio(degree=6, ridge=1e-6).fit(rng.standard_normal((60, 2)),
                                              rng.standard_normal((600, 2)))
    x = rng.standard_normal((200000, 2))
    assert np.exp(r.score(x)).mean() > 1.05


def test_lambda_mixture_is_martingale_on_average():
    rng = np.random.default_rng(2)
    finals = []
    for _ in range(4000):
        mix = LambdaMixture()
        for _ in range(20):
            s = rng.standard_normal(9)
            mix.update(mc_factor(s[0], s[1:]))
        finals.append(mix.value)
    finals = np.array(finals)
    assert abs(finals.mean() - 1) < 4 * finals.std() / np.sqrt(len(finals)) + 1e-2


def test_blr_agrees_with_batch_fit_and_shrinks_predictive():
    from e4g import BLRRatio
    rng = np.random.default_rng(3)
    real = rng.normal([0.7, -0.4], 1.0, (1500, 2))
    sims = rng.standard_normal((1500 * 8, 2))
    blr = BLRRatio(degree=2).fit_scaler(rng.standard_normal((4000, 2)))
    for i in range(1500):
        blr.update(real[i : i + 1], sims[8 * i : 8 * (i + 1)])
    batch = OnlineRatio(degree=2, ridge=1.0).fit(real, sims)
    x = rng.standard_normal((2000, 2))
    blr.mode = "mean"
    s_mean = blr.score(x)
    assert np.corrcoef(s_mean, batch.score(x))[0, 1] > 0.98
    blr.mode = "predictive"
    s_pred = blr.score(x)
    # predictive log-odds are the mean log-odds shrunk towards 0
    assert np.all(np.abs(s_pred) <= np.abs(s_mean) + 1e-12)
