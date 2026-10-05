"""ForestMP: Bayesian-bootstrap trees; both bets are fair under the null."""

import numpy as np
import pytest

from e4g import ForestMP


@pytest.mark.parametrize("mode", ["mixture", "forest"])
def test_forest_bets_fair_under_null(mode):
    rng = np.random.default_rng(0)
    lr = ForestMP(n_trees=8, mode=mode, refit=10, seed=0)
    for _ in range(40):                       # past data, then frozen trees
        lr.update(rng.normal(0.7, 1, size=(1, 2)), rng.normal(size=(8, 2)))
    assert lr.trees is not None
    vals = [lr.factor(rng.normal(size=(1, 2)), rng.normal(size=(8, 2))) for _ in range(20000)]
    assert abs(np.mean(vals) - 1.0) < 0.03
