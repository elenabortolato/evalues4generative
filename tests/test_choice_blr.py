"""ChoiceBLR: the posterior-predictive factor is fair under the null, and the
BLR gradient is the gradient of -log E averaged over the posterior nodes."""

import numpy as np

from e4g import ChoiceBLR


def make(seed=0):
    rng = np.random.default_rng(seed)
    lr = ChoiceBLR(degree=2, n_nodes=16).fit_scaler(rng.normal(size=(500, 2)))
    lr.m = rng.normal(size=lr.m.shape)            # an arbitrary, fixed posterior
    lr.prec = lr.prec * 4.0
    return lr, rng


def test_factor_fair_under_null():
    lr, rng = make()
    m = 8
    vals = [lr.factor(rng.normal(size=(1, 2)), rng.normal(size=(m, 2))) for _ in range(40000)]
    assert abs(np.mean(vals) - 1.0) < 0.02


def test_gradient_matches_finite_differences():
    lr, rng = make(1)
    xr, xs = rng.normal(size=(1, 2)), rng.normal(size=(8, 2))
    z = lr._z(np.vstack([xr, xs]))
    theta = lr._nodes()

    def loss(shift):
        le, _ = lr._log_e(theta + shift[None, :], z)
        return -le.mean()

    eps, d = 1e-5, len(lr.m)
    fd = np.array([(loss(eps * np.eye(d)[i]) - loss(-eps * np.eye(d)[i])) / (2 * eps)
                   for i in range(d)])
    m0, p0 = lr.m.copy(), lr.prec.copy()
    lr.update(xr, xs)
    grad = lr.prec @ (m0 - lr.m)                  # m_new = m0 - prec_new^{-1} grad
    assert np.allclose(grad, fd, atol=1e-6)
    assert np.all(np.linalg.eigvalsh(lr.prec - p0) >= -1e-10)   # Hessian term is PSD
