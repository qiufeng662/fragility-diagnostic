"""Minimal unit tests for the diagnostic machinery.

Run:  python -m pytest test_diagnostic.py   (or  python test_diagnostic.py)
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from optmask.diagnostics import _frac_knapsack, _exact_q
from optmask.certify import det_adj_precompute
from optmask.contract import Target


def _sqrtm(M):
    ev, Q = np.linalg.eigh(M)
    return (Q * np.sqrt(np.clip(ev, 0, None))) @ Q.T


def test_frac_knapsack():
    v = np.array([1.0, 0.5])
    w = np.array([1.0, 1.0])
    assert abs(_frac_knapsack(v, w, 1.0) - 1.0) < 1e-12
    assert abs(_frac_knapsack(v, w, 2.0) - 1.5) < 1e-12


def test_leverage_invariance():
    """h_g = lambda_max(P^{1/2} A_g P^{1/2}) is invariant under column rescaling
    X -> X D (Corollary: coordinate invariance)."""
    rng = np.random.default_rng(1)
    p = 5
    Ag = [rng.standard_normal((8, p)) for _ in range(6)]
    D = np.diag(np.exp(rng.standard_normal(p)))

    def hmax(Ags):
        M = sum(A.T @ A for A in Ags)
        Ps = _sqrtm(np.linalg.inv(M))
        return [float(np.linalg.eigvalsh(Ps @ (A.T @ A) @ Ps).max()) for A in Ags]

    assert np.allclose(hmax(Ag), hmax([A @ D for A in Ag]), atol=1e-9)


def test_detadj_matches_refit_fixed_design():
    """The determinant/adjugate q_j equals exact re-fitting for a fixed design
    (no fixed effects), validating Proposition 2.3."""
    import pandas as pd
    from optmask.estimators import OLS

    rng = np.random.default_rng(2)
    rows = []
    for g in range(6):
        for _ in range(10):
            x1, x2 = rng.standard_normal(), rng.standard_normal()
            rows.append((g, x1, x2, 2.0 * x1 - 1.0 * x2 + rng.standard_normal()))
    df = pd.DataFrame(rows, columns=["c", "x1", "x2", "y"])
    est = OLS("y", ["x1", "x2"], "c")
    prep = est.prepare(df)
    f = est.fit(prep, np.ones(len(df), bool))
    pre = det_adj_precompute(f)
    tgt = Target(column="x1", sign=1, alpha=0.05)

    q_da = _exact_q(pre, tgt, [0])
    keep = (df["c"] != 0).to_numpy()
    q_refit = float(est.fit(prep, keep).beta[0])
    assert abs(q_da - q_refit) < 1e-8


if __name__ == "__main__":
    test_frac_knapsack()
    test_leverage_invariance()
    test_detadj_matches_refit_fixed_design()
    print("all tests passed")
