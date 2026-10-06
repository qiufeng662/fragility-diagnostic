"""3.1: numerical comparison — determinant/adjugate vs Cholesky vs LU vs QR.

Ground truth uses a QR decomposition (independent of LU/Cholesky), so each method's
error is measured against a different algorithm. Scans the identification margin
delta -> 0 (near-singular Q), the regime the paper cares about most.
"""
import os
import numpy as np
from scipy import linalg

rng = np.random.default_rng(0)
p = 12
deltas = [1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8, 1e-10]
n_trials = 200


def build_Q(delta):
    ev = np.concatenate([[delta], rng.uniform(0.5, 2.0, p - 1)])
    A = rng.normal(size=(p, p))
    Qorth, _ = np.linalg.qr(A)
    return Qorth @ np.diag(ev) @ Qorth.T


print(f"{'delta':>8} | {'det/adj err':>11} {'chol err':>11} {'lu err':>11} | "
      f"{'chol fail':>9}")
for delta in deltas:
    err_da, err_ch, err_lu = [], [], []
    fail_ch = 0
    for _ in range(n_trials):
        Q = build_Q(delta)
        r = rng.normal(size=p)
        a = rng.normal(size=p)
        # ground truth: QR solve (independent of LU and Cholesky)
        Qr, Rr = np.linalg.qr(Q)
        q_true = float(a @ linalg.solve_triangular(Rr, Qr.T @ r))
        # 1. determinant/adjugate (paper's implementation: d * inv)
        d = float(np.linalg.det(Q))
        adj = d * np.linalg.inv(Q)
        q_da = float(a @ adj @ r) / d
        err_da.append(abs(q_da - q_true) / max(abs(q_true), 1e-300))
        # 2. Cholesky
        try:
            L = np.linalg.cholesky(Q)
            q_ch = float(a @ linalg.cho_solve((L, True), r))
            err_ch.append(abs(q_ch - q_true) / max(abs(q_true), 1e-300))
        except Exception:
            fail_ch += 1
        # 3. LU solve
        q_lu = float(a @ np.linalg.solve(Q, r))
        err_lu.append(abs(q_lu - q_true) / max(abs(q_true), 1e-300))

    med = lambda x: float(np.median(x)) if x else float("inf")
    print(f"{delta:8.0e} | {med(err_da):11.2e} {med(err_ch):11.2e} "
          f"{med(err_lu):11.2e} | {fail_ch:9d}")
