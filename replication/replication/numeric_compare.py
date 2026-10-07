"""3.1: numerical comparison — determinant/adjugate vs Cholesky vs LU vs QR.

Ground truth uses a QR decomposition (independent of LU/Cholesky), so each method's
error is measured against a different algorithm. Scans the identification margin
delta -> 0 (near-singular Q), the regime the paper cares about most.

Two regimes are distinguished:
  (a) legitimate deletion matrices Q = I - H(x) with all eigenvalues in (0, 1],
      which is what the determinant/adjugate evaluator actually faces;
  (b) a general SPD stress test (secondary eigenvalues allowed above 1), kept as
      a broader conditioning check and reported separately.
"""
import os
import numpy as np
from scipy import linalg

rng = np.random.default_rng(0)
p = 12
deltas = [1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8, 1e-10]
n_trials = 200


def build_Q(delta, legitimate=True):
    if legitimate:
        # 0 < Q <= I with lambda_min(Q)=delta and lambda_max(Q)=1 exactly
        # at the level of the constructed spectrum, hence kappa_2(Q)=1/delta.
        if not 0.0 < delta <= 1.0:
            raise ValueError("legitimate deletion matrices require 0 < delta <= 1")
        ev = np.concatenate(
            [[delta], rng.uniform(delta, 1.0, p - 2), [1.0]]
        )
    else:
        # general SPD: secondary eigenvalues may exceed 1.
        ev = np.concatenate([[delta], rng.uniform(0.5, 2.0, p - 1)])
    A = rng.normal(size=(p, p))
    Qorth, _ = np.linalg.qr(A)
    return Qorth @ np.diag(ev) @ Qorth.T


def run(delta, legitimate):
    err_da, err_ch, err_lu = [], [], []
    conds = []
    fail_ch = 0
    for _ in range(n_trials):
        Q = build_Q(delta, legitimate)
        conds.append(float(np.linalg.cond(Q, 2)))
        r = rng.normal(size=p)
        a = rng.normal(size=p)
        Qr, Rr = np.linalg.qr(Q)
        q_true = float(a @ linalg.solve_triangular(Rr, Qr.T @ r))
        d = float(np.linalg.det(Q))
        adj = d * np.linalg.inv(Q)
        q_da = float(a @ adj @ r) / d
        err_da.append(abs(q_da - q_true) / max(abs(q_true), 1e-300))
        try:
            L = np.linalg.cholesky(Q)
            q_ch = float(a @ linalg.cho_solve((L, True), r))
            err_ch.append(abs(q_ch - q_true) / max(abs(q_true), 1e-300))
        except Exception:
            fail_ch += 1
        q_lu = float(a @ np.linalg.solve(Q, r))
        err_lu.append(abs(q_lu - q_true) / max(abs(q_true), 1e-300))
    med = lambda x: float(np.median(x)) if x else float("inf")
    return med(conds), med(err_da), med(err_ch), med(err_lu), fail_ch


for legit, label in [(True, "legitimate Q (0<Q<=I)"), (False, "general SPD")]:
    print(f"\n=== {label} ===")
    print(f"{'delta':>8} {'median kappa':>13} | "
          f"{'det/adj err':>11} {'chol err':>11} {'lu err':>11} | "
          f"{'chol fail':>9}")
    for delta in deltas:
        cond, eda, ech, elu, fail = run(delta, legit)
        print(f"{delta:8.0e} {cond:13.3e} | "
              f"{eda:11.2e} {ech:11.2e} {elu:11.2e} | {fail:9d}")
