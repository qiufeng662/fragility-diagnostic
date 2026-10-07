"""Verification harness for the reviewer-reported bugs (items 1-5 of the recompute).

Checks, per panel:
  [1] || sum_g H_g - I ||_2   (must be < 1e-10 once Fit returns the full design)
  [2] max single-cluster relative leverage h_g
  [3] box-certificate failure point K (smallest K with eta_K >= 1)
  [4] firm panel: top-25 leverage deletion set -> rows, min eigenvalue of Q(x)
  [5] fig3 cumulative positive gain: paper first-order term vs the old resolvent form
  [6] det/adjugate q_j vs exact re-fit q_j gap for top-k leverage deletion sets
  [7] baseline beta/se/t under the unified CR1 convention with rank p_eff

Run:  python verify_fixes.py
"""
import os
import sys
import numpy as np
import pandas as pd
import scipy.linalg as sla

np.linalg.lstsq = lambda a, b, rcond=None: sla.lstsq(
    a, b, cond=(rcond if rcond is not None else None))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from optmask.cluster_valid import TWFE_CV
from optmask.certify import det_adj_precompute
from optmask.diagnostics import _exact_q, _frac_knapsack
from optmask.contract import Target

HERE = os.path.dirname(os.path.abspath(__file__))


def load(panel, n_regs):
    d = pd.read_csv(os.path.join(HERE, "data", panel))
    regs = [f"x{k}" for k in range(1, n_regs + 1)]
    est = TWFE_CV("y", regs, "firm_id", "year", "firm_id")
    prep = est.prepare(d)
    f0 = est.fit(prep, np.ones(len(d), bool))
    return d, est, prep, f0


def q_gap_table(est, prep, f0, pre, focal, sign, ks):
    j = f0.names.index(focal)
    target = Target(column=focal, sign=sign, alpha=0.05)
    code = prep["cluster_code"]
    complete = np.ones(len(prep["y"]), bool)
    h_g = np.array([np.linalg.norm(pre["H"][g], 2) for g in range(pre["G"])])
    order = np.argsort(-h_g)
    c = pre["c"]

    def refit_q(mask):
        keep = complete.copy()
        for g in mask:
            keep &= code != g
        return float(sign * est.fit(prep, keep).beta[j])

    print(f"[6] q_detadj vs q_refit (top-k leverage clusters), focal {focal}:")
    print(f"{'k':>3} {'rows':>5} {'q_detadj':>12} {'q_refit':>12} {'|gap|':>10} "
          f"{'min eig Q(x)':>13}")
    for kk in ks:
        mask = [int(g) for g in order[:kk]]
        rows = int(c[mask].sum())
        qd = _exact_q(pre, target, mask)
        qr = refit_q(mask)
        x = np.zeros(pre["G"])
        for g in mask:
            x[g] = 1.0
        Q = np.eye(pre["k"]) - np.einsum('g,gab->ab', x, pre["H"])
        mineig = float(np.linalg.eigvalsh(Q).min())
        if qd is None:
            print(f"{kk:>3} {rows:>5} {'None':>12} {qr:>12.6f} {'--':>10} {mineig:>13.6e}")
        else:
            print(f"{kk:>3} {rows:>5} {qd:>12.6f} {qr:>12.6f} {abs(qd - qr):>10.2e} "
                  f"{mineig:>13.6e}")


def report(panel, n_regs, focal, sign, qgap_ks, do_fig3=False):
    d, est, prep, f0 = load(panel, n_regs)
    n = len(d)
    pre = det_adj_precompute(f0)
    H, c, G, k = pre["H"], pre["c"], pre["G"], pre["k"]
    code = prep["cluster_code"]

    print("=" * 72)
    print(f"=== {panel}  (focal {focal}, sign={sign:+d}) ===")
    print("=" * 72)

    n_years = int(d["year"].nunique())
    n_ent = int(d["firm_id"].nunique())
    p_eff = n_regs + (n_years - 1) + n_ent
    k_all = prep["X"].shape[1]
    print(f"n={n}, G={G}, k_real={n_regs}, k_all={k_all}, years={n_years}, "
          f"entities={n_ent}, p_eff={p_eff}")
    print(f"year range: {d['year'].min():.0f}..{d['year'].max():.0f}")

    sumH = np.einsum('gab->ab', H)
    print(f"[1] ||sum_g H_g - I_k||_2 = {np.linalg.norm(sumH - np.eye(k), 2):.3e}  (k={k})")

    h_g = np.array([np.linalg.norm(H[g], 2) for g in range(G)])
    print(f"[2] max h_g = {h_g.max():.6f} (cluster {int(h_g.argmax())}); "
          f"top-3 = {np.round(np.sort(h_g)[-1:-4:-1], 4).tolist()}")

    fail_K = None
    for K in range(1, n + 1):
        if _frac_knapsack(h_g, c, float(K)) >= 1.0:
            fail_K = K
            break
    print(f"[3] eta_K >= 1 first at K = {fail_K} rows"
          if fail_K is not None else f"[3] eta_K < 1 for all K <= {n}")

    j = f0.names.index(focal)
    se0, t0 = float(f0.se[j]), float(f0.beta[j] / f0.se[j])
    se_peff = se0 * np.sqrt((n - k_all) / (n - p_eff))
    print(f"[7] baseline beta={f0.beta[j]:.6f}, se(CR1,k_all)={se0:.6f}, t={t0:.4f}")
    print(f"    unified CR1(p_eff={p_eff}): se={se_peff:.6f}, "
          f"t={float(f0.beta[j] / se_peff):.4f}")

    if do_fig3:
        X, u, P = f0.X, f0.resid, f0.xtx_inv
        db1 = np.zeros(G)
        dbres = np.zeros(G)
        for g in range(G):
            m = code == g
            Xg, ug = X[m], u[m]
            db1[g] = (P @ Xg.T @ ug)[j]
            wg = np.linalg.solve(np.eye(Xg.shape[0]) - Xg @ P @ Xg.T, ug)
            dbres[g] = (P @ Xg.T @ wg)[j]
        print(f"[5] fig3 cumulative positive gain: first-order = {db1[db1 > 0].sum():.6f}, "
              f"resolvent form (bug 2) = {dbres[dbres > 0].sum():.6f}, "
              f"theta0 = {f0.beta[j]:.6f}")

    q_gap_table(est, prep, f0, pre, focal, sign, qgap_ks)
    print()
    return d, est, prep, f0, pre


report("panel_firm.csv", 10, "x6", -1, [1, 2, 5, 10, 15, 25], do_fig3=True)
report("panel_listed.csv", 7, "x1", +1, [1, 2, 5, 9, 15])
print("verify_fixes done.")
