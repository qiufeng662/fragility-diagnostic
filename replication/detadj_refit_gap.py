"""Quantify the determinant/adjugate vs exact re-fit discrepancy under fixed-effect
re-absorption (reviewer M4).

For top-k leverage cluster deletion sets (k = 1..15) on both panels, compare the
fixed-design determinant/adjugate signed deviation q_j with the exact re-fitted
value.  The gap grows with the deletion set size, which is why the paper re-fits
for the empirical evaluations.

Run:  python detadj_refit_gap.py
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
from optmask.diagnostics import _exact_q
from optmask.contract import Target

HERE = os.path.dirname(os.path.abspath(__file__))


def run(panel, regs, focal, sign):
    d = pd.read_csv(os.path.join(HERE, "data", panel))
    est = TWFE_CV("y", regs, "firm_id", "year", "firm_id")
    prep = est.prepare(d)
    complete = np.ones(len(d), bool)
    f0 = est.fit(prep, complete)
    j = f0.names.index(focal)
    target = Target(column=focal, sign=sign, alpha=0.05)
    pre = det_adj_precompute(f0)
    code = prep["cluster_code"]
    c = pre["c"]
    G = pre["G"]

    h_g = np.array([np.linalg.norm(pre["H"][g], 2) for g in range(G)])
    order = np.argsort(-h_g)          # top-leverage clusters first

    def refit_q(mask):
        keep = complete.copy()
        for g in mask:
            keep &= code != g
        f = est.fit(prep, keep)
        return float(sign * f.beta[j])

    def det_Q(mask):
        x = np.zeros(G)
        for g in mask:
            x[g] = 1.0
        Q = np.eye(pre["k"]) - np.einsum('g,gab->ab', x, pre["H"])
        return float(np.linalg.det(Q))

    q0 = refit_q([])
    print(f"\n=== {panel}: focal {focal} (sign={sign:+d}), n={len(d)}, G={G} ===")
    print(f"full-data q_refit = {q0:.4f}")
    print(f"{'k':>2} {'rows':>5} {'q_detadj':>10} {'q_refit':>10} "
          f"{'|gap|':>9} {'|gap|/|q0|':>10} {'detQ':>11}")
    for k in range(1, 16):
        mask = [int(g) for g in order[:k]]
        rows = int(c[mask].sum())
        qd = _exact_q(pre, target, mask)
        qr = refit_q(mask)
        dq = det_Q(mask)
        if qd is None:
            print(f"{k:>2} {rows:>5} {'None':>10} {qr:>10.4f}  {'--':>9} {'--':>10} {dq:>11.2e}")
            continue
        gap = abs(qd - qr)
        print(f"{k:>2} {rows:>5} {qd:>10.4f} {qr:>10.4f} {gap:>9.4f} "
              f"{gap/abs(q0):>10.3f} {dq:>11.2e}")


run("panel_firm.csv", [f"x{k}" for k in range(1, 11)], "x6", -1)
run("panel_listed.csv", [f"x{k}" for k in range(1, 8)], "x1", +1)
