"""Full diagnostic (Algorithm 1 + pruned search) on the listed-firm panel.

Reproduces the Section 8 numbers for the second panel and adds the diagnostic
outputs the paper does not yet report: the leverage profile h_g, the box
certificate status, the first-order distortion factors, the search trajectory,
and an informal AMIP (first-order influence) comparison on the same data.

Run:  python diag_panel2.py
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import scipy.linalg as sla
from scipy import stats

np.linalg.lstsq = lambda a, b, rcond=None: sla.lstsq(
    a, b, cond=(rcond if rcond is not None else None))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from optmask.cluster_valid import TWFE_CV
from optmask.certify import det_adj_precompute
from optmask.diagnostics import _exact_q, _frac_knapsack
from optmask.contract import Target

HERE = os.path.dirname(os.path.abspath(__file__))

d = pd.read_csv(os.path.join(HERE, "data", "panel_listed.csv"))
regs = [f"x{k}" for k in range(1, 8)]
est = TWFE_CV("y", regs, "firm_id", "year", "firm_id")
prep = est.prepare(d)
complete = np.ones(len(d), bool)
f0 = est.fit(prep, complete)
j = f0.names.index("x1")
target = Target(column="x1", sign=1, alpha=0.05)
pre = det_adj_precompute(f0)
code = prep["cluster_code"]
c = pre["c"]
G = pre["G"]
n = len(d)
k_all = prep["X"].shape[1]          # regressors + year dummies

print("=" * 78)
print("LISTED-FIRM PANEL  --  full diagnostic (focal x1, sign=+1, alpha=0.05)")
print("=" * 78)
print(f"n={n}, G={G}, k_real={prep['k_real']}, k_all={k_all}")
print(f"full-data beta(x1)={f0.beta[j]:.4f}, se={f0.se[j]:.4f}, t={f0.beta[j]/f0.se[j]:.3f}")

# --- Algorithm 1, step 1: leverage profile ---------------------------------
h_g = np.array([np.linalg.norm(pre["H"][g], 2) for g in range(G)])
order = np.argsort(-h_g)
print("\n[Alg1.1] top-10 relative leverage h_g (cluster size in rows):")
for r in range(10):
    g = int(order[r])
    print(f"  g={g:3d}  h_g={h_g[g]:.3f}  rows={int(c[g]):3d}")

# --- Algorithm 1, step 2: box certificate ----------------------------------
K_scan = 120
eta = np.array([_frac_knapsack(h_g, c, float(K)) for K in range(1, K_scan + 1)])
fail = np.flatnonzero(eta >= 1.0)
box_fail_K = int(fail[0] + 1) if len(fail) else None
print("\n[Alg1.2] box certificate:")
print(f"  eta_K crosses 1 at K={box_fail_K} rows" if box_fail_K else
      f"  eta_K < 1 for all K<= {K_scan} (certificate available)")
print(f"  eta_K at K=6: {eta[5]:.3f}, K=10: {eta[9]:.3f}, K=55: {eta[54]:.3f}")

# --- Algorithm 1, steps 3-4: per-cluster exact margin + distortion ----------
print("\n[Alg1.3/1.4] dangerous clusters (h_g>=0.5): exact q_j and 1/(1-h):")
for g in range(G):
    hg = h_g[g]
    if hg >= 0.5:
        q = _exact_q(pre, target, [g])
        fo = 1.0 / max(1.0 - hg, 1e-12)
        print(f"  g={g:3d}  h_g={hg:.3f}  q_exact={q:.4f}  first-order factor={fo:.1f}x")


def refit(mask):
    keep = complete.copy()
    for g in mask:
        keep &= code != g
    f = est.fit(prep, keep)
    return f.beta[j], f.beta[j] / f.se[j], f.g


def greedy_search(candidates, label):
    """Add clusters one at a time (best single-step improvement), exact refit."""
    mask, flip_at, sig_at, traj = [], None, None, []
    t0 = time.perf_counter()
    nodes = 0
    for _ in range(200):
        bb, bg = -np.inf, refit(mask)[0]
        for g in candidates:
            if g in mask:
                continue
            if float(c[mask + [g]].sum()) > 600:
                continue
            b = refit(mask + [g])[0]
            nodes += 1
            if b > bb:
                bb, bg = b, g
        if bg is None:
            break
        mask.append(bg)
        beta, t, gnum = refit(mask)
        nodes += 1
        rows = int(c[mask].sum())
        traj.append((rows, len(mask), beta, t))
        crit = stats.t.ppf(1 - 0.05 / 2, gnum - 1)
        if flip_at is None and beta > 0:
            flip_at = rows
            print(f"  [{label}] sign flips after {rows} obs ({rows/n:.1%}), "
                  f"clusters={len(mask)}, beta={beta:.4f}, t={t:.3f}")
        if flip_at is not None and sig_at is None and abs(t) > crit:
            sig_at = rows
            print(f"  [{label}] significant after {rows} obs ({rows/n:.1%}), "
                  f"clusters={len(mask)}, beta={beta:.4f}, t={t:.3f} (crit={crit:.3f})")
            break
    dt = time.perf_counter() - t0
    return flip_at, sig_at, nodes, dt, mask, traj


# --- pruned search (Algorithm 2) --------------------------------------------
# candidate ranking: single-cluster exact q_j (det/adjugate), as in the paper
qs = np.array([(_exact_q(pre, target, [g]) if _exact_q(pre, target, [g]) is not None
                else -1e9) for g in range(G)])
cand_exact = np.argsort(-qs)[:150].tolist()
print("\n[Alg2] pruned search (candidates = top-150 by single-cluster exact q_j):")
flip_e, sig_e, nodes_e, dt_e, mask_e, traj_e = greedy_search(cand_exact, "exact")

# --- AMIP baseline: first-order influence ranking ---------------------------
# first-order single-cluster gain on q_j is -s_j * a_tilde^T psi_tilde_g;
# for sign=+1 the most flip-ward clusters have the most negative
# a_tilde^T psi_tilde_g.  AMIP ranks by (signed) first-order influence.
wv = np.zeros(pre["k"])
wv[j] = 1.0
a_tilde = pre["Ps"] @ wv
fo_score = a_tilde @ pre["psi_tilde"].T          # length G, = a_tilde^T psi_tilde_g
cand_amip = np.argsort(fo_score)[:150].tolist()   # most negative first (flip-ward)
print("\n[AMIP] first-order influence ranking (signed, most flip-ward first):")
flip_a, sig_a, nodes_a, dt_a, mask_a, traj_a = greedy_search(cand_amip, "AMIP")

print("\n[trajectory] pruned-search path (rows, clusters, beta, t):")
for rows, ncl, beta, t in traj_e:
    print(f"  rows={rows:3d}  clusters={ncl:2d}  beta={beta:+.4f}  t={t:+.3f}")

print("\n" + "=" * 78)
print("SUMMARY")
print("=" * 78)
print(f"pruned (exact q_j):  flip@ {flip_e} obs, significance@ {sig_e} obs, "
      f"nodes={nodes_e}, wall={dt_e:.1f}s")
print(f"AMIP (first-order):  flip@ {flip_a} obs, significance@ {sig_a} obs, "
      f"nodes={nodes_a}, wall={dt_a:.1f}s")
