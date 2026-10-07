"""Reproduce the empirical numbers of Section 8 from the anonymized panels.

Run:  python reproduce.py
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
from optmask.certify import det_adj_precompute, det_adj_margin
from optmask.diagnostics import _exact_q
from optmask.contract import Target

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------- Panel 1: firm-level (robust conclusion) ----------------
d1 = pd.read_csv(os.path.join(HERE, "data", "panel_firm.csv"))
regs1 = [f"x{k}" for k in range(1, 11)]
est1 = TWFE_CV("y", regs1, "firm_id", "year", "firm_id")
prep1 = est1.prepare(d1)
f1 = est1.fit(prep1, np.ones(len(d1), bool))
j1 = f1.names.index("x6")
t1 = f1.beta[j1] / f1.se[j1]
print(f"[Panel 1] n={f1.n}, G={f1.g}, beta(x6)={f1.beta[j1]:.4f}, "
      f"se={f1.se[j1]:.4f}, t={t1:.3f}")
# fit.se already uses the full-design-rank CR1 convention (p_eff).

pre1 = det_adj_precompute(f1)
h_g = np.array([np.linalg.norm(pre1["H"][g], 2) for g in range(pre1["G"])])
print(f"[Panel 1] top leverage: {sorted(np.round(h_g, 3), reverse=True)[:3]}")

# Panel 1 flip search: focal x6 (sign=-1), minimize beta until beta < 0.
target1 = Target(column="x6", sign=-1, alpha=0.05)
code1 = prep1["cluster_code"]
c1 = pre1["c"]
n1 = len(d1)
qs1 = np.array([(_exact_q(pre1, target1, [g]) if _exact_q(pre1, target1, [g]) is not None else -1e9)
                for g in range(pre1["G"])])
cand1 = np.argsort(-qs1)[:120].tolist()


def refit1(mask):
    keep = np.ones(n1, bool)
    for g in mask:
        keep &= code1 != g
    f = est1.fit(prep1, keep)
    return f.beta[j1], f.beta[j1] / f.se[j1], f.g


mask1 = []
nodes1 = 0
t1 = time.perf_counter()
flip_rows = None
reverse_rows = None
for _ in range(200):
    bb, bg = np.inf, None
    for g in cand1:
        if g in mask1:
            continue
        if float(c1[mask1 + [g]].sum()) > 1500:
            continue
        b = refit1(mask1 + [g])[0]
        nodes1 += 1
        if b < bb:
            bb, bg = b, g
    if bg is None:
        break
    mask1.append(bg)
    beta1, t1k, gk = refit1(mask1)
    nodes1 += 1
    rows1 = int(c1[mask1].sum())
    if flip_rows is None and beta1 < 0:
        flip_rows = rows1
        print(f"[Panel 1] sign flips after deleting {rows1} obs "
              f"({rows1/n1:.1%}), clusters={len(mask1)}, beta={beta1:.4f}")
    if flip_rows is not None:
        crit = stats.t.ppf(1 - 0.05 / 2, gk - 1)
        if abs(t1k) > crit:
            reverse_rows = rows1
            print(f"[Panel 1] reverse-significant after deleting {rows1} obs, "
                  f"clusters={len(mask1)}, t={t1k:.3f} (crit={crit:.3f})")
            break

if reverse_rows is None:
    print(f"[Panel 1] no reverse-significant deletion along the examined path "
          f"up to {rows1} obs ({len(mask1)} clusters), nodes={nodes1}, "
          f"wall={time.perf_counter()-t1:.0f}s")

# ---------------- Panel 2: listed firms (fragile conclusion) ----------------
d2 = pd.read_csv(os.path.join(HERE, "data", "panel_listed.csv"))
regs2 = [f"x{k}" for k in range(1, 8)]
est2 = TWFE_CV("y", regs2, "firm_id", "year", "firm_id")
prep2 = est2.prepare(d2)
f2 = est2.fit(prep2, np.ones(len(d2), bool))
j2 = f2.names.index("x1")
t2 = f2.beta[j2] / f2.se[j2]
print(f"[Panel 2] n={f2.n}, G={f2.g}, beta(x1)={f2.beta[j2]:.4f}, "
      f"se={f2.se[j2]:.4f}, t={t2:.3f}")
# fit.se already uses the full-design-rank CR1 convention (p_eff).

# flip search: target x1 positive, real refit, find sign flip (6 obs) then sig
pre2 = det_adj_precompute(f2)
target = Target(column="x1", sign=1, alpha=0.05)
code2 = prep2["cluster_code"]
c2 = pre2["c"]
n2 = len(d2)
k_all = prep2["X"].shape[1]           # regressors + year dummies


def stat(mask):
    keep = np.ones(n2, bool)
    for g in mask:
        keep &= code2 != g
    f = est2.fit(prep2, keep)
    return f.beta[j2], f.beta[j2] / f.se[j2], f.g


qs = np.array([(_exact_q(pre2, target, [g]) if _exact_q(pre2, target, [g]) is not None else -1e9)
               for g in range(pre2["G"])])
cand = np.argsort(-qs)[:150].tolist()
mask, flip_at, sig_at = [], None, None
for _ in range(120):
    bg, bb = None, stat(mask)[0]
    for g in cand:
        if g in mask:
            continue
        if float(c2[mask + [g]].sum()) > 600:
            continue
        b = stat(mask + [g])[0]
        if b > bb:
            bb, bg = b, g
    if bg is None:
        break
    mask.append(bg)
    beta, t, g = stat(mask)
    crit = stats.t.ppf(1 - 0.05 / 2, g - 1)
    if flip_at is None and beta > 0:
        flip_at = int(c2[mask].sum())
        print(f"[Panel 2] sign flips after deleting {flip_at} obs ({flip_at/n2:.1%})")
    if flip_at is not None and abs(t) > crit:
        sig_at = int(c2[mask].sum())
        print(f"[Panel 2] significant after deleting {sig_at} obs ({sig_at/n2:.1%}), "
              f"t={t:.3f} (crit={crit:.3f})")
        break

print("done.")
