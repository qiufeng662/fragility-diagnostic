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
# unified CR1 convention: rank adjustment from k_all (regressors + year dummies)
# to p_eff = k_real + (T-1) + n_entities (absorbed entity effects count in the rank)
k_all1 = prep1["X"].shape[1]
p_eff1 = len(regs1) + int(d1["year"].nunique()) - 1 + int(d1["firm_id"].nunique())
se1_u = float(f1.se[j1]) * np.sqrt((f1.n - k_all1) / (f1.n - p_eff1))
print(f"[Panel 1] unified CR1 (p_eff={p_eff1}): se={se1_u:.4f}, "
      f"t={f1.beta[j1] / se1_u:.3f}")

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


def refit_beta1(mask):
    keep = np.ones(n1, bool)
    for g in mask:
        keep &= code1 != g
    return est1.fit(prep1, keep).beta[j1]


mask1 = []
nodes1 = 0
t1 = time.perf_counter()
for _ in range(200):
    bb, bg = np.inf, refit_beta1(mask1)
    for g in cand1:
        if g in mask1:
            continue
        if float(c1[mask1 + [g]].sum()) > 1500:
            continue
        b = refit_beta1(mask1 + [g])
        nodes1 += 1
        if b < bb:
            bb, bg = b, g
    if bg is None:
        break
    mask1.append(bg)
    beta1 = refit_beta1(mask1)
    nodes1 += 1
    if beta1 < 0:
        rows1 = int(c1[mask1].sum())
        print(f"[Panel 1] sign flips after deleting {rows1} obs "
              f"({rows1/n1:.1%}), clusters={len(mask1)}, beta={beta1:.4f}, "
              f"nodes={nodes1}, wall={time.perf_counter()-t1:.0f}s")
        break

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
k_all2 = prep2["X"].shape[1]
p_eff2 = len(regs2) + int(d2["year"].nunique()) - 1 + int(d2["firm_id"].nunique())
se2_u = float(f2.se[j2]) * np.sqrt((f2.n - k_all2) / (f2.n - p_eff2))
print(f"[Panel 2] unified CR1 (p_eff={p_eff2}): se={se2_u:.4f}, "
      f"t={f2.beta[j2] / se2_u:.3f}")

# flip search: target x1 positive, real refit, find sign flip (6 obs) then sig (55 obs)
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
    n, g = f.n, f.g
    n_ent_kept = d2["firm_id"].nunique() - len(mask)
    t_fit = f.beta[j2] / f.se[j2]
    t_full = t_fit / np.sqrt((n - k_all) / (n - k_all - n_ent_kept))
    return f.beta[j2], t_full, g


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
