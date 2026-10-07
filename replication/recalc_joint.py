"""Recalc the joint leverage of the listed-firm significance set under the
corrected (full-design) code, and cross-check the empirical numbers."""
import os, sys
import numpy as np, pandas as pd, scipy.linalg as sla
np.linalg.lstsq = lambda a,b,rcond=None: sla.lstsq(a,b,cond=(rcond if rcond is not None else None))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from optmask.cluster_valid import TWFE_CV
from optmask.certify import det_adj_precompute
from optmask.diagnostics import _exact_q
from optmask.contract import Target
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
d = pd.read_csv(os.path.join(HERE, "data", "panel_listed.csv"))
regs = [f"x{k}" for k in range(1, 8)]
est = TWFE_CV("y", regs, "firm_id", "year", "firm_id")
prep = est.prepare(d)
complete = np.ones(len(d), bool)
f0 = est.fit(prep, complete)
j = f0.names.index("x1")
pre = det_adj_precompute(f0)
code = prep["cluster_code"]; c = pre["c"]; G = pre["G"]; n = len(d); k_all = prep["X"].shape[1]
h_g = np.array([np.linalg.norm(pre["H"][g], 2) for g in range(G)])
target = Target(column="x1", sign=1, alpha=0.05)

# verify sum H_g = I
S = sum(pre["H"][g] for g in range(G))
print(f"||sum H_g - I|| = {np.linalg.norm(S - np.eye(pre['k']), 2):.3e}")
print(f"max single h_g = {h_g.max():.4f} (cluster {int(h_g.argmax())})")

def refit(mask):
    keep = complete.copy()
    for g in mask:
        keep &= code != g
    f = est.fit(prep, keep)
    return f.beta[j], f

# greedy search toward positive beta, significance by real t critical value
qs = np.array([(_exact_q(pre, target, [g]) if _exact_q(pre, target, [g]) is not None else -1e9) for g in range(G)])
cand = np.argsort(-qs)[:150].tolist()
mask = []
flip_at = sig_at = None
for _ in range(200):
    bb, bg = -np.inf, refit(mask)[0]
    for g in cand:
        if g in mask:
            continue
        if float(c[mask + [g]].sum()) > 600:
            continue
        b = refit(mask + [g])[0]
        if b > bb:
            bb, bg = b, g
    if bg is None:
        break
    mask.append(bg)
    beta, f = refit(mask)
    rows = int(c[mask].sum())
    g = f.g
    n_ent = d["firm_id"].nunique() - len(mask)   # kept entities
    t_fit = f.beta[j] / f.se[j]
    t = t_fit / np.sqrt((f.n - k_all) / (f.n - k_all - n_ent))   # p_eff convention
    crit = stats.t.ppf(1 - 0.05/2, g - 1)
    if flip_at is None and beta > 0:
        flip_at = rows
        print(f"flip at {rows} rows ({len(mask)} clusters), beta={beta:.4f}")
    if flip_at is not None and abs(t) > crit:
        sig_at = rows
        Hsum = sum(pre["H"][gg] for gg in mask)
        h_joint = np.linalg.norm(Hsum, 2)
        print(f"significant at {rows} rows ({len(mask)} clusters), beta={beta:.4f}, "
              f"t={t:.4f}, crit={crit:.4f}, df={g-1}")
        print(f"JOINT leverage lambda_max(sum H_g over set) = {h_joint:.4f}")
        print(f"1/(1-h_joint) = {1/(1-h_joint):.3f}")
        break
