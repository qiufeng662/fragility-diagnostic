"""Recalc firm-panel x2 coefficient se/t under the unified p_eff CR1 convention."""
import os, sys
import numpy as np, pandas as pd, scipy.linalg as sla
np.linalg.lstsq = lambda a,b,rcond=None: sla.lstsq(a,b,cond=(rcond if rcond is not None else None))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from optmask.cluster_valid import TWFE_CV
from optmask.certify import det_adj_precompute
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
d = pd.read_csv(os.path.join(HERE, "data", "panel_firm.csv"))
regs = [f"x{k}" for k in range(1, 11)]
est = TWFE_CV("y", regs, "firm_id", "year", "firm_id")
prep = est.prepare(d)
complete = np.ones(len(d), bool)
f0 = est.fit(prep, complete)
pre = det_adj_precompute(f0)
code = prep["cluster_code"]
k_all = prep["X"].shape[1]          # 19
n_firms = prep["n_entities"]        # 447
p_eff = k_all + n_firms             # 466
n = len(d)

h_g = np.array([np.linalg.norm(pre["H"][g], 2) for g in range(pre["G"])])
order = np.argsort(-h_g)
j = 1  # x2

def stat(mask):
    keep = complete.copy()
    for g in mask:
        keep &= code != g
    f = est.fit(prep, keep)
    # p_eff CR1 scale: replace (n-1)/(n-k_all) by (n-1)/(n-p_eff)
    se_old = f.se[j]
    se_new = se_old * np.sqrt((f.n - k_all) / (f.n - p_eff))
    return f.beta[j], se_new, f.beta[j] / se_new

b, se, t = stat([])
print(f"x2 full (p_eff): beta={b:.4f}, se={se:.4f}, t={t:.2f}")
b1, se1, t1 = stat([int(order[0])])
print(f"x2 delete top-1 cluster (p_eff): beta={b1:.4f}, se={se1:.4f}, t={t1:.2f}")
