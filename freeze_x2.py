"""Freeze the x2 (Mfee) fragility evidence: a strongly significant coefficient that
collapses after deleting one 5-row high-leverage cluster."""
import sys
import os
import numpy as np
import pandas as pd
import scipy.linalg as sla
from scipy import stats

np.linalg.lstsq = lambda a, b, rcond=None: sla.lstsq(a, b, cond=(rcond if rcond is not None else None))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from optmask.cluster_valid import TWFE_CV
from optmask.certify import det_adj_precompute

HERE = os.path.dirname(os.path.abspath(__file__))
d = pd.read_csv(os.path.join(HERE, "data", "panel_firm.csv"))
regs = [f"x{k}" for k in range(1, 11)]
est = TWFE_CV("y", regs, "firm_id", "year", "firm_id")
prep = est.prepare(d)
f0 = est.fit(prep, np.ones(len(d), bool))
pre = det_adj_precompute(f0)
code = prep["cluster_code"]
h_g = np.array([np.linalg.norm(pre["H"][g], 2) for g in range(pre["G"])])
order = np.argsort(-h_g)


def stat(mask):
    keep = np.ones(len(d), bool)
    for g in mask:
        keep &= code != g
    return est.fit(prep, keep)


j = 1  # x2
f = stat([])
print(f"x2 full: beta={f.beta[j]:.4f}, se={f.se[j]:.4f}, t={f.beta[j]/f.se[j]:.2f}, n={f.n}, G={f.g}")
for k in [1, 2, 3]:
    mask = [int(g) for g in order[:k]]
    fk = stat(mask)
    rows = int(sum(pre["c"][g] for g in mask))
    print(f"x2 delete top-{k} clusters ({rows} rows): beta={fk.beta[j]:.4f}, "
          f"se={fk.se[j]:.4f}, t={fk.beta[j]/fk.se[j]:.2f}, G={fk.g}")

# exact number: deleting the single top cluster (5 rows) flips the sign and kills significance
f1 = stat([int(order[0])])
rows1 = int(pre["c"][int(order[0])])
print(f"\nSUMMARY: x2 t goes {f0.beta[j]/f0.se[j]:.2f} -> {f1.beta[j]/f1.se[j]:.2f} "
      f"after deleting {rows1} rows ({rows1/len(d):.1%} of the sample)")
