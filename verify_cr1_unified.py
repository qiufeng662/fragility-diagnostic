"""Verify the reviewer's 45-row / 48-row margin sign under the unified p_eff CR1."""
import os, sys
import numpy as np, pandas as pd, scipy.linalg as sla
np.linalg.lstsq = lambda a,b,rcond=None: sla.lstsq(a,b,cond=(rcond if rcond is not None else None))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from optmask.cluster_valid import TWFE_CV
from optmask.certify import det_adj_precompute, det_adj_margin
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
target = Target(column="x1", sign=1, alpha=0.05)
pre = det_adj_precompute(f0)

sets = {
    "45-row": [42, 59, 36, 53, 70, 18, 23],
    "48-row": [42, 59, 36, 53, 70, 18, 23, 51],
}

for name, mask in sets.items():
    # det_adj margin (now p_eff convention)
    m_da = det_adj_margin(pre, target, mask)
    # exact refit margin (p_eff convention, since fit now uses p_eff)
    keep = complete.copy()
    for g in mask:
        keep &= prep["cluster_code"] != g
    f = est.fit(prep, keep)
    q = float(target.sign * f.beta[j])
    t = f.beta[j] / f.se[j]
    crit = stats.t.ppf(1 - 0.05/2, f.g - 1)
    m_refit = q**2 - (crit * f.se[j])**2
    print(f"{name}: det_adj margin={m_da:+.4e}, refit margin={m_refit:+.4e}, "
          f"t={t:.4f}, crit={crit:.4f}, n={f.n}, G={f.g}, p_eff={f.X.shape[1]+f.n_entities}")

# also verify baseline t matches paper's full-rank CR1 (listed -0.457)
t_base = f0.beta[j] / f0.se[j]
print(f"\nbaseline listed t = {t_base:.4f} (paper: -0.457)")
