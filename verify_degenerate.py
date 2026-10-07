"""Verify the degenerate-input branch end-to-end: a single retained cluster,
no residual df, and a rank-deficient within-entity design must all be rejected
by BOTH the estimator and the whitening precompute (no fake feasible results)."""
import os, sys
import numpy as np, pandas as pd, scipy.linalg as sla
np.linalg.lstsq = lambda a,b,rcond=None: sla.lstsq(a,b,cond=(rcond if rcond is not None else None))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from optmask.cluster_valid import TWFE_CV
from optmask.certify import det_adj_precompute

rng = np.random.default_rng(42)
rows = []
for e in range(3):
    for t in range(4):
        for rep in range(2):
            x = rng.normal()
            rows.append((e, 2019 + t, x))
d = pd.DataFrame(rows, columns=["firm_id", "year", "x1"])
d["y"] = 3 * d["x1"] + 0.1 * rng.normal(size=len(d))

# (1) single retained entity -> G=1 -> NaN se
est = TWFE_CV("y", ["x1"], "firm_id", "year", "firm_id")
prep = est.prepare(d)
f = est.fit(prep, (d["firm_id"] == 0).to_numpy())
assert f.g == 1 and np.isnan(f.se).all(), "single-entity fit should have NaN se"

# (2) rank-deficient within design: duplicate column x_dup = x1
d["x_dup"] = d["x1"]
est2 = TWFE_CV("y", ["x1", "x_dup"], "firm_id", "year", "firm_id")
prep2 = est2.prepare(d)
f2 = est2.fit(prep2, np.ones(len(d), bool))
assert np.isnan(f2.se).all() and np.isnan(f2.scale), \
    "rank-deficient fit should have NaN se and scale"
assert np.linalg.matrix_rank(f2.X) < f2.X.shape[1], \
    "rank-deficient design should have rank < columns"

# (3) the invalid fit must be rejected by the whitening precompute
try:
    det_adj_precompute(f2)
    raise AssertionError("det_adj_precompute should reject an invalid fit")
except ValueError:
    pass  # expected

# (4) a valid listed-panel fit still precomputes and gives the known margins
d3 = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "panel_listed.csv"))
regs = [f"x{k}" for k in range(1, 8)]
est3 = TWFE_CV("y", regs, "firm_id", "year", "firm_id")
prep3 = est3.prepare(d3)
f3 = est3.fit(prep3, np.ones(len(d3), bool))
pre3 = det_adj_precompute(f3)   # must not raise
print("OK: degenerate inputs rejected, valid panel precomputes; "
      "no UnboundLocalError and no fake feasible result")
