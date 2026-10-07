"""Verify the degenerate-input branch: a single retained cluster (G<=1) and a
rank-deficient within-entity design must return NaN standard errors (not raise
UnboundLocalError)."""
import os, sys
import numpy as np, pandas as pd, scipy.linalg as sla
np.linalg.lstsq = lambda a,b,rcond=None: sla.lstsq(a,b,cond=(rcond if rcond is not None else None))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from optmask.cluster_valid import TWFE_CV

# three entities, 8 rows each, 4 years, one covariate -> keep one entity => G=1
rows = []
for e in range(3):
    for t in range(4):
        for rep in range(2):
            rows.append((e, 2019 + t, e * 0.5 + t * 0.1 + rep))
d = pd.DataFrame(rows, columns=["firm_id", "year", "x1"])
d["y"] = d["x1"] + np.random.default_rng(0).normal(size=len(d))
est = TWFE_CV("y", ["x1"], "firm_id", "year", "firm_id")
prep = est.prepare(d)

# keep only entity 0 -> G=1
keep = (d["firm_id"] == 0).to_numpy()
f = est.fit(prep, keep)
print(f"single-entity: G={f.g}, se={f.se} (should be NaN, not raise)")

# rank-deficient: add x_dup = x1 (duplicate column) -> within design rank < k
d["x_dup"] = d["x1"]
est2 = TWFE_CV("y", ["x1", "x_dup"], "firm_id", "year", "firm_id")
prep2 = est2.prepare(d)
keep2 = np.ones(len(d), bool)
f2 = est2.fit(prep2, keep2)
print(f"rank-deficient: se={np.round(f2.se, 6)} (NaN expected for the duplicate pair)")
print("OK: no UnboundLocalError")
