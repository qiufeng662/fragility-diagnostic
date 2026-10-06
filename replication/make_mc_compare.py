"""sim5: Monte-Carlo detection-rate comparison, exact vs first-order (AMIP).

Whitened-coordinate construction (M = I). A planted cluster with leverage h drives
a sign flip of the target coefficient; the flip is carried by the interaction term
Q^{-1} r, which the first-order model ignores.  Over MC draws of the planted score
we compare the *detection rate* (fraction of draws in which the method correctly
reports a flip) of the exact evaluator vs the first-order (AMIP) score.

Setup (Theorem 4.1 in p dimensions): target a = e_1, s = -1, full-data beta_1 = B/2
> 0 (so q^0 = -B/2 < 0). Deleting the planted cluster with Gram h e_1 e_1^T and
score psi_1 e_1 gives the exact shift beta_1 -> B/2 - psi_1/(1-h); the first-order
model gives B/2 - psi_1. A flip (beta_1 < 0) therefore happens exactly when
psi_1 > (B/2)(1-h), while the first-order model reports a flip only when psi_1 > B/2.
As h -> 1 the exact threshold vanishes but the first-order threshold does not, so
AMIP misses most flips.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.linewidth": 0.8,
                     "xtick.direction": "in", "ytick.direction": "in"})

rng = np.random.default_rng(0)
B2 = 2.0                       # threshold B/2
M = 2000                       # MC draws
hs = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.995, 0.999])

det_exact = []
det_amip = []
for h in hs:
    psi = rng.normal(size=M)                     # planted score along e_1
    flip = psi > B2 * (1 - h)                    # true flip (exact)
    amip = psi > B2                              # first-order reports flip
    # detection rate = P(report flip | true flip); denominator P(true flip)
    n_flip = flip.sum()
    det_exact.append(1.0 if n_flip else np.nan)
    det_amip.append((amip & flip).sum() / n_flip if n_flip else np.nan)

det_exact = np.array(det_exact)
det_amip = np.array(det_amip)

# theory: AMIP detection rate = P(psi > B2 | psi > B2(1-h))
amip_theory = np.array([(1 - stats.norm.cdf(B2)) / (1 - stats.norm.cdf(B2 * (1 - h))) for h in hs])

fig, ax = plt.subplots(figsize=(5.6, 4.0))
ax.plot(hs, det_exact, "o-", ms=5, lw=1.2, color="#BC6C25", label="exact (determinant/adjugate)")
ax.plot(hs, det_amip, "s-", ms=5, lw=1.2, color="#606C38", label="first-order (AMIP)")
ax.plot(hs, amip_theory, "--", lw=1.0, color="#606C38", alpha=0.6, label="AMIP theory")
ax.set_xlabel("planted leverage $h$")
ax.set_ylabel("flip detection rate")
ax.set_ylim(-0.05, 1.05)
ax.legend(frameon=False, fontsize=8, loc="center right")
fig.tight_layout()
fig.savefig(f"{OUT}/sim5_amip_detection.pdf")
fig.savefig(f"{OUT}/sim5_amip_detection.png", dpi=150)
plt.close(fig)

print("wrote sim5")
for h, de, da in zip(hs, det_exact, det_amip):
    print(f"  h={h:.3f}: exact det={de:.3f}, AMIP det={da:.3f}")
