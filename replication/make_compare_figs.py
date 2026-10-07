"""sim4: method comparison — exact (det-adj) vs first-order (AMIP) vs LOO-jackknife.

Panel 1 (left):  estimated signed deviation q_j after deleting the planted high-leverage
                 cluster, as its leverage h -> 1.  The exact (and the jackknife) value is
                 +B/2 (a genuine sign flip); the first-order model reads B(1/2 - h), which
                 turns negative once h > 1/2 — i.e. it predicts *no flip* where an exact
                 flip exists (Theorem on 1/(1-h) distortion).
Panel 2 (right): wall-clock cost of evaluating K candidate cluster-deletion sets:
                 naive leave-one-cluster-out refitting (K refits, O(K n p^2)) vs the
                 determinant/adjugate evaluator (one O(n p^2) precompute + K evals at O(p^3)).
"""
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"font.size": 11, "axes.linewidth": 0.8,
                     "xtick.direction": "in", "ytick.direction": "in"})

# ============================================================================
# Panel 1: whitened-coordinate construction (M = I), a = e_1, s = -1.
# Full-data beta_1 = B/2 > 0.  Deleting the planted cluster 0 flips the sign.
# I_j(e_0) = B exactly; the first-order gain is (1-h)B (Theorem on 1/(1-h)).
# ============================================================================
B = 4.0
hs = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.995, 0.999])

q_full = -B / 2.0                       # q^0 = s (a^T beta - 0) = -B/2
q_exact = np.full_like(hs, q_full + B)  # = +B/2, independent of h (genuine flip)
q_fo = q_full + (1.0 - hs) * B          # = B(1/2 - h), wrong sign for h > 1/2

# ============================================================================
# Panel 2: cost of evaluating K candidate deletions, n rows / p regressors / G clusters.
# ============================================================================
rng = np.random.default_rng(0)
n, p, G = 4000, 8, 80
X = rng.normal(size=(n, p))
X[:, 0] = 1.0
beta_true = rng.normal(size=(p,))
y = X @ beta_true + rng.normal(size=(n))
cluster = rng.integers(0, G, size=n)

# --- coefficient evaluation for the first regressor (a = e_1) ---
e1 = np.zeros(p); e1[0] = 1.0

# --- shared precompute: cluster sufficient statistics (cached for all methods) ---
t_pre = time.perf_counter()
A_g = np.zeros((G, p, p))
Xgty = np.zeros((G, p))
for g in range(G):
    Xg = X[cluster == g]
    A_g[g] = Xg.T @ Xg
    Xgty[g] = Xg.T @ y[cluster == g]
M = X.T @ X
Xty = X.T @ y
t_precompute = time.perf_counter() - t_pre


def refit_cost(ks):
    """Naive LOO refit: rebuild Gram and cross-term from raw rows each time."""
    t0 = time.perf_counter()
    for g in range(int(ks)):
        keep = cluster != g
        Xk, yk = X[keep], y[keep]
        np.linalg.solve(Xk.T @ Xk, Xk.T @ yk)[0]
    return time.perf_counter() - t0


def cholesky_cost(ks):
    """Cached sufficient statistics: one direct solve per candidate."""
    t0 = time.perf_counter()
    for g in range(int(ks)):
        Mg = M - A_g[g]
        v = Xty - Xgty[g]
        np.linalg.solve(Mg, v)[0]
    return time.perf_counter() - t0


def detadj_cost(ks):
    """Determinant/adjugate coefficient: det + inv(adjugate) + two matvecs."""
    t0 = time.perf_counter()
    for g in range(int(ks)):
        Mg = M - A_g[g]
        v = Xty - Xgty[g]
        d = np.linalg.det(Mg)
        adj = d * np.linalg.inv(Mg)
        q = e1 @ (adj @ v) / d
    return time.perf_counter() - t0


Ks = np.array([1, 2, 4, 8, 16, 32, 64, 80], dtype=float)
t_refit = np.array([refit_cost(k) for k in Ks])
t_chol = np.array([cholesky_cost(k) for k in Ks])
t_detadj = np.array([detadj_cost(k) for k in Ks])

# ============================================================================
# plot
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))

ax = axes[0]
ax.axhline(0.0, color="#AAAAAA", lw=0.8, ls="--")
ax.plot(hs, q_exact, "o-", ms=5, lw=1.2, color="#BC6C25",
        label="exact (det-adj / jackknife)")
ax.plot(hs, q_fo, "s-", ms=5, lw=1.2, color="#606C38",
        label="first-order (AMIP)")
ax.annotate("exact: $q_j=+B/2$ (flip)", xy=(0.85, q_exact[-3]),
            xytext=(0.52, 2.4), fontsize=9, color="#BC6C25",
            bbox=dict(fc="white", ec="none", alpha=0.7))
ax.annotate("first-order: $q_j=B(1/2-h)$ (no flip)", xy=(0.93, q_fo[-1]),
            xytext=(0.76, -1.95), fontsize=9, color="#606C38",
            bbox=dict(fc="white", ec="none", alpha=0.7))
ax.set_xlabel("planted leverage $h$")
ax.set_ylabel("signed deviation $q_j$ after deletion")
ax.set_ylim(-2.1, 2.6)
ax.set_title("(a)", loc="left")
ax.legend(frameon=False, fontsize=8, loc="upper right")

ax = axes[1]
ax.plot(Ks, t_refit * 1e3, "o-", ms=5, lw=1.2, color="#606C38",
        label="LOO refit (rebuild Gram)")
ax.plot(Ks, t_chol * 1e3, "^--", ms=5, lw=1.1, color="#7F4F24",
        label="cached Cholesky solve")
ax.plot(Ks, t_detadj * 1e3, "s-", ms=5, lw=1.2, color="#BC6C25",
        label="determinant/adjugate")
ax.set_xlabel("number of candidate deletions $K$")
ax.set_ylabel("coefficient-evaluation wall-clock (ms)")
ax.set_title("(b)", loc="left")
ax.legend(frameon=False, fontsize=8, loc="upper left")

fig.tight_layout()
fig.savefig(f"{OUT}/sim4_method_compare.pdf")
fig.savefig(f"{OUT}/sim4_method_compare.png", dpi=150)
plt.close(fig)

print("wrote sim4 to", OUT)
print(f"q_exact = {q_exact[0]:.3f} (constant, genuine flip)")
print(f"q_fo    = {np.round(q_fo, 3)}")
print(f"first-order wrong-sign (q_fo<0) from h={hs[np.argmax(q_fo < 0)]}")
print(f"precompute (shared sufficient statistics): {t_precompute*1e3:.2f} ms")
print(f"refit @K=80: {t_refit[-1]*1e3:.1f} ms; cached Cholesky @K=80: {t_chol[-1]*1e3:.1f} ms; det-adj @K=80: {t_detadj[-1]*1e3:.1f} ms")
