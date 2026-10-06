"""Generate the empirical figures (Section 8, firm-level panel) from anonymized data.

Fig 1: cluster-level relative leverage (sorted), dangerous clusters flagged.
Fig 2: global Neumann box eta_K vs budget K, failure threshold K ~ 6.
Fig 3: additive first-order gain vs the flip threshold (additive infeasible).
"""
import sys
import os
import numpy as np
import pandas as pd
import scipy.linalg as sla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.linalg.lstsq = lambda a, b, rcond=None: sla.lstsq(
    a, b, cond=(rcond if rcond is not None else None))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from optmask.cluster_valid import TWFE_CV

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)


def sqrtm(P):
    ev, Q = np.linalg.eigh(P)
    return (Q * np.sqrt(np.clip(ev, 0, None))) @ Q.T


def frac_knapsack(values, weights, K):
    order = np.argsort(-values / np.maximum(weights, 1e-12))
    budget = float(K); total = 0.0
    for g in order:
        w = weights[g]
        if w <= budget:
            total += values[g]; budget -= w
        elif budget > 0:
            total += values[g] * (budget / w); budget = 0
        else:
            break
    return total


plt.rcParams.update({"font.size": 11, "axes.linewidth": 0.8,
                     "xtick.direction": "in", "ytick.direction": "in"})

df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "panel_firm.csv"))
regs = [f"x{k}" for k in range(1, 11)]
est = TWFE_CV("y", regs, "firm_id", "year", "firm_id")
prep = est.prepare(df)
f0 = est.fit(prep, np.ones(len(df), bool))

X, u, P = f0.X, f0.resid, f0.xtx_inv
code = f0.cluster_code
k = X.shape[1]
G = f0.score_cluster.shape[0]
Ps = sqrtm(P)

h_g = np.zeros(G); c_g = np.zeros(G)
A_g = np.zeros((G, k, k)); psi_g = np.zeros((G, k))
for g in range(G):
    m = code == g
    Xg = X[m]
    A_g[g] = Xg.T @ Xg
    psi_g[g] = Xg.T @ u[m]
    c_g[g] = m.sum()
H_g = np.einsum('ab,gbr,rs->gas', Ps, A_g, Ps)
h_g = np.array([np.linalg.norm(H_g[g], 2) for g in range(G)])

# ---- Fig 1: leverage distribution (top-40 by h_g) ----
order = np.argsort(-h_g)[:40]
fig, ax = plt.subplots(figsize=(6.2, 3.6))
ax.plot(np.arange(1, 41), h_g[order], "o-", ms=4, lw=0.8, color="#606C38")
ax.axhline(0.5, color="#BC6C25", lw=0.8, ls="--")
ax.text(2, 0.52, "danger threshold $\\tau=0.5$", color="#BC6C25", fontsize=9,
        bbox=dict(fc="white", ec="none", alpha=0.7))
ax.annotate(f"$h_{{g_*}}={h_g[order[0]]:.3f}$ (cluster {int(order[0])})",
            xy=(1, h_g[order[0]]), xytext=(8, 0.85),
            arrowprops=dict(arrowstyle="->", lw=0.7), fontsize=9,
            bbox=dict(fc="white", ec="none", alpha=0.8))
ax.set_xlabel("Cluster rank (by $h_g$ descending)")
ax.set_ylabel("$h_g=\\lambda_{\\max}(P^{1/2}A_gP^{1/2})$")
ax.set_ylim(0, 1.05)
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_leverage.pdf"); fig.savefig(f"{OUT}/fig1_leverage.png", dpi=150)
plt.close(fig)

# ---- Fig 2: box certificate failure eta_K vs K ----
Ks = np.arange(1, 201)
eta = np.array([frac_knapsack(h_g, c_g, float(K)) for K in Ks])
fig, ax = plt.subplots(figsize=(6.2, 3.6))
ax.plot(Ks, eta, lw=1.2, color="#606C38")
ax.axhline(1.0, color="#BC6C25", lw=0.8, ls="--")
kcross = int(np.argmax(eta >= 1.0) + 1)
ax.axvline(kcross, color="#AAAAAA", lw=0.6, ls=":")
ax.annotate(f"$\\widehat\\eta_K\\geq1$ from $K\\approx{kcross}$ rows",
            xy=(kcross, 1.0), xytext=(kcross + 15, 0.6),
            arrowprops=dict(arrowstyle="->", lw=0.7), fontsize=9,
            bbox=dict(fc="white", ec="none", alpha=0.8))
ax.set_xlabel("Deletion budget $K$ (rows)")
ax.set_ylabel("$\\widehat\\eta_K=\\max\\sum_g h_g x_g$")
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_box_failure.pdf"); fig.savefig(f"{OUT}/fig2_box_failure.png", dpi=150)
plt.close(fig)

# ---- Fig 3: additive first-order gain vs flip threshold ----
j = f0.names.index("x6")
db = np.zeros(G)
for g in range(G):
    Xg = X[code == g]; ug = u[code == g]
    wg = np.linalg.solve(np.eye(Xg.shape[0]) - Xg @ P @ Xg.T, ug)
    db[g] = (P @ Xg.T @ wg)[j]
theta0 = float(f0.beta[j])
pos = np.sort(db[db > 0])[::-1]
cum = np.cumsum(pos)
fig, ax = plt.subplots(figsize=(6.2, 3.6))
ax.plot(np.arange(1, len(cum) + 1), cum, "o-", ms=4, lw=1.2, color="#606C38",
        label="additive gain $\\sum_{g} \\max(\\Delta_g^{(1)},0)$")
ax.axhline(theta0, color="#BC6C25", lw=0.8, ls="--")
ax.text(1, theta0 * 1.08, f"flip threshold $\\theta_0={theta0:.2f}$", color="#BC6C25", fontsize=9,
        bbox=dict(fc="white", ec="none", alpha=0.7))
ax.annotate(f"max additive gain $={cum.max():.3f}<\\theta_0$: infeasible",
            xy=(len(cum), cum.max()), xytext=(len(cum) * 0.55, 0.30),
            arrowprops=dict(arrowstyle="->", lw=0.7), fontsize=9,
            bbox=dict(fc="white", ec="none", alpha=0.8))
ax.set_xlabel("Clusters with $\\Delta_g^{(1)}>0$ (cumulative)")
ax.set_ylabel("Additive first-order gain")
ax.set_ylim(0, 1.6)
ax.legend(fontsize=8, frameon=False, loc="center right")
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_additive_gap.pdf"); fig.savefig(f"{OUT}/fig3_additive_gap.png", dpi=150)
plt.close(fig)

print("wrote figures to", OUT)
print(f"h_g max = {h_g.max():.4f} (cluster {int(h_g.argmax())})")
print(f"eta_K >= 1 from K = {kcross} rows")
print(f"theta0 = {theta0:.4f}, max additive gain = {cum.max():.4f}")
