"""Unified Figure 1: the four simulation-verification panels in one 2x2 figure,
so the subplots share one size, font and palette.

(a) first-order distortion 1/(1-h);  (b) shared pole (I_j and E_j);
(c) box certificate under high vs low leverage;  (d) planted-leverage recovery.
"""
import os
import sys
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

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

OLIVE, TAN, BROWN, DARK, REF = "#606C38", "#A68A64", "#BC6C25", "#7F4F24", "#AAAAAA"

plt.rcParams.update({"font.size": 11, "axes.linewidth": 0.8,
                     "xtick.direction": "in", "ytick.direction": "in"})


def style(ax):
    ax.grid(True, linestyle="--", alpha=0.15, color="#E0E0E0")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


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


# ---- (c) box duality needs the firm-level leverage profile ----
df = pd.read_csv(os.path.join(HERE, "data", "panel_firm.csv"))
regs = [f"x{k}" for k in range(1, 11)]
est = TWFE_CV("y", regs, "firm_id", "year", "firm_id")
prep = est.prepare(df)
f0 = est.fit(prep, np.ones(len(df), bool))
Ps = sqrtm(f0.xtx_inv)
code = f0.cluster_code
G = f0.score_cluster.shape[0]
h_g = np.zeros(G); c_g = np.zeros(G)
for g in range(G):
    m = code == g
    Xg = f0.X[m]
    c_g[g] = m.sum()
    h_g[g] = np.linalg.norm(Ps @ (Xg.T @ Xg) @ Ps, 2)

fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.6))

# (a) first-order distortion
ax = axes[0, 0]; style(ax)
hs = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.995, 0.999])
xx = 1.0 / (1.0 - hs)
ax.loglog(xx, xx, "--", lw=0.9, color=REF, label=r"$1/(1-h)$")
ax.loglog(xx, xx, "o", ms=5, color=OLIVE, label=r"$I_j/I_j^{(1)}$")
ax.set_xlabel(r"$1/(1-h)$")
ax.set_ylabel(r"$I_j/I_j^{(1)}$")
ax.legend(frameon=False, fontsize=10, loc="upper left")
ax.text(0.03, 0.95, "(a)", transform=ax.transAxes, fontsize=12, va="top", weight="bold")

# (b) shared pole
ax = axes[0, 1]; style(ax)
h_rest = np.array([0.5, 0.3, 0.2, 0.1]); z = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
h1s = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, 0.9999])
I = np.array([-np.sum(z / (1 - np.concatenate([[h1], h_rest]))) for h1 in h1s])
E = np.array([-np.sum(np.concatenate([[h1], h_rest]) * z / (1 - np.concatenate([[h1], h_rest]))) for h1 in h1s])
ax.loglog(1 / (1 - h1s), -I, "o-", ms=4, lw=1.1, color=OLIVE, label=r"$I_j$")
ax.loglog(1 / (1 - h1s), -E, "s-", ms=4, lw=1.1, color=DARK, label=r"$E_j$")
ax.set_xlabel(r"$1/(1-h_1)$")
ax.set_ylabel("magnitude")
ax.legend(frameon=False, fontsize=10, loc="upper left")
ax.text(0.03, 0.95, "(b)", transform=ax.transAxes, fontsize=12, va="top", weight="bold")

# (c) box certificate
ax = axes[1, 0]; style(ax)
h_lo = np.full(G, 1.5 / G); c_lo = c_g
Ks = np.arange(1, 301)
eta_hi = np.array([frac_knapsack(h_g, c_g, float(K)) for K in Ks])
eta_lo = np.array([frac_knapsack(h_lo, c_lo, float(K)) for K in Ks])
kcross = int(np.argmax(eta_hi >= 1.0) + 1)
ax.plot(Ks, eta_hi, lw=1.3, color=BROWN, label="high leverage")
ax.plot(Ks, eta_lo, lw=1.3, color=OLIVE, label="low leverage")
ax.axhline(1.0, color=REF, lw=0.9, ls="--", label=r"$\widehat\eta_K=1$")
ax.annotate(r"$\widehat\eta_K\geq1$ at $K\approx%d$" % kcross, xy=(kcross, 1.0),
            xytext=(kcross + 25, 1.9), color=BROWN, fontsize=10,
            arrowprops=dict(arrowstyle="->", lw=0.7, color="#555555"),
            bbox=dict(fc="white", ec="none", alpha=0.8))
ax.set_xlabel("Deletion budget $K$ (rows)")
ax.set_ylabel(r"$\widehat\eta_K$")
ax.legend(frameon=False, fontsize=10, loc="lower right")
ax.text(0.03, 0.95, "(c)", transform=ax.transAxes, fontsize=12, va="top", weight="bold")

# (d) planted-leverage recovery
ax = axes[1, 1]; style(ax)
rng = np.random.default_rng(20261002)


def gen_panel(G_, r_, p_, lev, seed):
    rs = np.random.default_rng(seed)
    X = np.zeros((G_ * r_, p_))
    X[0:r_, 0] = np.sqrt(r_ * lev)
    X[r_:, 0] = np.sqrt(r_ * (1.0 - lev) / (G_ - 1))
    X[r_:, 1:] = rs.normal(0, 1, ((G_ - 1) * r_, p_ - 1))
    X[0:r_, 1:] = rs.normal(0, 0.01, (r_, p_ - 1))
    beta = rs.normal(0, 1, p_)
    y = X @ beta + rs.normal(0, 1, G_ * r_)
    return X, y


def h0_est(X, y, G_, r_):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    u = y - X @ beta
    Ps_ = sqrtm(np.linalg.pinv(X.T @ X))
    Ag = X[0:r_].T @ X[0:r_]
    return np.linalg.norm(Ps_ @ Ag @ Ps_, 2)


for lev, c in [(0.3, OLIVE), (0.6, TAN), (0.9, BROWN), (0.99, DARK)]:
    vals = []
    for r_ in [5, 20]:
        for rep in range(50):
            X, y = gen_panel(60, r_, 5, lev, 1000 + rep)
            vals.append(h0_est(X, y, 60, r_))
    ax.scatter([lev] * len(vals), vals, s=14, color=c, alpha=0.55, linewidth=0,
               label="%.2f" % lev)
ax.plot([0.2, 1.05], [0.2, 1.05], "--", lw=0.9, color=REF)
ax.set_xlabel("Planted relative leverage")
ax.set_ylabel(r"Estimated $h_0$")
ax.set_xlim(0.25, 1.03); ax.set_ylim(0.25, 1.03)
ax.legend(frameon=False, fontsize=10, loc="lower right", title="leverage")
ax.text(0.03, 0.95, "(d)", transform=ax.transAxes, fontsize=12, va="top", weight="bold")

fig.tight_layout()
fig.savefig(f"{OUT}/fig_sim_combined.pdf")
fig.savefig(f"{OUT}/fig_sim_combined.png", dpi=200)
plt.close(fig)
print("wrote", f"{OUT}/fig_sim_combined.pdf")
