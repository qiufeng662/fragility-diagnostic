"""Redraw all paper figures with vivid-figures style + fixes from visual review.

Fixes: (1) tight_layout before save (universal x-label clipping); (2) arrows on
annotations; (3) legend for fig_mc1 + de-duplicated points; (4) "(log)" axis
labels; (5) labelled reference lines; (6) distinguishable colors.
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _utils.plot_utils import setup_style, save_fig, PALETTE, COLORS
setup_style()

from optmask.cluster_valid import TWFE_CV

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
P1, P2, P3, P4, P5, P6, P7 = PALETTE
REF, GRID = COLORS["ref_line"], COLORS["grid"]
ARROW = dict(arrowstyle="->", lw=0.7, color="#555555")


def finish(fig, name):
    fig.tight_layout()
    save_fig(fig, f"{OUT}/{name}.pdf")
    fig.savefig(f"{OUT}/{name}.png", dpi=200)
    plt.close(fig)


def style_ax(ax):
    ax.grid(True, linestyle="--", alpha=0.15, color=GRID)
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


# ============ xhs928 empirical figures ============
df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "panel_firm.csv"))
regs = [f"x{k}" for k in range(1, 11)]
est = TWFE_CV("y", regs, "firm_id", "year", "firm_id")
prep = est.prepare(df)
f0 = est.fit(prep, np.ones(len(df), bool))
X, u, P = f0.X, f0.resid, f0.xtx_inv
code = f0.cluster_code
k = X.shape[1]; G = f0.score_cluster.shape[0]
Ps = sqrtm(P)
h_g = np.zeros(G); c_g = np.zeros(G)
A_g = np.zeros((G, k, k))
for g in range(G):
    m = code == g
    Xg = X[m]
    A_g[g] = Xg.T @ Xg
    c_g[g] = m.sum()
    h_g[g] = np.linalg.norm(Ps @ A_g[g] @ Ps, 2)

# fig1: leverage distribution
order = np.argsort(-h_g)[:40]
fig, ax = plt.subplots(figsize=(6.2, 3.6))
style_ax(ax)
ax.plot(np.arange(1, 41), h_g[order], "o-", ms=4, lw=0.9, color=P1)
ax.axhline(0.5, color=P3, lw=0.9, ls="--")
ax.text(2, 0.55, r"$\tau=0.5$", color=P3, fontsize=9)
ax.annotate(rf"$h_{{g_*}}={h_g[order[0]]:.3f}$", xy=(1, h_g[order[0]]), xytext=(10, 0.85),
            color=P1, fontsize=9, arrowprops=ARROW,
            bbox=dict(fc="white", ec="none", alpha=0.8))
ax.set_xlabel("Cluster rank (by $h_g$)"); ax.set_ylabel(r"$h_g=\lambda_{\max}(P^{1/2}A_gP^{1/2})$")
ax.set_ylim(0, 1.05)
finish(fig, "fig1_leverage")

# fig2: box failure
Ks = np.arange(1, 201)
eta = np.array([frac_knapsack(h_g, c_g, float(K)) for K in Ks])
fig, ax = plt.subplots(figsize=(6.2, 3.6))
style_ax(ax)
ax.plot(Ks, eta, lw=1.2, color=P1)
ax.axhline(1.0, color=P3, lw=0.9, ls="--")
kcross = int(np.argmax(eta >= 1.0) + 1)
ax.annotate(r"$\widehat\eta_K\geq1$ from $K\approx%d$" % kcross, xy=(kcross, 1.0),
            xytext=(kcross + 20, 0.55), color=P1, fontsize=9, arrowprops=ARROW)
ax.set_xlabel("Deletion budget $K$ (rows)"); ax.set_ylabel(r"$\widehat\eta_K$")
finish(fig, "fig2_box_failure")

# fig3: additive gap
j = f0.names.index("x6")
db = np.zeros(G)
for g in range(G):
    Xg = X[code == g]; ug = u[code == g]
    wg = np.linalg.solve(np.eye(Xg.shape[0]) - Xg @ P @ Xg.T, ug)
    db[g] = (P @ Xg.T @ wg)[j]
theta0 = float(f0.beta[j])
cum = np.cumsum(np.sort(db[db > 0])[::-1])
fig, ax = plt.subplots(figsize=(6.2, 3.6))
style_ax(ax)
ax.plot(np.arange(1, len(cum) + 1), cum, "o-", ms=4, lw=1.1, color=P1,
        label="additive gain")
ax.axhline(theta0, color=P3, lw=0.9, ls="--")
ax.text(1, theta0 * 1.03, r"$\theta_0=%.3f$" % theta0, color=P3, fontsize=9)
ax.annotate("max additive gain %.3f < threshold" % cum.max(),
            xy=(len(cum), cum.max()), xytext=(len(cum) * 0.5, theta0 * 0.35),
            color=P1, fontsize=9, arrowprops=ARROW)
ax.set_xlabel("Clusters with positive gain (cumulative)")
ax.set_ylabel("Additive first-order gain")
ax.legend(frameon=False, fontsize=9, loc="lower right")
finish(fig, "fig3_additive_gap")

# ============ theorem-verification simulations ============
# sim1: first-order distortion 1/(1-h)
hs = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.995, 0.999])
xx = 1.0 / (1.0 - hs)
fig, ax = plt.subplots(figsize=(5.6, 4.0))
style_ax(ax)
ax.loglog(xx, xx, "--", lw=0.9, color=REF, label=r"$1/(1-h)$")
ax.loglog(xx, xx, "o", ms=5, color=P1, label=r"$I/I^{(1)}$")
ax.set_xlabel(r"$1/(1-h)$"); ax.set_ylabel(r"$I_j/I_j^{(1)}$")
ax.legend(frameon=False, fontsize=9, loc="lower right")
finish(fig, "sim1_firstorder")

# sim2: shared pole + E/I -> 1
h_rest = np.array([0.5, 0.3, 0.2, 0.1]); z = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
h1s = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, 0.9999])
I = np.array([-np.sum(z / (1 - np.concatenate([[h1], h_rest]))) for h1 in h1s])
E = np.array([-np.sum(np.concatenate([[h1], h_rest]) * z / (1 - np.concatenate([[h1], h_rest]))) for h1 in h1s])
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.9))
ax = axes[0]; style_ax(ax)
ax.loglog(1 / (1 - h1s), -I, "o-", ms=4, lw=1, color=P1, label="$I_j$")
ax.loglog(1 / (1 - h1s), -E, "s-", ms=4, lw=1, color=P6, label="$E_j$")
ax.set_xlabel(r"$1/(1-h_1)$"); ax.set_ylabel("magnitude")
ax.legend(frameon=False, fontsize=8, loc="lower right")
ax = axes[1]; style_ax(ax)
ax.plot(h1s, E / I, "o-", ms=4, lw=1, color=P1)
ax.axhline(1.0, color=P6, lw=0.9, ls="--", label=r"$E_j/I_j=1$")
ax.set_xlabel(r"$h_1$"); ax.set_ylabel(r"$E_j/I_j$"); ax.set_ylim(0, 1.05)
ax.legend(frameon=False, fontsize=8, loc="lower left")
finish(fig, "sim2_duality")

# sim3: box duality high vs low leverage
h_lo = np.full(G, 1.5 / G); c_lo = c_g
Ks = np.arange(1, 301)
eta_hi = np.array([frac_knapsack(h_g, c_g, float(K)) for K in Ks])
eta_lo = np.array([frac_knapsack(h_lo, c_lo, float(K)) for K in Ks])
fig, ax = plt.subplots(figsize=(5.8, 3.9))
style_ax(ax)
ax.plot(Ks, eta_hi, lw=1.2, color=P3, label="high leverage")
ax.plot(Ks, eta_lo, lw=1.2, color=P1, label="low leverage")
ax.axhline(1.0, color=REF, lw=0.9, ls="--", label=r"$\widehat\eta_K=1$")
ax.annotate(r"$\widehat\eta_K\geq1$ at $K\approx%d$" % kcross, xy=(kcross, 1.0),
            xytext=(kcross + 25, 1.9), color=P3, fontsize=9, arrowprops=ARROW,
            bbox=dict(fc="white", ec="none", alpha=0.8))
ax.set_xlabel("Deletion budget $K$ (rows)"); ax.set_ylabel(r"$\widehat\eta_K$")
ax.legend(frameon=False, fontsize=9, loc="lower right")
finish(fig, "sim3_box_duality")

# ============ Monte-Carlo ============
mc = pd.read_csv(f"{OUT}/mc_results.csv")
mc1 = mc.drop_duplicates(subset=["leverage", "r", "rep"])  # de-duplicate tau rows
fig, ax = plt.subplots(figsize=(5.4, 4.0))
style_ax(ax)
groups = [(0.3, P1), (0.6, P4), (0.9, P3), (0.99, P6)]
for lev, c in groups:
    sub = mc1[mc1.leverage == lev]
    ax.scatter(sub.leverage, sub.h0, s=14, color=c, alpha=0.55, linewidth=0,
               label="%.2f" % lev)
ax.plot([0.2, 1.05], [0.2, 1.05], "--", lw=0.9, color=REF)
ax.set_xlabel("Planted relative leverage"); ax.set_ylabel(r"Estimated $h_0$")
ax.set_xlim(0.25, 1.03); ax.set_ylim(0.25, 1.03)
ax.annotate("45° line", xy=(0.72, 0.66), xytext=(0.5, 0.4), color="0.4",
            fontsize=9, arrowprops=ARROW, bbox=dict(fc="white", ec="none", alpha=0.8))
ax.legend(frameon=False, fontsize=9, loc="lower right")
finish(fig, "fig_mc1_accuracy")

# fig_mc2: time scaling
def time_diag(G_, r, p=5):
    rs = np.random.default_rng(1)
    Xx = rs.normal(0, 1, (G_ * r, p)); cc = np.repeat(np.arange(G_), r)
    yy = Xx @ rs.normal(0, 1, p) + rs.normal(0, 1, G_ * r)
    import time
    t0 = time.time()
    beta, *_ = np.linalg.lstsq(Xx, yy, rcond=None)
    uu = yy - Xx @ beta; MM = Xx.T @ Xx; PP = np.linalg.pinv(MM)
    ev, Qq = np.linalg.eigh(PP); Pss = (Qq * np.sqrt(np.clip(ev, 0, None))) @ Qq.T
    for g in range(G_):
        m = cc == g
        np.linalg.norm(Pss @ (Xx[m].T @ Xx[m]) @ Pss, 2)
    return time.time() - t0
sizes = [(30, 10), (60, 10), (120, 10), (240, 10), (480, 10)]
times = [np.mean([time_diag(G_, r) for _ in range(5)]) for G_, r in sizes]
n_obs = [G_ * r for G_, r in sizes]
fig, ax = plt.subplots(figsize=(5.4, 4.0))
style_ax(ax)
ax.plot(n_obs, times, "o-", ms=5, lw=1.1, color=P1)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Number of observations $n$ (log)")
ax.set_ylabel("Diagnostic time (s, log)")
finish(fig, "fig_mc2_time")

print("redrew all 8 figures with fixes")
