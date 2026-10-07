"""Listed-firm panel figure (Section 8, second panel): systematic diagnostic.

(a) relative-leverage profile (sorted) -- all clusters are low-leverage;
(b) pruned-search trajectory -- sign flip at 6 rows, significance at 55 rows.
"""
import sys
import os
import numpy as np
import pandas as pd
import scipy.linalg as sla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

np.linalg.lstsq = lambda a, b, rcond=None: sla.lstsq(
    a, b, cond=(rcond if rcond is not None else None))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from optmask.cluster_valid import TWFE_CV
from optmask.certify import det_adj_precompute
from optmask.diagnostics import _exact_q
from optmask.contract import Target

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"font.size": 11, "axes.linewidth": 0.8,
                     "xtick.direction": "in", "ytick.direction": "in"})

d = pd.read_csv(os.path.join(HERE, "data", "panel_listed.csv"))
regs = [f"x{k}" for k in range(1, 8)]
est = TWFE_CV("y", regs, "firm_id", "year", "firm_id")
prep = est.prepare(d)
complete = np.ones(len(d), bool)
f0 = est.fit(prep, complete)
j = f0.names.index("x1")
target = Target(column="x1", sign=1, alpha=0.05)
pre = det_adj_precompute(f0)
code = prep["cluster_code"]
c = pre["c"]
G = pre["G"]
n = len(d)
k_all = prep["X"].shape[1]

h_g = np.array([np.linalg.norm(pre["H"][g], 2) for g in range(G)])


def refit(mask):
    keep = complete.copy()
    for g in mask:
        keep &= code != g
    f = est.fit(prep, keep)
    return f.beta[j], f.beta[j] / f.se[j], f.g


def search():
    qs = np.array([(_exact_q(pre, target, [g]) if _exact_q(pre, target, [g]) is not None
                    else -1e9) for g in range(G)])
    cand = np.argsort(-qs)[:150].tolist()
    b0, t0, _ = refit([])
    mask, rows_list, beta_list, t_list = [], [0], [b0], [t0]
    for _ in range(200):
        bb, bg = -np.inf, refit(mask)[0]
        for g in cand:
            if g in mask:
                continue
            if float(c[mask + [g]].sum()) > 600:
                continue
            b = refit(mask + [g])[0]
            if b > bb:
                bb, bg = b, g
        if bg is None:
            break
        mask.append(bg)
        beta, t, gnum = refit(mask)
        rows_list.append(int(c[mask].sum()))
        beta_list.append(beta)
        t_list.append(t)
        crit = stats.t.ppf(1 - 0.05 / 2, gnum - 1)
        if abs(t) > crit and beta > 0:
            break
    return np.array(rows_list), np.array(beta_list), np.array(t_list), len(mask)


fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.6))

# (a) leverage profile
order = np.argsort(-h_g)[:40]
ax[0].plot(np.arange(1, 41), h_g[order], "o-", ms=4, lw=0.8, color="#606C38")
ax[0].axhline(0.5, color="#BC6C25", lw=0.8, ls="--")
ax[0].text(2, 0.52, "danger threshold $\\tau=0.5$", color="#BC6C25", fontsize=9,
           bbox=dict(fc="white", ec="none", alpha=0.7))
ax[0].annotate(f"max $h_g={h_g.max():.2f}$: no high-leverage cluster",
               xy=(1, h_g.max()), xytext=(8, 0.32),
               arrowprops=dict(arrowstyle="->", lw=0.7), fontsize=9,
               bbox=dict(fc="white", ec="none", alpha=0.8))
ax[0].set_xlabel("Cluster rank (by $h_g$ descending)")
ax[0].set_ylabel("$h_g=\\lambda_{\\max}(P^{1/2}A_gP^{1/2})$")
ax[0].set_ylim(0, 1.05)
ax[0].set_title("(a) leverage profile", fontsize=10)

# (b) trajectory
rows, beta, t, n_del = search()
ax[1].plot(rows, beta, "o-", ms=4, lw=1.2, color="#606C38")
ax[1].axhline(0, color="#AAAAAA", lw=0.7, ls=":")
axb = ax[1].twinx()
axb.plot(rows, np.abs(t), "s--", ms=3, lw=1.0, color="#BC6C25")
crit_final = stats.t.ppf(1 - 0.05 / 2, f0.g - n_del - 1)
axb.axhline(crit_final, color="#BC6C25", lw=0.8, ls="--", alpha=0.6)
ifl = int(np.argmax(beta > 0))
ax[1].annotate(f"sign flip at {rows[ifl]} rows",
               xy=(rows[ifl], 0.0), xytext=(10, 0.0115),
               arrowprops=dict(arrowstyle="->", lw=0.7), fontsize=9,
               bbox=dict(fc="white", ec="none", alpha=0.8))
ax[1].annotate(f"significant: $t={t[-1]:.2f}$ at {rows[-1]} rows",
               xy=(rows[-1], beta[-1]), xytext=(22, 0.0005),
               arrowprops=dict(arrowstyle="->", lw=0.7), fontsize=9,
               bbox=dict(fc="white", ec="none", alpha=0.8))
ax[1].set_xlabel(f"Rows deleted ({n_del} clusters)")
ax[1].set_ylabel("$\\hat\\beta_{x1}$ (exact re-fit)")
axb.set_ylabel("$|t|$", color="#BC6C25")
axb.set_ylim(0, 3.0)
ax[1].set_ylim(-0.004, 0.013)
ax[1].set_title("(b) pruned-search trajectory", fontsize=10)

fig.tight_layout()
fig.savefig(f"{OUT}/fig4_listed_diag.pdf")
fig.savefig(f"{OUT}/fig4_listed_diag.png", dpi=150)
plt.close(fig)
print("wrote figures to", OUT)
print(f"max h_g = {h_g.max():.4f}")
print(f"sign flip at {rows[ifl]} rows, significance at {rows[-1]} rows (t={t[-1]:.3f})")
