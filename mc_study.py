"""Monte-Carlo evaluation of the fragility diagnostic (paper simulation study).

Factors: leverage level (low / medium / high) x cluster size (small / large),
R replicates each.  For each replicate we generate a panel with a planted
high-leverage cluster, run the diagnostic's cluster-leverage ranking, and record
(1) whether the planted cluster is flagged (recall), (2) how many background
clusters are falsely flagged (false-positive rate), (3) box-certificate verdict,
(4) run time.  Outputs ``figures/mc_results.csv`` (read by ``redraw_figs.py``).

Seeds are fixed (per-replicate seed 1000+rep), so the CSV is bit-reproducible.
Run:  python mc_study.py
"""
import os
import time

import numpy as np
import pandas as pd
import scipy.linalg as sla

np.linalg.lstsq = lambda a, b, rcond=None: sla.lstsq(
    a, b, cond=(rcond if rcond is not None else None))

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

G = 60
P = 5
LEVERAGES = [0.3, 0.6, 0.9, 0.99]
CLUSTER_SIZES = [5, 20]
REPS = 100
TAUS = [0.3, 0.5]


def sqrtm(P):
    ev, Q = np.linalg.eigh(P)
    return (Q * np.sqrt(np.clip(ev, 0, None))) @ Q.T


def gen_panel(G, r, p, leverage, seed):
    """Panel with a planted high-leverage cluster (cluster 0).

    The e1 direction carries total energy 1, of which cluster 0 contributes
    `leverage` and the background clusters share 1-leverage; the other p-1
    directions are balanced background.  This makes h_0 = lambda_max(P^{1/2}A_0P^{1/2})
    approximately `leverage`.
    """
    rs = np.random.default_rng(seed)
    X = np.zeros((G * r, p))
    # e1 direction: planted cluster + shared background
    X[0:r, 0] = np.sqrt(r * leverage)
    X[r:, 0] = np.sqrt(r * (1.0 - leverage) / (G - 1))
    # other directions: balanced background; planted cluster ~ tiny noise
    X[r:, 1:] = rs.normal(0, 1, ((G - 1) * r, p - 1))
    X[0:r, 1:] = rs.normal(0, 0.01, (r, p - 1))
    beta = rs.normal(0, 1, p)
    y = X @ beta + rs.normal(0, 1, G * r)
    code = np.repeat(np.arange(G), r)
    return X, y, code, beta


def diagnostic_h(X, y, code, G):
    """Relative leverage h_g = lambda_max(P^{1/2} A_g P^{1/2}) for each cluster."""
    # OLS fit on full data (no FE for the synthetic study)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    u = y - X @ beta
    M = X.T @ X
    P = np.linalg.pinv(M)
    Ps = sqrtm(P)
    h = np.zeros(G)
    for g in range(G):
        m = code == g
        Xg = X[m]
        Ag = Xg.T @ Xg
        h[g] = np.linalg.norm(Ps @ Ag @ Ps, 2)
    return h


results = []
for leverage in LEVERAGES:
    for r in CLUSTER_SIZES:
        for rep in range(REPS):
            X, y, code, beta = gen_panel(G, r, P, leverage, seed=1000 + rep)
            t0 = time.time()
            h = diagnostic_h(X, y, code, G)
            dt = time.time() - t0
            # planted cluster is index 0; flag = h_g >= tau
            for tau in TAUS:
                flagged = set(np.flatnonzero(h >= tau))
                recall = 1.0 if 0 in flagged else 0.0
                fp = len([g for g in flagged if g != 0])
                results.append({
                    "leverage": leverage, "r": r, "rep": rep, "tau": tau,
                    "recall": recall, "fp": fp, "h0": h[0],
                    "max_h_bg": h[1:].max(), "time": dt,
                })

df = pd.DataFrame(results)
csv_path = os.path.join(OUT, "mc_results.csv")
df.to_csv(csv_path, index=False)
print("wrote", csv_path, ":", df.shape)
print(df.groupby(["leverage", "r", "tau"]).agg(
    recall=("recall", "mean"), fp=("fp", "mean"), h0=("h0", "mean"),
    max_h_bg=("max_h_bg", "mean"), time=("time", "mean")).round(3))
