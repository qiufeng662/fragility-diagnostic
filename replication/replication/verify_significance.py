"""Check whether the planted 3-cluster deletion is sign-flip + SIGNIFICANT (CR1
margin > 0) under the controllable-leverage whitened DGP, and how the margin
behaves along the search path."""
import numpy as np
from scipy import stats

rng = np.random.default_rng(20261005)
P = 6
B = 4.0
K = 3
H_LOW = 0.6


def build_instance(G):
    A = rng.normal(size=(P, P))
    R, _ = np.linalg.qr(A)
    a = R[:, 0]
    H = np.zeros((G, P, P))
    Dp = np.zeros((P, P)); Dp[0, 0] = H_LOW / K
    for g in range(K):
        H[g] = R @ Dp @ R.T
    Db = np.zeros((P, P))
    Db[0, 0] = (1.0 - H_LOW) / (G - K)
    for d in range(1, P):
        Db[d, d] = 1.0 / (G - K)
    for g in range(K, G):
        H[g] = R @ Db @ R.T
    HD = H[:K].sum(axis=0)
    Qdel = np.eye(P) - HD
    r = B * (Qdel @ a)
    psi = np.zeros((G, P))
    psi[:K] = r[None, :] / K
    psi[K:] = -r[None, :] / (G - K)
    c = rng.integers(P, 21, size=G).astype(float)
    return H, psi, c, a


def stat(mask, H, psi, c, a):
    if not mask:
        return B / 2.0, 0.0, None
    HD = H[mask].sum(axis=0)
    Q = np.eye(P) - HD
    r = psi[mask].sum(axis=0)
    beta1 = B / 2.0 - a @ np.linalg.solve(Q, r)
    # retained-cluster re-estimated scores
    kept = [g for g in range(len(H)) if g not in mask]
    v = 0.0
    for g in kept:
        psig = psi[g] + H[g] @ np.linalg.solve(Q, r)
        v += (a @ np.linalg.solve(Q, psig)) ** 2
    G_new = len(kept)
    n_new = sum(c[g] for g in kept)
    if G_new < 1 or n_new <= P:
        return beta1, v, None
    rho = stats.t.ppf(1 - 0.05/2, G_new - 1) ** 2 * (G_new / (G_new - 1)) * ((n_new - 1) / (n_new - P))
    margin = (-beta1) ** 2 - rho * v
    return beta1, v, margin


for G in [15, 30]:
    H, psi, c, a = build_instance(G)
    b_full, _, _ = stat([], H, psi, c, a)
    for name, mask in [("planted-3", list(range(K))), ("planted-2", [0, 1])]:
        b, v, m = stat(mask, H, psi, c, a)
        print(f"G={G} {name}: beta1={b:+.3f}, v={v:.2e}, margin={m:+.4f} "
              f"({'significant' if (m is not None and m > 0) else 'NOT significant'})")
