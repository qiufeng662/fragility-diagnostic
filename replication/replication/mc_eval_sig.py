"""Full-significance method evaluation (the CR1 significance margin target).

Same controllable-leverage randomized whitened DGP as mc_eval.py, but the target
is now the full sign-and-significance margin m_j(x) = q_j(x)^2 - rho_j(x) v_j(x)
> 0 (not just the sign reversal beta_1 < 0).  The planted k=3 deletion set is
minimal AND significant (margin = +4.0); two clusters are neither flipping nor
significant.  We report the success rate, the feasible cost (cluster count), the
paired exhaustive optimum, and the evaluation cost of the margin criterion vs the
sign-only criterion.

Run:  python mc_eval_sig.py
"""
import time
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


def beta1_of(mask, H, psi, a):
    if not mask:
        return B / 2.0
    HD = H[mask].sum(axis=0)
    Q = np.eye(P) - HD
    r = psi[mask].sum(axis=0)
    return B / 2.0 - a @ np.linalg.solve(Q, r)


def margin_of(mask, H, psi, c, a):
    """Full CR1 significance margin m_j = q_j^2 - rho * v_j (sign s=-1)."""
    if not mask:
        return -np.inf
    HD = H[mask].sum(axis=0)
    Q = np.eye(P) - HD
    r = psi[mask].sum(axis=0)
    beta1 = B / 2.0 - a @ np.linalg.solve(Q, r)
    kept = [g for g in range(len(H)) if g not in mask]
    v = 0.0
    for g in kept:
        psig = psi[g] + H[g] @ np.linalg.solve(Q, r)
        v += (a @ np.linalg.solve(Q, psig)) ** 2
    G_new = len(kept)
    n_new = sum(c[g] for g in kept)
    if G_new < 1 or n_new <= P:
        return -np.inf
    rho = stats.t.ppf(1 - 0.05/2, G_new - 1) ** 2 * (G_new / (G_new - 1)) * ((n_new - 1) / (n_new - P))
    return (-beta1) ** 2 - rho * v


def search_recursive_greedy(H, psi, c, a, tau):
    h_g = np.array([np.linalg.norm(H[g], 2) for g in range(len(H))])
    cand = [g for g in range(len(H)) if h_g[g] >= tau]
    cand.sort(key=lambda g: -h_g[g])
    mask = []
    trials = accepts = 0
    for _ in range(len(cand)):
        best_g, best_b = None, float('inf')
        for g in cand:
            if g in mask:
                continue
            trials += 1
            b = beta1_of(mask + [g], H, psi, a)
            if b < best_b:
                best_b, best_g = b, g
        if best_g is None:
            break
        mask.append(best_g)
        accepts += 1
        if margin_of(mask, H, psi, c, a) > 0:
            return mask, True, trials, accepts
    return mask, False, trials, accepts


def exhaustive_min_sig(H, psi, c, a, G):
    best = None
    for bits in range(1 << G):
        mask = [g for g in range(G) if (bits >> g) & 1]
        if best is not None and len(mask) >= best:
            continue
        if margin_of(mask, H, psi, c, a) > 0:
            best = len(mask)
    return best


print(f"{'G':>3} {'tau':>5} | {'sig recall':>10} {'found size':>10} "
      f"{'true min':>8} {'paired ratio':>12} {'margin eval(us)':>14} {'beta eval(us)':>12}")
for G in [15, 30]:
    for tau in [0.5, 0.1]:
        found = 0
        sizes = []
        ratios = []
        tms = []
        t_margin = 0.0
        t_beta = 0.0
        for _ in range(200):
            H, psi, c, a = build_instance(G)
            t0 = time.perf_counter()
            mask, ok, _, _ = search_recursive_greedy(H, psi, c, a, tau)
            t_margin += time.perf_counter() - t0
            if ok:
                found += 1
                sizes.append(len(mask))
            if G == 15:
                t1 = time.perf_counter()
                opt = exhaustive_min_sig(H, psi, c, a, G)
                t_beta += time.perf_counter() - t1
                tms.append(opt)
                if ok:
                    ratios.append(len(mask) / opt)
        print(f"{G:>3} {tau:>5.2f} | {found/200*100:>9.0f}% "
              f"{np.mean(sizes) if sizes else 0:>10.2f} "
              f"{int(np.median(tms)) if tms else 0:>8} "
              f"{np.mean(ratios) if ratios else 0:>12.3f} "
              f"{t_margin/200*1e6:>14.1f} {t_beta/200*1e6 if G==15 else 0:>12.1f}")
