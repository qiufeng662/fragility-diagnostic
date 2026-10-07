"""Monte-Carlo method evaluation on randomized realizable whitened instances.

Random rotation + planted k=3 minimal deletion set of low individual leverage +
background so sum_g H_g = I.  Compares the recursive-greedy search (Algorithm 2)
against an AMIP-style first-order-influence ordering and an exhaustive optimum
(G = 15), reporting recall, found size and the optimality gap.  The flip
criterion is the sign reversal beta_1 < 0 (simplified demo, not the full CR1
margin).

Run:  python mc_eval.py
"""
import numpy as np

rng = np.random.default_rng(20261005)
P = 6
B = 4.0
K = 3
H_LOW = 0.6        # each planted cluster gets H_LOW/K = 0.2


def build_instance(G):
    A = rng.normal(size=(P, P))
    R, _ = np.linalg.qr(A)
    e1 = R[:, 0]
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
    r = B * (Qdel @ e1)
    psi = np.zeros((G, P))
    psi[:K] = r[None, :] / K
    psi[K:] = -r[None, :] / (G - K)
    noise = rng.normal(0, 0.01, (G, P))
    noise -= noise.mean(axis=0)
    psi = psi + noise
    return H, psi, e1


def beta1_of(mask, H, psi, e1):
    if not mask:
        return B / 2.0
    HD = H[mask].sum(axis=0)
    Q = np.eye(P) - HD
    r = psi[mask].sum(axis=0)
    return B / 2.0 - e1 @ np.linalg.solve(Q, r)


def flip(mask, H, psi, e1):
    return beta1_of(mask, H, psi, e1) < 0


def search_recursive_greedy(H, psi, e1, tau):
    h_g = np.array([np.linalg.norm(H[g], 2) for g in range(len(H))])
    cand = [g for g in range(len(H)) if h_g[g] >= tau]
    cand.sort(key=lambda g: -h_g[g])
    mask = []
    for _ in range(len(cand)):
        best_g, best_b = None, float('inf')
        for g in cand:
            if g in mask:
                continue
            b = beta1_of(mask + [g], H, psi, e1)
            if b < best_b:
                best_b, best_g = b, g
        if best_g is None:
            break
        mask.append(best_g)
        if flip(mask, H, psi, e1):
            return mask, True
    return mask, False


def search_amip(H, psi, e1):
    """AMIP-style: rank by first-order influence |psi_g[0]|, evaluate prefixes exactly."""
    order = sorted(range(len(H)), key=lambda g: -abs(psi[g, 0]))
    mask = []
    for g in order:
        mask = mask + [g]
        if flip(mask, H, psi, e1):
            return mask, True
    return mask, False


def exhaustive_min(H, psi, e1, G):
    best = None
    for bits in range(1 << G):
        mask = [g for g in range(G) if (bits >> g) & 1]
        if best is not None and len(mask) >= best:
            continue
        if flip(mask, H, psi, e1):
            best = len(mask)
    return best


print(f"{'G':>3} {'tau':>5} | {'pruned recall':>13} {'AMIP recall':>11} "
      f"{'pruned size':>11} {'AMIP size':>9} {'true min':>8}")
for G in [15, 30]:
    for tau in [0.5, 0.3, 0.1]:
        found = miss = 0
        found_a = 0
        sizes = []
        sizes_a = []
        for _ in range(200):
            H, psi, e1 = build_instance(G)
            mask, ok = search_recursive_greedy(H, psi, e1, tau)
            if ok:
                found += 1
                sizes.append(len(mask))
            else:
                miss += 1
            mask_a, ok_a = search_amip(H, psi, e1)
            if ok_a:
                found_a += 1
                sizes_a.append(len(mask_a))
        tm = None
        if G == 15:
            tms = []
            for _ in range(50):
                H, psi, e1 = build_instance(G)
                tms.append(exhaustive_min(H, psi, e1, G))
            tm = int(np.median(tms))
        print(f"{G:>3} {tau:>5.2f} | {found/200*100:>12.0f}% "
              f"{found_a/200*100:>10.0f}% {np.mean(sizes) if sizes else 0:>11.1f} "
              f"{np.mean(sizes_a) if sizes_a else 0:>9.1f} {str(tm):>8}")
