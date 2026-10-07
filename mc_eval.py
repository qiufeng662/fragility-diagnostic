"""Monte-Carlo method evaluation on randomized realizable whitened instances.

Random rotation + planted k=3 minimal deletion set of low individual leverage +
background chosen so sum_g H_g = I exactly.  Scores are constructed to lie in
each cluster's Gram column space (planted clusters are rank-1 along the target
direction, background clusters have full-rank Gram), so the instance is an
OLS-realizable whitened design; no off-direction score noise is added (randomness
comes from the random rotation).  The AMIP baseline ranks by first-order
influence along the target direction a^T psi_g.  For G=15 the exhaustive optimum
is computed on the SAME instance, so best/opt is a per-instance paired ratio.

The flip criterion is the sign reversal beta_1 < 0 (a simplified demonstration,
not the full CR1 significance margin), and the cost is the cluster count (a
simplified counting convention; the paper's minimal-row problem uses sum c_g).

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
    e1 = R[:, 0]                     # the target direction a
    H = np.zeros((G, P, P))
    Dp = np.zeros((P, P)); Dp[0, 0] = H_LOW / K
    for g in range(K):
        H[g] = R @ Dp @ R.T          # rank-1 along e1 (target direction)
    Db = np.zeros((P, P))
    Db[0, 0] = (1.0 - H_LOW) / (G - K)
    for d in range(1, P):
        Db[d, d] = 1.0 / (G - K)
    for g in range(K, G):
        H[g] = R @ Db @ R.T          # full-rank background Gram
    # planted scores: invert so that deleting the K planted clusters flips beta_1
    HD = H[:K].sum(axis=0)
    Qdel = np.eye(P) - HD
    r = B * (Qdel @ e1)
    psi = np.zeros((G, P))
    psi[:K] = r[None, :] / K          # parallel to e1 -> in the rank-1 column space
    psi[K:] = -r[None, :] / (G - K)   # background: full-rank Gram, any direction OK
    # cluster sizes: at least the design rank P, so n_g >= rank(H_g)
    c = rng.integers(P, 21, size=G).astype(float)
    return H, psi, c, e1


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
    """AMIP-style: rank by first-order influence along the target direction
    a^T psi_g (signed toward the flip), then evaluate prefixes exactly."""
    order = sorted(range(len(H)), key=lambda g: -(e1 @ psi[g]))
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
      f"{'pruned size':>11} {'AMIP size':>9} {'true min':>8} {'paired ratio':>12}")
for G in [15, 30]:
    for tau in [0.5, 0.3, 0.1]:
        found = 0
        found_a = 0
        sizes = []
        sizes_a = []
        ratios = []
        tms = []
        for _ in range(200):
            H, psi, c, e1 = build_instance(G)
            mask, ok = search_recursive_greedy(H, psi, e1, tau)
            if ok:
                found += 1
                sizes.append(len(mask))
            mask_a, ok_a = search_amip(H, psi, e1)
            if ok_a:
                found_a += 1
                sizes_a.append(len(mask_a))
            if G == 15:
                opt = exhaustive_min(H, psi, e1, G)   # SAME instance
                tms.append(opt)
                if ok:
                    ratios.append(len(mask) / opt)
        print(f"{G:>3} {tau:>5.2f} | {found/200*100:>12.0f}% "
              f"{found_a/200*100:>10.0f}% {np.mean(sizes) if sizes else 0:>11.2f} "
              f"{np.mean(sizes_a) if sizes_a else 0:>9.2f} "
              f"{int(np.median(tms)) if tms else 0:>8} "
              f"{np.mean(ratios) if ratios else 0:>12.3f}")
