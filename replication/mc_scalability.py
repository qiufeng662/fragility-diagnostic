"""Scalability benchmark for Algorithm 2 (recursive-greedy search).

Randomized, realizable whitened DGP: a random rotation, a planted k=3 minimal
deletion set of individually low leverage, and background clusters such that
sum_g H_g = I exactly.  Each replication draws a fresh instance via the RNG.
The flip criterion is the sign reversal beta_1 < 0 (a simplified demo, not the
full CR1 significance margin; see the paper text).

Run:  python mc_scalability.py
"""
import time
import numpy as np

rng = np.random.default_rng(20261005)
P = 6
B = 4.0
K = 3            # planted minimal deletion set
H_LOW = 0.6      # total planted leverage; each planted cluster gets H_LOW/K = 0.2


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
    c = rng.integers(5, 21, size=G).astype(float)
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
    """Algorithm 2: each step tries every remaining candidate and keeps the one
    giving the largest decrease toward the flip (smallest beta1)."""
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
            b = beta1_of(mask + [g], H, psi, e1)
            if b < best_b:
                best_b, best_g = b, g
        if best_g is None:
            break
        mask.append(best_g)
        accepts += 1
        if flip(mask, H, psi, e1):
            return mask, True, trials, accepts
    return mask, False, trials, accepts


def exhaustive_min(H, psi, e1, G):
    best = None
    for bits in range(1 << G):
        mask = [g for g in range(G) if (bits >> g) & 1]
        if best is not None and len(mask) >= best:
            continue
        if flip(mask, H, psi, e1):
            best = len(mask)
    return best


print(f"{'G':>3} {'tau':>5} {'cand':>5} {'trials':>7} {'accepts':>8} "
      f"{'runtime(ms)':>11} {'recall':>7} {'best size':>10} {'best/opt':>9}")
for G, tau in [(15, 0.5), (15, 0.1), (30, 0.1), (50, 0.1)]:
    found = miss = 0
    trials_sum = accepts_sum = 0
    sizes = []
    cand_sizes = []
    t0 = time.perf_counter()
    for _ in range(200):
        H, psi, c, e1 = build_instance(G)
        h_g = np.array([np.linalg.norm(H[g], 2) for g in range(G)])
        cand = [g for g in range(G) if h_g[g] >= tau]
        cand_sizes.append(len(cand))
        mask, ok, trials, accepts = search_recursive_greedy(H, psi, e1, tau)
        trials_sum += trials
        accepts_sum += accepts
        if ok:
            found += 1
            sizes.append(len(mask))
        else:
            miss += 1
    dt = (time.perf_counter() - t0) * 1000
    opt = None
    if G == 15:
        opts = []
        for _ in range(50):
            Hx, psix, cx, e1x = build_instance(G)
            opts.append(exhaustive_min(Hx, psix, e1x, G))
        opt = int(np.median(opts))
    recall = found / 200
    best = f"{np.mean(sizes):.1f}" if sizes else "--"
    ratio = f"{np.mean(sizes)/opt:.2f}" if (sizes and opt) else "--"
    print(f"{G:>3} {tau:>5.2f} {int(np.median(cand_sizes)):>5} "
          f"{trials_sum/200:>7.1f} {accepts_sum/200:>8.1f} {dt:>11.2f} "
          f"{recall:>7.2f} {best:>10} {ratio:>9}")
