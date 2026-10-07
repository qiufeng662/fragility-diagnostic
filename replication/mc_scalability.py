"""Scalability benchmark for Algorithm 2 (pruned search) -- a small table.

Reports candidate-set size, visited nodes, runtime, recall and (for G=15)
the ratio to the exhaustive optimum, for G in {15, 30, 50} and threshold
tau in {0.5, 0.1}, under the concentrated-leverage DGP with a planted k=3
minimal deletion set of individually low leverage (tau_plant = 0.2).

Run:  python mc_scalability.py
"""
import time
import numpy as np

rng = np.random.default_rng(20261005)
p = 6
B = 4.0
K = 3            # planted minimal deletion set
TAU_PLANT = 0.2


def build_instance(G):
    """Dispersed leverage: G-3 background clusters of size 0.9/G, and a planted
    k=3 minimal deletion set of individually low leverage TAU_PLANT."""
    h = np.full(G, 0.9 / G)
    h[:K] = TAU_PLANT
    h = h * (0.95 / h.sum())
    denom = 1.0 - h[:K].sum()
    s_total = (B / 2) * denom + 0.2
    psi = np.zeros((G, p))
    psi[:K, 0] = s_total / K
    psi[K:, 0] = -psi[:K, 0].sum() / (G - K)
    return h, psi


def flip_del(mask, h, psi):
    H = sum(h[g] for g in mask)
    r1 = psi[mask, 0].sum()
    beta1 = B / 2 - r1 / (1.0 - H)
    return beta1 < 0


def beta1_of(mask, h, psi):
    H = sum(h[g] for g in mask)
    r1 = psi[mask, 0].sum()
    return B / 2 - r1 / (1.0 - H)


def search_recursive_greedy(h, psi, tau):
    """Algorithm 2: each step tries every remaining candidate and keeps the one
    giving the largest increase toward the flip (smallest beta1)."""
    cand = [g for g in range(len(h)) if h[g] >= tau]
    cand.sort(key=lambda g: -h[g])
    mask = []
    trials = accepts = 0
    for _ in range(len(cand)):
        best_g, best_beta = None, float('inf')
        for g in cand:
            if g in mask:
                continue
            trials += 1
            b = beta1_of(mask + [g], h, psi)
            if b < best_beta:
                best_beta, best_g = b, g
        if best_g is None:
            break
        mask.append(best_g)
        accepts += 1
        if flip_del(mask, h, psi):
            return mask, True, trials, accepts
    return mask, False, trials, accepts


def exhaustive_min(h, psi, G):
    best = None
    for bits in range(1 << G):
        mask = [g for g in range(G) if (bits >> g) & 1]
        if best is not None and len(mask) >= best:
            continue
        if flip_del(mask, h, psi):
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
        h, psi = build_instance(G)
        cand = [g for g in range(G) if h[g] >= tau]
        cand_sizes.append(len(cand))
        mask, ok, trials, accepts = search_recursive_greedy(h, psi, tau)
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
        opts = [exhaustive_min(*build_instance(G), G) for _ in range(50)]
        opt = int(np.median(opts))
    recall = found / 200
    best = f"{np.mean(sizes):.1f}" if sizes else "--"
    ratio = f"{np.mean(sizes)/opt:.2f}" if (sizes and opt) else "--"
    print(f"{G:>3} {tau:>5.2f} {int(np.median(cand_sizes)):>5} "
          f"{trials_sum/200:>7.1f} {accepts_sum/200:>8.1f} {dt:>11.2f} "
          f"{recall:>7.2f} {best:>10} {ratio:>9}")

# exhaustive cost at G=15 (2^15 = 32768 masks) for contrast
t0 = time.perf_counter()
h, psi = build_instance(15)
exhaustive_min(h, psi, 15)
dt = (time.perf_counter() - t0) * 1000
print(f"\n[contrast] G=15 exhaustive enumeration (2^15 masks): {dt:.1f} ms "
      f"per instance; G=30 is 2^30 and G=50 is 2^50 (infeasible).")
