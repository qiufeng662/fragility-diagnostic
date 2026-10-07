"""Monte-Carlo method evaluation (random DGP, whitened coordinates).

Responds to both reviewers' request for a real statistical/computational evaluation:
random leverage concentration, planted minimal deletion sets, candidate pruning,
AMIP baseline, exhaustive true minimum on a small scale, recall and optimality gap.
"""
import time
import numpy as np
from scipy import stats

rng = np.random.default_rng(20261005)
p = 6
e1 = np.zeros(p); e1[0] = 1.0
B = 4.0


def build_instance(G, k, concentration, tau_plant):
    """Whitenened coordinates M = I. Plant k clusters whose joint deletion flips
    the target, with individually low leverage h_g = tau_plant (< tau)."""
    # leverage distribution
    if concentration == "concentrated":
        # one dominant cluster (h ~ 0.8), rest small
        h = np.zeros(G)
        h[0] = 0.8
        h[1:] = (1.0 - 0.8) / (G - 1)
    else:  # dispersed
        h = np.full(G, 0.9 / G)
    # the planted minimal set occupies the first k clusters; give them tau_plant
    h[:k] = tau_plant
    h = h * (0.95 / h.sum())  # renormalize so sum < 1
    # scores: joint deletion of the first k clusters flips beta_1
    denom = 1.0 - h[:k].sum()
    s_total = (B / 2) * denom + 0.2
    psi = np.zeros((G, p))
    psi[:k, 0] = s_total / k
    # background score so full-data is consistent (sum psi = 0)
    psi[k:, 0] = -psi[:k, 0].sum() / (G - k)
    return h, psi


def flip_del(mask, h, psi):
    """Exact whitened post-deletion beta_1 shift; flip means beta_1 < 0."""
    H = sum(h[g] for g in mask)
    r1 = psi[mask, 0].sum()
    beta1 = B / 2 - r1 / (1.0 - H)
    return beta1 < 0


def search_pruned(h, psi, tau):
    """Greedy pruned search: add clusters in descending h (only h >= tau), exact eval."""
    cand = [g for g in range(len(h)) if h[g] >= tau]
    cand.sort(key=lambda g: -h[g])
    mask = []
    for g in cand:
        if flip_del(mask + [g], h, psi):
            return mask + [g], True
        # also try joint addition
        trial = mask + [g]
        if flip_del(trial, h, psi):
            return trial, True
        mask = trial
    return mask, False


def search_amip(h, psi):
    """AMIP: order by first-order influence |psi_1| (leverage ignored), greedy."""
    cand = sorted(range(len(h)), key=lambda g: -abs(psi[g, 0]))
    mask = []
    for g in cand:
        trial = mask + [g]
        if flip_del(trial, h, psi):
            return trial, True
        mask = trial
    return mask, False


def exhaustive_min(h, psi, G):
    """True minimum deletion size (G small)."""
    best = None
    for bits in range(1 << G):
        mask = [g for g in range(G) if (bits >> g) & 1]
        if best is not None and len(mask) >= best:
            continue
        if flip_del(mask, h, psi):
            best = len(mask)
    return best


print(f"{'concen':>10} {'G':>3} {'k':>2} {'tau_plant':>8} {'tau':>5} | "
      f"{'pruned found':>12} {'pruned size':>11} {'AMIP size':>9} {'true min':>8} | "
      f"{'gap':>5}")
for concentration in ["concentrated", "dispersed"]:
    for G, k, tau_plant in [(15, 3, 0.3), (15, 3, 0.15), (30, 3, 0.2)]:
        for tau in [0.5, 0.3, 0.1]:
            found_n = miss_n = 0
            gap_sum = 0.0
            amip_size = []
            t0 = time.perf_counter()
            for rep in range(200):
                h, psi = build_instance(G, k, concentration, tau_plant)
                mask_p, found = search_pruned(h, psi, tau)
                mask_a, _ = search_amip(h, psi)
                amip_size.append(len(mask_a))
                if found:
                    found_n += 1
                    gap_sum += len(mask_p) / k
                else:
                    miss_n += 1
            dt = time.perf_counter() - t0
            # exhaustive only for G=15
            true_min = None
            if G == 15:
                tm = []
                for rep in range(200):
                    h, psi = build_instance(G, k, concentration, tau_plant)
                    tm.append(exhaustive_min(h, psi, G))
                true_min = int(np.median(tm))
            print(f"{concentration:>10} {G:>3} {k:>2} {tau_plant:>8.2f} {tau:>5.2f} | "
                  f"{found_n/200*100:>11.0f}% {gap_sum/max(found_n,1):>11.2f} "
                  f"{np.median(amip_size):>9.1f} {str(true_min):>8} | "
                  f"{gap_sum/max(found_n,1):>5.2f}")
