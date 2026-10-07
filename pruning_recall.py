"""3.2: candidate-pruning recall — can pruning miss a minimal deletion set made of
individually low-leverage clusters whose joint interaction flips the target?

Whitened coordinates (M = I), p = 4. A planted minimal deletion set of k clusters,
each with leverage h_low, flips the target only through their joint interaction.
We scan h_low and the pruning threshold tau and report whether the pruned search
space contains the planted set (recall).
"""
import os
import numpy as np

p = 4
G = 40
B = 4.0                      # full-data beta_1 = B/2 = 2

e1 = np.zeros(p); e1[0] = 1.0


def planted_instance(k, h_low):
    """k clusters each with H_g = h_low e1 e1^T; deleting all k flips beta_1."""
    # full-data: H = 0, Q = I, beta_1 = B/2
    # delete all k: H = k*h_low e1 e1^T, Q = I - k*h_low e1e1^T
    # shift beta_1 = - (Q^{-1} r)_1, need shift > B/2
    # with score sum r_1 = s, shift = s / (1 - k*h_low); need s > (B/2)(1 - k*h_low)
    denom = 1.0 - k * h_low
    s = (B / 2) * denom + 0.2          # total score so the joint flip holds
    # distribute s over k clusters
    score_each = s / k
    h_g = np.zeros(G)
    h_g[:k] = h_low
    return h_g, score_each, k


def pruned_contains(h_g, tau, k):
    """Does the pruned search space (clusters with h_g >= tau) contain all k planted?"""
    pruned = np.where(h_g >= tau)[0]
    return set(range(k)).issubset(set(pruned.tolist()))


print(f"{'k':>2} {'h_low':>6} {'tau':>5} | recall")
for k in [2, 3, 4]:
    for h_low in [0.05, 0.1, 0.2, 0.3, 0.4]:
        for tau in [0.5, 0.3, 0.1]:
            h_g, _, _ = planted_instance(k, h_low)
            ok = pruned_contains(h_g, tau, k)
            if tau in (0.5, 0.1):  # report only the two extremes to keep the table short
                print(f"{k:>2} {h_low:>6.2f} {tau:>5.2f} | {'contained' if ok else 'MISSED'}")
