"""Exact minimal-deletion search (branch-and-bound) for the fragility diagnostic.

Given the determinant/adjugate precompute ``certify.det_adj_precompute`` and a
target, returns a minimum-cost cluster-deletion set D satisfying q_j(D) >= 0 and
m_j(D) >= 0 (sign flip + significance).  This is the paper's diagnostic quantity
C*: the number of rows whose deletion flips the target — i.e. how fragile the
conclusion is.  The search is complete (it returns the true optimum), hence
worst-case exponential, matching the NP-hardness result; branch ordering by
descending relative leverage plus a cost-bound prune keep it practical on the
panel sizes of the empirical section.

This is a *defensive diagnostic*, not a solver: its output is a fragility report
(minimum rows to flip a claim), which is exactly what a researcher needs to audit
whether a conclusion is riding on a handful of clusters.
"""
from __future__ import annotations

import numpy as np

if __package__ in (None, ""):
    # Direct execution (python optmask/search.py): re-anchor the package so the
    # relative imports below resolve as they do under python -m optmask.search.
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "optmask"

from .certify import det_adj_margin
from .diagnostics import _exact_q


def _feasible(pre, target, deleted):
    """q_j > 0 and m_j > 0 for the deletion set (sign flip + significance)."""
    q = _exact_q(pre, target, deleted)
    if q is None or not (q > 0):
        return False
    m = det_adj_margin(pre, target, deleted)
    return m is not None and m > 0


def greedy_upper(pre, target, K_max=None):
    """Greedy upper bound: add clusters in descending relative leverage until feasible."""
    G = pre["G"]
    c = pre["c"]
    h_g = np.array([np.linalg.norm(pre["H"][g], 2) for g in range(G)])
    order = np.argsort(-h_g)
    mask = []
    for g in order:
        mask.append(int(g))
        if K_max is not None and float(c[mask].sum()) > K_max:
            break
        if _feasible(pre, target, mask):
            return float(c[mask].sum()), list(mask)
    return float("inf"), []


def minimal_deletion(pre, target, K_max=None, node_limit=2_000_000):
    """Branch-and-bound minimum-cost deletion set.

    Returns ``(cost, deleted_clusters, completed)``.  ``cost = inf`` when no
    feasible deletion within ``K_max`` rows is found.  ``completed`` is True only
    when the search exhausted the tree (or was provably pruned by the exact cost
    bound); only then is the returned set a certified global optimum.  If the
    node limit is hit, ``completed`` is False and the returned set is only an
    incumbent (a feasible upper bound, not necessarily optimal).
    """
    G = pre["G"]
    c = pre["c"]
    h_g = np.array([np.linalg.norm(pre["H"][g], 2) for g in range(G)])
    order = np.argsort(-h_g)          # explore high-leverage clusters first

    best_cost, best_mask = greedy_upper(pre, target, K_max)
    best = [best_cost, best_mask]     # mutable closure state
    nodes = [0]
    completed = [True]

    def dfs(pos, mask, cost):
        if nodes[0] > node_limit:
            completed[0] = False
            return
        nodes[0] += 1
        if cost >= best[0]:
            return                    # exact cost bound: no better solution below
        if pos == G:
            if _feasible(pre, target, mask):
                best[0] = cost
                best[1] = list(mask)
            return
        g = int(order[pos])
        dfs(pos + 1, mask, cost)      # keep g
        new_cost = cost + c[g]
        if K_max is None or new_cost <= K_max:
            mask.append(g)
            dfs(pos + 1, mask, new_cost)  # delete g (within budget)
            mask.pop()

    dfs(0, [], 0.0)
    return best[0], best[1], completed[0]


if __name__ == "__main__":
    # Whitened-coordinate self-test (M = I, p = 4, G = 4): deleting cluster 0
    # flips the target with q = +B/2 and m = B^2/4; no other deletion does.
    from types import SimpleNamespace
    from .contract import Target

    p, G = 4, 6
    B, h = 4.0, 0.9
    a = (1.0 - h) / (G - 1)
    e1 = np.zeros(p); e1[0] = 1.0

    H = np.zeros((G, p, p))
    psi = np.zeros((G, p))
    H[0] = h * np.outer(e1, e1);      psi[0] = (1.0 - h) * B * e1
    for g in range(1, G):
        H[g] = a * np.outer(e1, e1);  psi[g] = -a * B * e1

    pre = {
        "H": H, "psi_tilde": psi, "c": np.ones(G), "k": p, "G": G,
        "Ps": np.eye(p), "n0": float(G), "G0": float(G), "n_entities": 0,
        "fit": SimpleNamespace(names=["x1", "x2", "x3", "x4"],
                               beta=np.array([B / 2.0, 0, 0, 0])),
    }
    target = Target(column="x1", sign=-1, alpha=0.05)

    cost, mask, completed = minimal_deletion(pre, target)
    print(f"minimal deletion: cost={cost}, clusters={mask}, completed={completed} "
          f"(expected cost=1, clusters=[0])")
    assert cost == 1.0 and mask == [0] and completed, "self-test failed"

    # K_max = 0 must find no feasible deletion (budget constrains the DFS, too)
    c0, m0, done0 = minimal_deletion(pre, target, K_max=0)
    assert c0 == float("inf") and done0, f"K_max=0 should be infeasible, got cost={c0}"
    # a tiny node_limit must report the search as incomplete
    c1, m1, done1 = minimal_deletion(pre, target, node_limit=0)
    assert done1 is False, "node_limit=0 should report incomplete"
    print("self-test passed (budget and completion status checked)")
