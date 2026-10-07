"""Method comparison on randomized clustered regressions (no planted set).

G=12 clusters, unequal sizes; focal beta_1 = 2 > 0, target sign=-1 (beta_1 < 0
AND CR1 margin > 0).  Three procedures run on the SAME instance, SAME
whole-cluster deletion rule, SAME row cost, and SAME CR1 convention:
  - first-order influence prefix, evaluated exactly;
  - recursive-greedy search (Algorithm 2);
  - exhaustive enumeration over 2^12 masks (small-scale ground truth).

Both search methods share the SAME candidate pool (all G clusters) and run to
completion: each stops when it finds a feasible set, or when its search over the
candidate pool is exhausted.  "Evaluations" is the number of exact post-deletion
coefficient evaluations (one Cholesky solve each); the significance margin is
computed only for masks whose coefficient passes the sign condition, and its
cost is captured in wall-clock time.

Reports, per setting (100 replications, seed 20261005): infeasible count,
feasible count, found fraction over feasible instances, mean found-cost /
exhaustive-optimum ratio, mean evaluations, mean wall-clock per feasible
instance, and the miss breakdown (feasible instances not found).

Run:  python mc_method_compare.py
Writes: data/mc_method_compare_results.csv
"""
import time
import csv
import numpy as np
from scipy import linalg as sla
from scipy import stats

rng = np.random.default_rng(20261005)
P = 5
B = 4.0
G = 12
REPS = 100
ALPHA = 0.05


def build_instance(concentration, sigma):
    A = rng.normal(size=(P, P))
    R, _ = np.linalg.qr(A)
    a = R[:, 0]
    U = rng.normal(size=(G, P))
    S = U.T @ U
    ev, Q = np.linalg.eigh(S)
    Sinvsqrt = Q @ np.diag(1.0 / np.sqrt(np.clip(ev, 1e-12, None))) @ Q.T
    V = U @ Sinvsqrt
    H = np.array([np.outer(V[g], V[g]) for g in range(G)])
    if concentration == "concentrated":
        h1 = 0.5
        H[0] = h1 * np.outer(a, a)
        rest = (np.eye(P) - H[0]) / (G - 1)
        for g in range(1, G):
            H[g] = rest
    psi = rng.normal(0, sigma, (G, P))
    psi -= psi.mean(axis=0)
    c = rng.integers(5, 21, size=G).astype(float)
    return H, psi, c, a


def _chol(Q):
    try:
        return np.linalg.cholesky(Q)
    except np.linalg.LinAlgError:
        return None


def make_evaluator(H, psi, c, a):
    evals = [0]

    def beta1(mask):
        evals[0] += 1
        if not mask:
            return B / 2.0
        Q = np.eye(P) - H[mask].sum(axis=0)
        L = _chol(Q)
        if L is None:
            return None
        r = psi[mask].sum(axis=0)
        z = sla.solve_triangular(L.T, sla.solve_triangular(L, r, lower=True), lower=False)
        return B / 2.0 - a @ z

    def margin(mask):
        Q = np.eye(P) - H[mask].sum(axis=0)
        L = _chol(Q)
        if L is None:
            return None
        r = psi[mask].sum(axis=0)
        z = sla.solve_triangular(L.T, sla.solve_triangular(L, r, lower=True), lower=False)
        beta1 = B / 2.0 - a @ z
        kept = [g for g in range(len(H)) if g not in mask]
        G_new = len(kept)
        n_new = sum(c[g] for g in kept)
        if G_new <= 1 or n_new <= P:
            return None
        v = 0.0
        for g in kept:
            psig = psi[g] + H[g] @ z
            zg = sla.solve_triangular(L.T, sla.solve_triangular(L, psig, lower=True), lower=False)
            v += (a @ zg) ** 2
        rho = stats.t.ppf(1 - ALPHA / 2, G_new - 1) ** 2 * (G_new / (G_new - 1)) * ((n_new - 1) / (n_new - P))
        return (-beta1) ** 2 - rho * v

    def feasible(mask):
        b = beta1(mask)
        if b is None or b >= 0:
            return False
        m = margin(mask)
        return m is not None and m > 0

    return beta1, margin, feasible, evals


def cost(mask, c):
    return sum(c[g] for g in mask)


def exhaustive_min(H, psi, c, a, G):
    best = None
    for bits in range(1 << G):
        mask = [g for g in range(G) if (bits >> g) & 1]
        if best is not None and cost(mask, c) >= best:
            continue
        b = beta1_of(mask, H, psi, a)
        if b is None or b >= 0:
            continue
        m = margin_of(mask, H, psi, c, a)
        if m is not None and m > 0:
            best = cost(mask, c)
    return best


def beta1_of(mask, H, psi, a):
    if not mask:
        return B / 2.0
    Q = np.eye(P) - H[mask].sum(axis=0)
    L = _chol(Q)
    if L is None:
        return None
    r = psi[mask].sum(axis=0)
    z = sla.solve_triangular(L.T, sla.solve_triangular(L, r, lower=True), lower=False)
    return B / 2.0 - a @ z


def margin_of(mask, H, psi, c, a):
    Q = np.eye(P) - H[mask].sum(axis=0)
    L = _chol(Q)
    if L is None:
        return None
    r = psi[mask].sum(axis=0)
    z = sla.solve_triangular(L.T, sla.solve_triangular(L, r, lower=True), lower=False)
    beta1 = B / 2.0 - a @ z
    kept = [g for g in range(len(H)) if g not in mask]
    G_new = len(kept)
    n_new = sum(c[g] for g in kept)
    if G_new <= 1 or n_new <= P:
        return None
    v = 0.0
    for g in kept:
        psig = psi[g] + H[g] @ z
        zg = sla.solve_triangular(L.T, sla.solve_triangular(L, psig, lower=True), lower=False)
        v += (a @ zg) ** 2
    rho = stats.t.ppf(1 - ALPHA / 2, G_new - 1) ** 2 * (G_new / (G_new - 1)) * ((n_new - 1) / (n_new - P))
    return (-beta1) ** 2 - rho * v


def first_order_prefix(H, psi, c, a, beta1, feasible, evals):
    order = sorted(range(len(H)), key=lambda g: -(a @ psi[g]))
    mask = []
    for g in order:
        mask = mask + [g]
        if feasible(mask):
            return mask, cost(mask, c), evals[0], 'found'
    return mask, None, evals[0], 'pool'


def recursive_search(H, psi, c, a, beta1, feasible, evals):
    cand = list(range(len(H)))
    mask = []
    for _ in range(len(cand)):
        best_g, best_b = None, float('inf')
        for g in cand:
            if g in mask:
                continue
            b = beta1(mask + [g])
            if b is not None and b < best_b:
                best_b, best_g = b, g
        if best_g is None:
            return mask, None, evals[0], 'stall'
        mask.append(best_g)
        if feasible(mask):
            return mask, cost(mask, c), evals[0], 'found'
    return mask, None, evals[0], 'pool'


rows = []
header = ["setting", "infeasible", "feasible", "fo_found", "rg_found",
          "fo_cost_opt", "rg_cost_opt", "fo_evals", "rg_evals",
          "fo_ms", "rg_ms", "fo_miss_pool", "rg_miss_pool", "rg_miss_stall"]

print(f"{'setting':>20} {'infeas':>6} {'feas':>5} {'FO found':>9} {'RG found':>9} "
      f"{'FO c/opt':>9} {'RG c/opt':>9} {'FO evals':>9} {'RG evals':>9} "
      f"{'FO ms':>8} {'RG ms':>8} | FO miss RG miss(pool/stall)")
for conc in ["concentrated", "dispersed"]:
    for sigma in [0.5, 1.0]:
        label = f"{conc}, sigma={sigma}"
        feasible_n = 0
        infeasible_n = 0
        fo_found = 0
        rg_found = 0
        fo_ratios, rg_ratios, fo_evals, rg_evals = [], [], [], []
        t_fo = 0.0
        t_rg = 0.0
        fo_miss = 0
        rg_miss_pool = 0
        rg_miss_stall = 0
        for _ in range(REPS):
            H, psi, c, a = build_instance(conc, sigma)
            opt = exhaustive_min(H, psi, c, a, G)
            if opt is None:
                infeasible_n += 1
                continue
            feasible_n += 1
            beta1, _, feasible, evals = make_evaluator(H, psi, c, a)
            t0 = time.perf_counter()
            _, fo_cost, fo_e, fo_stop = first_order_prefix(H, psi, c, a, beta1, feasible, evals)
            t_fo += time.perf_counter() - t0
            beta1, _, feasible, evals = make_evaluator(H, psi, c, a)
            t0 = time.perf_counter()
            _, rg_cost, rg_e, rg_stop = recursive_search(H, psi, c, a, beta1, feasible, evals)
            t_rg += time.perf_counter() - t0
            fo_evals.append(fo_e)
            rg_evals.append(rg_e)
            if fo_cost is not None:
                fo_found += 1
                fo_ratios.append(fo_cost / opt)
            else:
                fo_miss += 1
            if rg_cost is not None:
                rg_found += 1
                rg_ratios.append(rg_cost / opt)
            else:
                if rg_stop == 'stall':
                    rg_miss_stall += 1
                else:
                    rg_miss_pool += 1
        fo_frac = fo_found / feasible_n if feasible_n else 0.0
        rg_frac = rg_found / feasible_n if feasible_n else 0.0
        fo_ratio = np.mean(fo_ratios) if fo_ratios else float('nan')
        rg_ratio = np.mean(rg_ratios) if rg_ratios else float('nan')
        fo_e = np.mean(fo_evals) if fo_evals else float('nan')
        rg_e = np.mean(rg_evals) if rg_evals else float('nan')
        fo_ms = t_fo / feasible_n * 1e3 if feasible_n else 0.0
        rg_ms = t_rg / feasible_n * 1e3 if feasible_n else 0.0
        rows.append([label, infeasible_n, feasible_n, fo_found, rg_found,
                     round(fo_ratio, 4), round(rg_ratio, 4),
                     round(fo_e, 2), round(rg_e, 2),
                     round(fo_ms, 3), round(rg_ms, 3),
                     fo_miss, rg_miss_pool, rg_miss_stall])
        print(f"{label:>20} {infeasible_n:>6} {feasible_n:>5} "
              f"{fo_frac:>9.2f} {rg_frac:>9.2f} "
              f"{fo_ratio:>9.2f} {rg_ratio:>9.2f} "
              f"{fo_e:>9.1f} {rg_e:>9.1f} "
              f"{fo_ms:>8.2f} {rg_ms:>8.2f} | "
              f"FO miss {fo_miss}  RG miss({rg_miss_pool}/{rg_miss_stall})")

with open("data/mc_method_compare_results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(header)
    w.writerows(rows)
print("wrote data/mc_method_compare_results.csv")
