"""Method comparison on randomized clustered regressions (no planted set).

G=12 clusters, unequal sizes; focal beta_1 = 2 > 0, target sign=-1 (beta_1 < 0
AND CR1 margin > 0).  Three procedures run on the SAME instance, SAME
whole-cluster deletion rule, SAME row cost, and SAME CR1 convention:
  - first-order influence prefix, evaluated exactly;
  - recursive-greedy search (Algorithm 2);
  - exhaustive enumeration over 2^12 masks (small-scale ground truth).

The data-generating process respects the OLS column-space constraint: each
cluster score lies in the column space of its own Gram matrix (psi_g = H_g v_g,
whitened), and the scores sum to zero across clusters (the full-sample first-
order condition), so every instance is realizable by a genuine clustered OLS
design.  Positive definiteness of a retained design is checked by the smallest
eigenvalue of Q = I - sum_g x_g H_g, not by whether a Cholesky factorization
happens to succeed.

Both search methods share the SAME candidate pool (all G clusters); each runs
to its own stopping condition, and the reported evaluations and wall-clock are
what each actually uses.  The recursive search stops when a feasible set is
found, when no remaining candidate improves the target coefficient, when no
remaining candidate admits a full-rank retained design, or when the candidate
pool is exhausted (these are recorded separately).

Run:  python mc_method_compare.py
Writes: data/mc_method_compare_results.csv
"""
import time
import csv
import numpy as np
from scipy import stats

rng = np.random.default_rng(20261005)
P = 5
B = 2.0
G = 12
REPS = 100
ALPHA = 0.05
EIG_TOL = 1e-9


def build_instance(concentration, sigma):
    A = rng.normal(size=(P, P))
    R, _ = np.linalg.qr(A)
    a = R[:, 0]
    U = rng.normal(size=(G, P))
    S = U.T @ U
    ev, Q = np.linalg.eigh(S)
    Sinvsqrt = Q @ np.diag(1.0 / np.sqrt(np.clip(ev, 1e-12, None))) @ Q.T
    V = U @ Sinvsqrt
    H = np.array([np.outer(V[g], V[g]) for g in range(G)])  # sum_g H_g = I
    if concentration == "concentrated":
        h1 = 0.7
        H[0] = h1 * np.outer(a, a)
        rest = (np.eye(P) - H[0]) / (G - 1)
        for g in range(1, G):
            H[g] = rest
        # psi_0 lies along a (the one-dimensional range of H_0); the other
        # clusters have full-rank Gram matrices, and the last cluster absorbs
        # the zero-sum constraint.
        psi = np.zeros((G, P))
        psi[0] = rng.normal(0, sigma) * a
        for g in range(1, G - 1):
            psi[g] = rng.normal(0, sigma, P)
        psi[G - 1] = -(psi[:G - 1].sum(axis=0))
    else:
        # Each H_g is rank one along V_g, so psi_g = c_g V_g; the coefficients
        # are projected onto the zero-sum subspace sum_g c_g V_g = 0.
        cc = rng.normal(0, sigma, G)
        cc = cc - V @ (V.T @ cc)
        psi = np.array([cc[g] * V[g] for g in range(G)])
    c = rng.integers(5, 21, size=G).astype(float)
    return H, psi, c, a


def _solve_z(Q, r):
    """Return Q^{-1} r if Q is positive definite, else None."""
    if np.linalg.eigvalsh(Q).min() <= EIG_TOL:
        return None
    return np.linalg.solve(Q, r)


def make_evaluator(H, psi, c, a):
    evals = [0]

    def beta1(mask):
        evals[0] += 1
        if not mask:
            return B / 2.0
        Q = np.eye(P) - H[mask].sum(axis=0)
        z = _solve_z(Q, psi[mask].sum(axis=0))
        if z is None:
            return None
        return B / 2.0 - a @ z

    def margin(mask):
        Q = np.eye(P) - H[mask].sum(axis=0)
        z = _solve_z(Q, psi[mask].sum(axis=0))
        if z is None:
            return None
        beta1 = B / 2.0 - a @ z
        kept = [g for g in range(len(H)) if g not in mask]
        G_new = len(kept)
        n_new = sum(c[g] for g in kept)
        if G_new <= 1 or n_new <= P:
            return None
        v = 0.0
        for g in kept:
            psig = psi[g] + H[g] @ z
            zg = np.linalg.solve(Q, psig)
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
    z = _solve_z(Q, psi[mask].sum(axis=0))
    if z is None:
        return None
    return B / 2.0 - a @ z


def margin_of(mask, H, psi, c, a):
    Q = np.eye(P) - H[mask].sum(axis=0)
    z = _solve_z(Q, psi[mask].sum(axis=0))
    if z is None:
        return None
    beta1 = B / 2.0 - a @ z
    kept = [g for g in range(len(H)) if g not in mask]
    G_new = len(kept)
    n_new = sum(c[g] for g in kept)
    if G_new <= 1 or n_new <= P:
        return None
    v = 0.0
    for g in kept:
        psig = psi[g] + H[g] @ z
        zg = np.linalg.solve(Q, psig)
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
    current_b = beta1([])
    for _ in range(len(cand)):
        best_g, best_b = None, current_b
        any_valid = False
        for g in cand:
            if g in mask:
                continue
            b = beta1(mask + [g])
            if b is None:
                continue
            any_valid = True
            if b < best_b:
                best_b, best_g = b, g
        if best_g is None:
            return mask, None, evals[0], ('invalid' if not any_valid else 'stall')
        mask.append(best_g)
        current_b = best_b
        if feasible(mask):
            return mask, cost(mask, c), evals[0], 'found'
    return mask, None, evals[0], 'pool'


rows = []
header = ["setting", "infeasible", "feasible", "fo_found", "rg_found",
          "fo_cost_opt", "rg_cost_opt", "fo_evals", "rg_evals",
          "fo_ms", "rg_ms", "fo_miss_pool", "rg_miss_stall",
          "rg_miss_invalid", "rg_miss_pool"]

print(f"{'setting':>20} {'infeas':>6} {'feas':>5} {'FO found':>9} {'RG found':>9} "
      f"{'FO c/opt':>9} {'RG c/opt':>9} {'FO evals':>9} {'RG evals':>9} "
      f"{'FO ms':>8} {'RG ms':>8} | RG miss(stall/invalid/pool)")
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
        rg_miss = {'stall': 0, 'invalid': 0, 'pool': 0}
        for rep in range(REPS):
            H, psi, c, a = build_instance(conc, sigma)
            if rep == 0:
                res = [np.linalg.norm(psi[g] - H[g] @ np.linalg.pinv(H[g]) @ psi[g]) for g in range(G)]
                print(f"  [{label}] colspace residual max = {max(res):.2e}")
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
                rg_miss[rg_stop] += 1
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
                     fo_miss, rg_miss['stall'], rg_miss['invalid'], rg_miss['pool']])
        print(f"{label:>20} {infeasible_n:>6} {feasible_n:>5} "
              f"{fo_frac:>9.2f} {rg_frac:>9.2f} "
              f"{fo_ratio:>9.2f} {rg_ratio:>9.2f} "
              f"{fo_e:>9.1f} {rg_e:>9.1f} "
              f"{fo_ms:>8.2f} {rg_ms:>8.2f} | "
              f"RG miss({rg_miss['stall']}/{rg_miss['invalid']}/{rg_miss['pool']})")

with open("data/mc_method_compare_results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(header)
    w.writerows(rows)
print("wrote data/mc_method_compare_results.csv")
