"""Full-significance method evaluation (the CR1 sign-and-significance target).

Same controllable-leverage randomized whitened DGP as mc_eval.py, but the target
is the full sign-and-significance condition: beta_1 < 0 (sign, for s = -1) AND
m_j(x) = q_j(x)^2 - rho_j(x) v_j(x) > 0 (significance), on a valid design
(Q positive-definite, G_new > 1, n_new > p_eff).  The planted k=3 deletion set is
minimal, flipping and significant; background clusters can produce sign-only or
non-significant sets that the full criterion must reject.

The margin evaluator caches a single Cholesky factorisation of Q and reuses it
for every retained-cluster right-hand side, so a margin evaluation costs
O(p^3 + G p^2) rather than O(G p^3).

Run:  python mc_eval_sig.py
"""
import time
import numpy as np
from scipy import linalg as sla
from scipy import stats

rng = np.random.default_rng(20261005)
P = 6
B = 4.0
K = 3
H_LOW = 0.6


def build_instance(G, eps=0.05):
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
    # background: -r/(G-K) plus a centered target-direction perturbation; eps=0
    # recovers the degenerate zero-variance case, eps>0 gives a nonzero retained
    # variance so the CR1 significance constraint is non-trivial.
    if eps > 0:
        pert = eps * rng.standard_normal(G - K)
        pert -= pert.mean()
        psi[K:] = (-r[None, :] / (G - K)) + (pert[:, None] * a[None, :])
    else:
        psi[K:] = -r[None, :] / (G - K)
    c = rng.integers(P, 21, size=G).astype(float)
    return H, psi, c, a


def _solve_many(L, rhs_list):
    out = []
    for rhs in rhs_list:
        out.append(sla.solve_triangular(
            L.T, sla.solve_triangular(L, rhs, lower=True), lower=False))
    return out


def beta1_of(mask, H, psi, a):
    if not mask:
        return B / 2.0
    HD = H[mask].sum(axis=0)
    Q = np.eye(P) - HD
    try:
        L = np.linalg.cholesky(Q)
    except np.linalg.LinAlgError:
        return None
    r = psi[mask].sum(axis=0)
    z = sla.solve_triangular(L.T, sla.solve_triangular(L, r, lower=True), lower=False)
    return B / 2.0 - a @ z


def evaluate(mask, H, psi, c, a):
    """Return (beta1, margin) for the deletion set; (None, None) if invalid
    (singular Q, G_new <= 1, or n_new <= P)."""
    if not mask:
        return B / 2.0, -np.inf
    HD = H[mask].sum(axis=0)
    Q = np.eye(P) - HD
    try:
        L = np.linalg.cholesky(Q)
    except np.linalg.LinAlgError:
        return None, None
    r = psi[mask].sum(axis=0)
    z = sla.solve_triangular(L.T, sla.solve_triangular(L, r, lower=True), lower=False)
    beta1 = B / 2.0 - a @ z
    kept = [g for g in range(len(H)) if g not in mask]
    G_new = len(kept)
    n_new = sum(c[g] for g in kept)
    if G_new <= 1 or n_new <= P:
        return beta1, None
    v = 0.0
    for g in kept:
        psig = psi[g] + H[g] @ z
        zg = sla.solve_triangular(
            L.T, sla.solve_triangular(L, psig, lower=True), lower=False)
        v += (a @ zg) ** 2
    rho = stats.t.ppf(1 - 0.05/2, G_new - 1) ** 2 * (G_new / (G_new - 1)) * ((n_new - 1) / (n_new - P))
    margin = (-beta1) ** 2 - rho * v
    return beta1, margin


def feasible(mask, H, psi, c, a):
    beta1, margin = evaluate(mask, H, psi, c, a)
    return beta1 is not None and margin is not None and beta1 < 0 and margin > 0


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
            if b is not None and b < best_b:
                best_b, best_g = b, g
        if best_g is None:
            break
        mask.append(best_g)
        accepts += 1
        if feasible(mask, H, psi, c, a):
            return mask, True, trials, accepts
    return mask, False, trials, accepts


def exhaustive_min_sig(H, psi, c, a, G):
    best = None
    for bits in range(1 << G):
        mask = [g for g in range(G) if (bits >> g) & 1]
        if best is not None and len(mask) >= best:
            continue
        if feasible(mask, H, psi, c, a):
            best = len(mask)
    return best


# per-evaluation timing on a fixed valid mask (the planted set)
H0, psi0, c0, a0 = build_instance(15)
mask0 = list(range(K))
t0 = time.perf_counter()
for _ in range(1000):
    beta1_of(mask0, H0, psi0, a0)
t_beta_eval = (time.perf_counter() - t0) / 1000 * 1e6
t0 = time.perf_counter()
for _ in range(1000):
    evaluate(mask0, H0, psi0, c0, a0)
t_margin_eval = (time.perf_counter() - t0) / 1000 * 1e6
print(f"per-evaluation cost on the planted mask: "
      f"beta1_only = {t_beta_eval:.2f} us, full margin = {t_margin_eval:.2f} us")

for eps in [0.0, 0.05]:
    print(f"\n=== eps = {eps} ===")
    print(f"{'G':>3} {'tau':>5} | {'sig recall':>10} {'found size':>10} "
          f"{'true min':>8} {'paired ratio':>12} {'search(ms)':>10}")
    for G in [15, 30]:
        for tau in [0.5, 0.3, 0.1]:
            found = 0
            sizes = []
            ratios = []
            tms = []
            t_search = 0.0
            for _ in range(200):
                H, psi, c, a = build_instance(G, eps)
                t0 = time.perf_counter()
                mask, ok, _, _ = search_recursive_greedy(H, psi, c, a, tau)
                t_search += time.perf_counter() - t0
                if ok:
                    found += 1
                    sizes.append(len(mask))
                if G == 15:
                    opt = exhaustive_min_sig(H, psi, c, a, G)  # SAME instance
                    tms.append(opt)
                    if ok and opt:
                        ratios.append(len(mask) / opt)
            ratio_str = f"{np.mean(ratios):.3f}" if ratios else "NA"
            tm_str = str(int(np.median(tms))) if tms else "NA"
            print(f"{G:>3} {tau:>5.2f} | {found/200*100:>9.0f}% "
                  f"{np.mean(sizes) if sizes else 0:>10.2f} "
                  f"{tm_str:>8} {ratio_str:>12} {t_search/200*1e3:>10.1f}")
