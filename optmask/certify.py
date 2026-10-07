"""Verification harness for the certification lower bound.

Computes the significance margin m_j(x) = (s_j·β̂_j(x))² − t_crit²·se_j(x)² EXACTLY
(for any deletion set, via full refit) plus its first/second discrete derivatives,
so Codex's closed-form derivatives can be checked against ground truth.
"""
from __future__ import annotations

import numpy as np

from . import influence as inf
from .cluster_valid import TWFE_CV


def exact_margin(est, prepared, complete, target, deleted_clusters, data):
    """m_j for the deletion set (clusters are fully deleted)."""
    keep = complete.copy()
    code = prepared["cluster_code"]
    for g in deleted_clusters:
        keep &= code != g
    f = est.fit(prepared, keep)
    j = f.names.index(target.column) if target.column in f.names else None
    # weighted targets: use the linear combination theta
    wv = inf._linear_weights(f, target)
    theta, se = inf._theta_se(f, wv)
    s = float(target.sign if target.sign != 0 else 1.0)
    tcrit = inf.t_crit(target.alpha, f.df)
    return (s * theta) ** 2 - (tcrit * se) ** 2


def first_diff(est, prepared, complete, target, g, data):
    m0 = exact_margin(est, prepared, complete, target, [], data)
    mg = exact_margin(est, prepared, complete, target, [g], data)
    return mg - m0


def second_diff(est, prepared, complete, target, g, h, data):
    m0 = exact_margin(est, prepared, complete, target, [], data)
    mg = exact_margin(est, prepared, complete, target, [g], data)
    mh = exact_margin(est, prepared, complete, target, [h], data)
    mgh = exact_margin(est, prepared, complete, target, [g, h], data)
    return mgh - mg - mh + m0


def check(est, prepared, complete, target, pairs, data):
    for g, h in pairs:
        m0 = exact_margin(est, prepared, complete, target, [], data)
        fg = first_diff(est, prepared, complete, target, g, data)
        fh = first_diff(est, prepared, complete, target, h, data)
        s2 = second_diff(est, prepared, complete, target, g, h, data)
        mgh = exact_margin(est, prepared, complete, target, [g, h], data)
        taylor2 = m0 + fg + fh + s2
        print(f"pair({g},{h}): m0={m0:.4f} exact_mgh={mgh:.4f} taylor2={taylor2:.4f} "
              f"err={abs(mgh - taylor2):.2e}")


def _sqrtm(P):
    ev, Q = np.linalg.eigh(P)
    return (Q * np.sqrt(np.clip(ev, 0, None))) @ Q.T


def det_adj_precompute(fit):
    """Extract the whitened quantities for the determinant/adjugate exact evaluator.

    ``fit`` must be a FULL-data fit (all rows kept), so ``fit.cluster_code`` spans
    every cluster.  Returns a dict keyed off ``fit``; the evaluator then computes
    the exact margin for ANY cluster-deletion set in O(p^3) without refitting and
    without a Woodbury LU (Codex MATH_RESEARCH §12.6, eq. 12.55-12.65).
    """
    X, u, P = fit.X, fit.resid, fit.xtx_inv
    code = fit.cluster_code
    k = X.shape[1]
    G = int(code.max()) + 1 if len(code) else 0
    A = np.zeros((G, k, k))
    psi = np.zeros((G, k))
    c = np.zeros(G)
    for g in range(G):
        m = code == g
        Xg = X[m]
        A[g] = Xg.T @ Xg
        psi[g] = Xg.T @ u[m]
        c[g] = m.sum()
    Ps = _sqrtm(P)
    H = np.einsum('ab,gbr,rs->gas', Ps, A, Ps)   # H_g = P^{1/2} A_g P^{1/2}
    psi_tilde = (Ps @ psi.T).T
    return {"fit": fit, "P": P, "Ps": Ps, "A": A, "psi": psi, "H": H,
            "psi_tilde": psi_tilde, "c": c, "k": k, "G": G,
            "n0": fit.n, "G0": fit.g,
            "n_entities": getattr(fit, "n_entities", 0)}


def det_adj_margin(pre, target, deleted_clusters):
    """Exact significance margin m_j(x) via determinant/adjugate (no refit, no inverse).

    Equivalent to ``exact_margin`` (full refit) but inverse-free.  Returns ``-inf``
    when the post-deletion information matrix Q(x) is not positive definite (the
    margin is then undefined).  Weighted targets are handled through the linear
    combination a_j^T beta, exactly as in ``influence._theta_se``.
    """
    fit = pre["fit"]
    wv = inf._linear_weights(fit, target)          # length = k_real (real regressors)
    wv_full = np.zeros(pre["k"])
    wv_full[: len(wv)] = wv                         # real regressors are the first columns
    a_tilde = pre["Ps"] @ wv_full
    s = float(target.sign if target.sign != 0 else 1.0)
    theta_full = float(wv @ fit.beta)

    x = np.zeros(pre["G"])
    for g in deleted_clusters:
        if 0 <= g < pre["G"]:
            x[g] = 1.0
    Q = np.eye(pre["k"]) - np.einsum('g,gab->ab', x, pre["H"])
    r = np.einsum('g,ga->a', x, pre["psi_tilde"])
    d = float(np.linalg.det(Q))
    if d <= 1e-12:
        return -np.inf
    adjQ = d * np.linalg.inv(Q)

    N = s * (theta_full * d - float(a_tilde @ adjQ @ r))
    L = np.zeros(pre["G"])
    for g in range(pre["G"]):
        L[g] = float(a_tilde @ adjQ @ (d * pre["psi_tilde"][g] + pre["H"][g] @ adjQ @ r))
    S = float(np.sum((1.0 - x) * L ** 2))

    n_new = pre["n0"] - float(pre["c"] @ x)
    G_new = pre["G0"] - float(x.sum())
    n_entities_kept = pre["n_entities"] - float(x.sum())
    p_eff = pre["k"] + max(n_entities_kept, 0.0)   # full-design rank under the mask
    if G_new <= 1 or n_new <= p_eff:              # single cluster or no residual df
        return -np.inf
    scale = (G_new / (G_new - 1.0)) * ((n_new - 1.0) / (n_new - p_eff))
    rho = inf.t_crit(target.alpha, G_new - 1) ** 2 * scale
    return (N ** 2 * d ** 2 - rho * S) / d ** 4
