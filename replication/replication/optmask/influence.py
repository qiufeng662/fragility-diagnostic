"""Influence functions: cheap first-order ranking of deletion moves.

These predict, for each candidate move (delete a row / delete a whole cluster),
the resulting signed-t margin of each target.  They are used ONLY to shortlist
candidates; every accepted move is re-evaluated exactly by the estimator.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def t_crit(alpha: float, df: int) -> float:
    return float(stats.t.ppf(1.0 - alpha / 2.0, max(int(df), 1)))


def _target_margin(t, target, crit) -> float:
    """Gap to threshold (>=0 means satisfied) for a raw t-statistic."""
    if getattr(target, "kind", "significant") == "null":
        return crit - abs(t)
    tt = abs(t) if target.sign == 0 else target.sign * t
    return tt - crit


def _linear_weights(fit, target) -> np.ndarray:
    wv = np.zeros(len(fit.names))
    for c, wt in target.linear().items():
        wv[fit.names.index(c)] = wt
    return wv


def _theta_se(fit, wv):
    theta = float(wv @ fit.beta)
    V = fit.xtx_inv @ fit.meat @ fit.xtx_inv * fit.scale
    Vr = V[: len(fit.names), : len(fit.names)]
    se = float(np.sqrt(max(wv @ Vr @ wv, 1e-30)))
    return theta, se


def margin(fit, target) -> float:
    wv = _linear_weights(fit, target)
    theta, se = _theta_se(fit, wv)
    return _target_margin(theta / se, target, t_crit(target.alpha, fit.df))


def margins(fit, targets) -> dict:
    return {t.column: margin(fit, t) for t in targets}


def worst_margin(fit, targets) -> float:
    return min(margins(fit, targets).values())


def row_predicted_margins(fit, targets) -> dict:
    """column -> (n_kept,) predicted margin after deleting that single row."""
    X, u, A = fit.X, fit.resid, fit.xtx_inv
    k = fit.X.shape[1]
    h = np.clip(np.sum((X @ A) * X, axis=1), 0, 0.9999)
    denom = np.clip(1.0 - h, 1e-9, None)
    delta_beta = -np.einsum('jm,im,i->ij', A, X, u / denom)
    ssr_new = fit.ssr - (u * u) / denom
    s2 = np.clip(ssr_new, 1e-30, None) / max(fit.n - 1 - k, 1)
    out = {}
    for t in targets:
        j = fit.names.index(t.column)
        proj = A[j, :] @ X.T
        diag = A[j, j] + (proj * proj) / denom
        se = np.sqrt(np.clip(s2 * diag, 1e-30, None))
        val = (fit.beta[j] + delta_beta[:, j]) / se
        out[t.column] = _target_margin(val, t, t_crit(t.alpha, fit.df))
    return out


def cluster_predicted_margins(fit, targets) -> dict:
    """column -> (n_clusters,) predicted margin after deleting the whole cluster.

    Beta shift is the first-order sum of row LOO shifts; the sandwich meat drops
    the cluster's score exactly; the bread is held at the current fit (small O(r/n)).
    """
    X, u, A = fit.X, fit.resid, fit.xtx_inv
    k = fit.X.shape[1]
    code = fit.cluster_code
    n_cl = fit.score_cluster.shape[0]
    h = np.clip(np.sum((X @ A) * X, axis=1), 0, 0.9999)
    denom = np.clip(1.0 - h, 1e-9, None)
    delta_row = -np.einsum('jm,im,i->ij', A, X, u / denom)
    delta_cl = np.zeros((n_cl, k))
    for m in range(k):
        delta_cl[:, m] = np.bincount(code, weights=delta_row[:, m], minlength=n_cl)
    counts = np.bincount(code, minlength=n_cl)
    a_meat_a = A @ fit.meat @ A
    a_s = A @ fit.score_cluster.T          # (k, n_cl)
    n, g = fit.n, fit.g
    out = {}
    for t in targets:
        j = fit.names.index(t.column)
        base = float(a_meat_a[j, j])
        contrib = a_s[j, :] ** 2
        vals = np.full(n_cl, -np.inf)
        for gi in range(n_cl):
            if counts[gi] == 0:
                continue
            n_new = n - int(counts[gi])
            g_new = g - 1
            if g_new < 1 or n_new <= k:
                continue
            scale_new = (g_new / (g_new - 1)) * ((n_new - 1) / (n_new - k))
            var = max(scale_new * (base - contrib[gi]), 1e-30)
            val = (fit.beta[j] + delta_cl[gi, j]) / np.sqrt(var)
            vals[gi] = _target_margin(val, t, t_crit(t.alpha, g_new - 1))
        out[t.column] = vals
    return out


def signed_influence_scores(fit, target) -> np.ndarray:
    """Per-row score for pushing ONE coefficient toward its target sign.

    Mirrors the proven influence heuristic of the reference engine: reward the
    direction push, the signed-t improvement, the cluster push and the leverage.
    """
    j = fit.names.index(target.column)
    X, u, A = fit.X, fit.resid, fit.xtx_inv
    n, k = X.shape
    h = np.clip(np.sum((X @ A) * X, axis=1), 0.0, 0.9999)
    infl = (A @ X.T)[j]                       # row influence on coefficient j
    delta_beta = -(infl * u) / (1.0 - h + 1e-12)
    se = float(np.sqrt(max(A[j, j] * fit.ssr / max(n - k, 1), 1e-12)))
    cluster_push = np.abs(infl * u)
    delta_se = (infl * u) * cluster_push / (se * n + 1e-12)
    s = float(target.sign if target.sign != 0 else 1.0)
    delta_t = s * (delta_beta * se - fit.beta[j] * delta_se) / (se * se + 1e-12)
    direction = np.maximum(s * delta_beta, 0.0)
    return np.maximum(2.0 * direction + 5.0 * np.maximum(delta_t, 0.0)
                      + 2.0 * cluster_push + h * np.abs(u), 0.0)


def cluster_signed_scores(fit, target) -> np.ndarray:
    """Per-cluster aggregate of the signed influence scores."""
    row = signed_influence_scores(fit, target)
    n_cl = fit.score_cluster.shape[0]
    return np.bincount(fit.cluster_code, weights=row, minlength=n_cl)


def exact_row_margins(fit, targets) -> dict:
    """column -> (n_kept,) EXACT predicted margin after deleting that single row.

    Fully vectorized.  Bread via Sherman-Morrison, meat via the cluster score
    update, finite-sample scale — all holding the absorbed-FE demeaned design
    fixed.  The per-row quadratic forms are expanded so no Python row loop is
    needed (only a per-target loop and two cluster gathers).
    """
    X, u, A = fit.X, fit.resid, fit.xtx_inv
    n, k = X.shape
    code = fit.cluster_code
    s_cl = fit.score_cluster
    meat = fit.meat
    counts = np.bincount(code, minlength=s_cl.shape[0])
    h = np.clip(np.sum((X @ A) * X, axis=1), 0, 0.9999)
    denom = 1.0 - h
    w = A @ X.T                          # (k, n): w[:, i] = A x_i
    A_meat_A = A @ meat @ A
    M_diag = np.sum((X @ A_meat_A) * X, axis=1)     # w_i' meat w_i
    s_A = s_cl @ A                       # (n_cl, k): s_g' A
    ASg = A @ s_cl.T                     # (k, n_cl): A s_g

    g_kept = np.where(counts[code] == 1, fit.g - 1, fit.g)
    n_new = n - 1
    scale = (g_kept / (g_kept - 1)) * ((n_new - 1) / (n_new - k))

    out = {}
    for t in targets:
        j = fit.names.index(t.column)
        Aj = A[:, j]
        c = w[j] / denom
        beta_new = fit.beta[j] - w[j] * (u / denom)

        t1 = s_A[code, j]                # Aj' s_g
        t2 = np.sum(X * ASg[:, code].T, axis=1)    # w_i' s_g
        a_sg = t1 + c * t2

        v = A @ (meat @ Aj)
        Q0 = float(Aj @ meat @ Aj) + 2.0 * c * (X @ v) + (c * c) * M_diag
        a_x = w[j] + c * h
        Q2 = (a_sg - u * a_x) ** 2
        var = scale * (Q0 - a_sg * a_sg + Q2)
        se_new = np.sqrt(np.clip(var, 1e-30, None))
        tv = beta_new / se_new
        crit = stats.t.ppf(1.0 - t.alpha / 2.0, np.maximum(g_kept - 1, 1))
        out[t.column] = _target_margin(tv, t, crit)
    return out


def row_beta_se_new(fit, targets) -> dict:
    """column -> (beta_new (n,), se_new (n,)) after deleting each single row.

    Same exact cluster-robust LOO as ``exact_row_margins`` but exposes the raw
    beta and se so a linearized integer program can be built on top.
    """
    X, u, A = fit.X, fit.resid, fit.xtx_inv
    n, k = X.shape
    code = fit.cluster_code
    s_cl = fit.score_cluster
    meat = fit.meat
    counts = np.bincount(code, minlength=s_cl.shape[0])
    h = np.clip(np.sum((X @ A) * X, axis=1), 0, 0.9999)
    denom = 1.0 - h
    w = A @ X.T
    A_meat_A = A @ meat @ A
    M_diag = np.sum((X @ A_meat_A) * X, axis=1)
    s_A = s_cl @ A
    ASg = A @ s_cl.T
    g_kept = np.where(counts[code] == 1, fit.g - 1, fit.g)
    scale = (g_kept / (g_kept - 1)) * ((n - 2) / (n - 1 - k))
    out = {}
    for t in targets:
        j = fit.names.index(t.column)
        Aj = A[:, j]
        c = w[j] / denom
        beta_new = fit.beta[j] - w[j] * (u / denom)
        t1 = s_A[code, j]
        t2 = np.sum(X * ASg[:, code].T, axis=1)
        a_sg = t1 + c * t2
        v = A @ (meat @ Aj)
        Q0 = float(Aj @ meat @ Aj) + 2.0 * c * (X @ v) + (c * c) * M_diag
        a_x = w[j] + c * h
        Q2 = (a_sg - u * a_x) ** 2
        var = scale * (Q0 - a_sg * a_sg + Q2)
        out[t.column] = (beta_new, np.sqrt(np.clip(var, 1e-30, None)))
    return out


def cluster_beta_se_new(fit, targets) -> dict:
    """Exact cluster-deletion beta shift and se (for building a lower-bound LP)."""
    from scipy.linalg import lu_factor, lu_solve

    X, u, A = fit.X, fit.resid, fit.xtx_inv
    k = fit.X.shape[1]
    code = fit.cluster_code
    n_cl = int(getattr(fit, "n_clusters", 0) or fit.score_cluster.shape[0])
    s_cl = np.zeros((n_cl, k))
    s_cl[:fit.score_cluster.shape[0]] = fit.score_cluster
    meat = fit.meat
    n, g = fit.n, fit.g
    db = {t.column: np.zeros(n_cl) for t in targets}
    se_new = {t.column: np.full(n_cl, np.inf) for t in targets}
    for gi in range(n_cl):
        mask = code == gi
        Xg, ug = X[mask], u[mask]
        r = int(Xg.shape[0])
        if r == 0:
            continue
        n_new, g_new = n - r, g - 1
        if g_new < 1 or n_new <= k:
            continue
        try:
            lu = lu_factor(np.eye(r) - Xg @ A @ Xg.T)
        except Exception:
            continue
        wg = lu_solve(lu, ug)
        AXg = A @ Xg.T
        db_full = AXg @ wg
        sg = s_cl[gi]
        meat_new = meat - np.outer(sg, sg)
        scale = (g_new / (g_new - 1)) * ((n_new - 1) / (n_new - k))
        for t in targets:
            j = fit.names.index(t.column)
            zg = lu_solve(lu, Xg @ A[:, j])
            aj = A[:, j] + AXg @ zg
            var = scale * float(aj @ meat_new @ aj)
            se_new[t.column][gi] = np.sqrt(max(var, 1e-30))
            db[t.column][gi] = db_full[j]
    return db, se_new


def generic_row_margins(fit, targets) -> dict:
    """column -> (n_kept,) first-order row-deletion margin, ANY estimator.

    Uses only ``Fit.influence`` (per-row influence on beta) and holds ``se``
    constant.  This is the method-agnostic ranking path; the shortlisted moves
    are exact-refitted by the solver.
    """
    out = {}
    for t in targets:
        wv = _linear_weights(fit, t)
        theta, se = _theta_se(fit, wv)
        # influence lives in the full design coordinates (real regressors first);
        # pad the weight vector with zeros on the internal dummy columns.
        wv_full = np.zeros(fit.X.shape[1])
        wv_full[: len(wv)] = wv
        theta_new = theta - fit.influence @ wv_full
        out[t.column] = _target_margin(theta_new / se, t, t_crit(t.alpha, fit.df))
    return out


def generic_cluster_margins(fit, targets) -> dict:
    """column -> (n_clusters,) first-order cluster-deletion margin, ANY estimator."""
    n_cl = int(getattr(fit, "n_clusters", 0) or fit.score_cluster.shape[0])
    code = fit.cluster_code
    out = {}
    for t in targets:
        wv = _linear_weights(fit, t)
        theta, se = _theta_se(fit, wv)
        wv_full = np.zeros(fit.X.shape[1])
        wv_full[: len(wv)] = wv
        db = np.bincount(code, weights=fit.influence @ wv_full, minlength=n_cl)
        out[t.column] = _target_margin((theta - db) / se, t, t_crit(t.alpha, fit.df))
    return out


def fast_cluster_margins(fit, targets) -> dict:
    """column -> (n_clusters,) vectorized cluster-deletion margin.

    Beta shift = first-order sum of row LOO shifts (bincount); the sandwich meat
    drops the cluster score exactly; the bread is held at the current fit.  The
    bread/rank-r corrections are O(r/n) and are absorbed by the exact re-fit of
    shortlisted candidates, so this is used only for ranking.
    """
    X, u, A = fit.X, fit.resid, fit.xtx_inv
    k = fit.X.shape[1]
    code = fit.cluster_code
    n_cl = int(getattr(fit, "n_clusters", 0) or fit.score_cluster.shape[0])
    # pad the per-cluster score to the full cluster id space (kept subset may not
    # span every id); clusters with zero kept rows get score 0 and are marked off.
    s_cl = np.zeros((n_cl, k))
    s_cl[:fit.score_cluster.shape[0]] = fit.score_cluster
    h = np.clip(np.sum((X @ A) * X, axis=1), 0, 0.9999)
    denom = 1.0 - h
    w = A @ X.T                          # (k, n)
    row_db = -w * (u / denom)            # (k, n)
    db = np.empty((n_cl, k))
    for m in range(k):
        db[:, m] = np.bincount(code, weights=row_db[m], minlength=n_cl)
    counts = np.bincount(code, minlength=n_cl).astype(float)
    A_s = A @ s_cl.T                     # (k, n_cl)
    A_meat_A = A @ fit.meat @ A
    n, g = fit.n, fit.g
    n_new = n - counts
    g_new = float(g - 1)
    ok = (counts > 0) & (g_new >= 1) & (n_new > k)
    scale = np.where(ok, (g_new / (g_new - 1)) * ((n_new - 1) / (n_new - k)), np.nan)

    out = {}
    for t in targets:
        j = fit.names.index(t.column)
        var = scale * np.clip(A_meat_A[j, j] - A_s[j, :] ** 2, 1e-30, None)
        se = np.sqrt(np.where(np.isfinite(var), var, np.inf))
        beta_new = fit.beta[j] + db[:, j]
        tv = beta_new / se
        crit = stats.t.ppf(1.0 - t.alpha / 2.0, np.maximum(g_new - 1, 1))
        m = _target_margin(tv, t, crit)
        out[t.column] = np.where((counts > 0) & np.isfinite(tv), m, -np.inf)
    return out


def exact_cluster_margins(fit, targets) -> dict:
    """column -> (n_clusters,) EXACT predicted margin after deleting the whole cluster.

    Bread via the rank-r Woodbury identity (LU-factored once per cluster), meat
    drops the cluster score exactly, scale uses the true post-deletion n/g.
    Only the needed diagonal of the sandwich is computed.  Holds the absorbed-FE
    demeaned design fixed.
    """
    from scipy.linalg import lu_factor, lu_solve

    X, u, A = fit.X, fit.resid, fit.xtx_inv
    k = fit.X.shape[1]
    code = fit.cluster_code
    n_cl = fit.score_cluster.shape[0]
    s_cl = fit.score_cluster
    meat = fit.meat
    n, g = fit.n, fit.g
    out = {t.column: np.full(n_cl, -np.inf) for t in targets}
    for gi in range(n_cl):
        mask = code == gi
        Xg, ug = X[mask], u[mask]
        r = int(Xg.shape[0])
        if r == 0:
            continue
        n_new, g_new = n - r, g - 1
        if g_new < 1 or n_new <= k:
            continue
        try:
            lu = lu_factor(np.eye(r) - Xg @ A @ Xg.T)
        except Exception:
            continue
        wg = lu_solve(lu, ug)
        AXg = A @ Xg.T
        db_full = AXg @ wg
        sg = s_cl[gi]
        meat_new = meat - np.outer(sg, sg)
        scale = (g_new / (g_new - 1)) * ((n_new - 1) / (n_new - k))
        for t in targets:
            j = fit.names.index(t.column)
            zg = lu_solve(lu, Xg @ A[:, j])
            aj = A[:, j] + AXg @ zg
            var = scale * float(aj @ meat_new @ aj)
            se = np.sqrt(max(var, 1e-30))
            tv = (fit.beta[j] + db_full[j]) / se
            out[t.column][gi] = _target_margin(tv, t, t_crit(t.alpha, g_new - 1))
    return out
