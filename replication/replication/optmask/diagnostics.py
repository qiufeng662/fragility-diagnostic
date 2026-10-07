"""Certified fragility diagnostics (paper contribution 3).

Turns the verified machinery (relative leverage, box certificate, exact
determinant/adjugate evaluation) into a concrete, theory-backed procedure for
*honestly reporting* how fragile a fitted regression is to data deletion.

Every step carries the theorem it rests on:
  1. relative leverage h_g          -> Thm 13.5 (coordinate-invariant)
  2. box certificate eta_K          -> Thm 12.6 / Cor 13.2
  3. exact deletion margin          -> Thm 12.9-12.11 (inverse-free)
  4. first-order distortion factor  -> Thm 13.6 (1/(1-h))
"""
from __future__ import annotations

import numpy as np

from .certify import det_adj_precompute, det_adj_margin


def _frac_knapsack(values, weights, K):
    order = np.argsort(-values / np.maximum(weights, 1e-12))
    budget = float(K); total = 0.0
    for g in order:
        w = weights[g]
        if w <= budget:
            total += values[g]; budget -= w
        elif budget > 0:
            total += values[g] * (budget / w); budget = 0
        else:
            break
    return total


def fragility_report(pre, target, tau=0.5, K_max=200):
    """Certified fragility report for one target.

    Parameters
    ----------
    pre : dict from ``certify.det_adj_precompute`` (full-data fit).
    target : Target.
    tau : dangerous-cluster leverage threshold.
    K_max : largest deletion budget (rows) to scan for the box certificate.

    Returns
    -------
    dict with keys:
      h_g, dangerous, eta_K, box_ok, box_fail_K, per_cluster, warnings
    """
    H = pre["H"]
    G = pre["G"]
    c = pre["c"]
    h_g = np.array([np.linalg.norm(H[g], 2) for g in range(G)])  # lambda_max(H_g)

    order = np.argsort(-h_g)
    dangerous = [(int(g), float(h_g[g])) for g in order if h_g[g] >= tau]

    eta_K = np.array([_frac_knapsack(h_g, c, float(K)) for K in range(1, K_max + 1)])
    box_ok = bool(np.all(eta_K < 1.0))
    fail = np.flatnonzero(eta_K >= 1.0)
    box_fail_K = int(fail[0] + 1) if len(fail) else None

    per_cluster = []
    for g, hg in dangerous:
        q = None
        m = det_adj_margin(pre, target, [g])
        # q_j via the same determinant form (N_j/d), reusing det_adj internals
        q = _exact_q(pre, target, [g])
        fo_factor = 1.0 / max(1.0 - hg, 1e-12)
        per_cluster.append({
            "g": g, "h_g": hg, "q_exact": q, "m_exact": m,
            "first_order_factor": fo_factor,
            "flips": bool(q is not None and q > 0),
            "significant": bool(m is not None and m > 0),
        })

    warnings = []
    if not box_ok:
        warnings.append(f"no finite global box certificate: eta_K>=1 from K={box_fail_K} rows")
    for pc in per_cluster:
        if pc["h_g"] > 0.9:
            warnings.append(
                f"cluster {pc['g']}: h_g={pc['h_g']:.3f} -> first-order influence "
                f"distorted by factor {pc['first_order_factor']:.0f}x; use exact evaluation")
    return {"h_g": h_g, "dangerous": dangerous, "eta_K": eta_K, "box_ok": box_ok,
            "box_fail_K": box_fail_K, "per_cluster": per_cluster, "warnings": warnings}


def _exact_q(pre, target, deleted_clusters):
    """Exact signed deviation q_j = s_j (a_j^T beta(x) - b_j^0) via determinant form."""
    fit = pre["fit"]
    wv = np.zeros(pre["k"])
    for col, wt in target.linear().items():
        wv[fit.names.index(col)] = wt
    a_tilde = pre["Ps"] @ wv
    s = float(target.sign if target.sign != 0 else 1.0)
    theta_full = float(wv[: len(fit.names)] @ fit.beta)

    x = np.zeros(pre["G"])
    for g in deleted_clusters:
        if 0 <= g < pre["G"]:
            x[g] = 1.0
    Q = np.eye(pre["k"]) - np.einsum('g,gab->ab', x, pre["H"])
    r = np.einsum('g,ga->a', x, pre["psi_tilde"])
    d = float(np.linalg.det(Q))
    if d <= 1e-12:
        return None
    adjQ = d * np.linalg.inv(Q)
    N = s * (theta_full * d - float(a_tilde @ adjQ @ r))
    return N / d
