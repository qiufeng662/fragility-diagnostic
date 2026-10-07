"""Cluster-valid TWFE estimator for leave-cluster-out influence.

The standard TWFE estimator absorbs entity and time fixed effects simultaneously
with a two-way demeaning step.  When clusters coincide with entities, the
absorbed time effects are not nested inside clusters, so leaving a cluster out
also removes part of the time-effect definition.  This makes cluster influence /
jackknife machinery inconsistent (MacKinnon, Nielsen & Webb, arXiv:2205.03288).

The remedy: keep the entity fixed effect absorbed (nested in the cluster
dimension) and represent the time effects explicitly as year dummy regressors.
The dummies are used internally for estimation and influence calculations but
are not exposed as target coefficients.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

try:
    from optmask.estimators import Fit, _codes
except ModuleNotFoundError:  # pragma: no cover - direct execution fallback
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from optmask.estimators import Fit, _codes


def _demean_one_way(values: np.ndarray, keep: np.ndarray, code: np.ndarray,
                    size: int) -> np.ndarray:
    """Entity-demean values within the kept subset (one pass, exact)."""
    mask = keep.astype(float)
    total = np.bincount(code, weights=values * mask, minlength=size)
    count = np.bincount(code, weights=mask, minlength=size)
    mean = np.divide(total, count, out=np.zeros(size), where=count > 0)
    return values - mean[code]


class TWFE_cluster_valid:
    """Two-way FE estimator valid for leave-cluster-out influence.

    Entity FE is absorbed; time effects enter as dummy regressors.  ``Fit.X``,
    ``xtx_inv``, ``meat``, ``score_cluster`` and ``influence`` use the FULL
    design (real regressors first, then year dummies) so the whitened cluster
    matrices satisfy sum_g H_g = I; ``names``, ``beta`` and ``se`` expose only
    the k_real real regressors.

    Interface restrictions.  This estimator assumes (i) a one-to-one
    correspondence between entities and clusters (the cluster dimension is the
    entity dimension), and (ii) a full-rank within-entity demeaned design with
    n > p_eff and G > 1 residual clusters.  For degenerate inputs (a single
    cluster, no residual degrees of freedom, or a rank-deficient design) the
    standard errors are set to NaN rather than fabricated by denominator
    clipping; the general rank-deficient case should use the actual design rank
    instead of the entity-count convention used here.
    """

    def __init__(self, y_col: str, regressors: Sequence[str], entity_col: str,
                 time_col: str, cluster_col: str):
        self.y_col = y_col
        self.regressors = tuple(regressors)
        self.entity_col = entity_col
        self.time_col = time_col
        self.cluster_col = cluster_col
        self.k_real = len(self.regressors)

    def required_columns(self) -> Tuple[str, ...]:
        return (self.y_col, *self.regressors, self.entity_col,
                self.time_col, self.cluster_col)

    def prepare(self, data) -> dict:
        y = data[self.y_col].to_numpy(float)
        X_real = np.column_stack([data[c].to_numpy(float) for c in self.regressors])
        entity_code = _codes(data[self.entity_col].to_numpy())
        time_code = _codes(data[self.time_col].to_numpy())
        cluster_code = _codes(data[self.cluster_col].to_numpy())

        # Time dummies, dropping the first year to avoid collinearity.
        n_time = int(time_code.max()) + 1
        time_dummies = np.column_stack(
            [(time_code == t).astype(float) for t in range(1, n_time)]
        )

        X = np.column_stack([X_real, time_dummies])
        return {
            "y": y,
            "X": X,
            "cluster_code": cluster_code,
            "entity_code": entity_code,
            "n_entities": int(entity_code.max()) + 1,
            "n_clusters": int(cluster_code.max()) + 1,
            "k_real": self.k_real,
        }

    def fit(self, prepared: dict, keep: np.ndarray) -> Fit:
        y, X, cluster_code = prepared["y"], prepared["X"], prepared["cluster_code"]
        entity_code = prepared["entity_code"]
        n_entities = prepared["n_entities"]
        k_real = prepared["k_real"]

        yd = _demean_one_way(y, keep, entity_code, n_entities)
        Xd = np.column_stack([
            _demean_one_way(X[:, j], keep, entity_code, n_entities)
            for j in range(X.shape[1])
        ])

        sub = keep
        ys, Xs = yd[sub], Xd[sub]
        beta_all, *_ = np.linalg.lstsq(Xs, ys, rcond=None)
        resid = ys - Xs @ beta_all
        xtx_inv_all = np.linalg.pinv(Xs.T @ Xs)
        n, k_all = Xs.shape
        code = cluster_code[sub]
        size = int(code.max()) + 1

        agg_all = np.empty((size, k_all))
        for j in range(k_all):
            agg_all[:, j] = np.bincount(code, weights=Xs[:, j] * resid, minlength=size)
        meat_all = agg_all.T @ agg_all
        g = int(np.unique(code).size)
        n_entities_kept = int(np.unique(entity_code[sub]).size)
        p_eff = k_all + n_entities_kept          # full-design rank (absorbed entity FE)
        rank_ok = np.linalg.matrix_rank(Xs) >= k_all
        if g <= 1 or n <= p_eff or not rank_ok:
            # inference unavailable: single cluster, no residual df, or a
            # rank-deficient within-entity design
            se_all = np.full(k_all, np.nan)
            scale = np.nan
        else:
            scale = (g / (g - 1)) * ((n - 1) / (n - p_eff))
            vcov_all = xtx_inv_all @ meat_all @ xtx_inv_all * scale
            se_all = np.sqrt(np.clip(np.diag(vcov_all), 1e-30, None))

        h = np.clip(np.sum((Xs @ xtx_inv_all) * Xs, axis=1), 0, 0.9999)
        infl = (xtx_inv_all @ Xs.T) * (resid / (1.0 - h))
        influence_all = infl.T

        # The Fit contract exposes the FULL design (real regressors + year
        # dummies): X, xtx_inv, meat, score_cluster and influence all live in
        # the full k_all coordinate system, so sum_g H_g = I holds exactly for
        # the whitened cluster matrices.  Target-level quantities (names, beta,
        # se) remain the k_real real regressors, which occupy the FIRST k_real
        # columns of the full design.
        return Fit(
            names=tuple(self.regressors),
            beta=beta_all[:k_real],
            se=se_all[:k_real],
            n=n,
            g=g,
            df=g - 1,
            X=Xs,
            resid=resid,
            xtx_inv=xtx_inv_all,
            ssr=float(resid @ resid),
            cluster_code=code,
            full_index=np.flatnonzero(keep),
            meat=meat_all,
            score_cluster=agg_all,
            scale=scale,
            n_clusters=prepared.get("n_clusters", size),
            n_entities=n_entities_kept,
            influence=influence_all,
        )


# Backward-compatible alias used by existing scripts in this repo.
TWFE_CV = TWFE_cluster_valid


if __name__ == "__main__":
    import pandas as pd
    from optmask.estimators import TWFE

    path = r"D:\op projects\phacking\data\agent_tasks\xhs928_six_x_v2_20261002\input_joint.csv"
    data = pd.read_csv(path)

    y_col = "AI"
    regressors = [
        "Size", "Mfee", "TechGap", "RD", "Policydep",
        "AS", "OCF", "ROA", "Lev", "Growth",
    ]
    entity_col = "firmid"
    time_col = "year"
    cluster_col = "firmid"

    # Drop rows with any missing required variable (listwise deletion).
    # entity_col and cluster_col may be the same column; keep unique columns.
    cols = list(dict.fromkeys([y_col, *regressors, entity_col, time_col, cluster_col]))
    data = data[cols].dropna().reset_index(drop=True)
    keep = np.ones(len(data), dtype=bool)

    est_twfe = TWFE(y_col, regressors, entity_col, time_col, cluster_col)
    prep_twfe = est_twfe.prepare(data)
    fit_twfe = est_twfe.fit(prep_twfe, keep)

    est_valid = TWFE_cluster_valid(y_col, regressors, entity_col, time_col, cluster_col)
    prep_valid = est_valid.prepare(data)
    fit_valid = est_valid.fit(prep_valid, keep)

    print("\nComparison of TWFE (two-way demeaned) vs TWFE_cluster_valid (entity-demeaned + year dummies)\n")
    print(f"{'Regressor':>12} {'TWFE_beta':>14} {'Valid_beta':>14} {'Diff_beta':>14} "
          f"{'TWFE_se':>12} {'Valid_se':>12} {'Diff_se':>12}")
    print("-" * 92)
    for name, b1, b2, s1, s2 in zip(fit_twfe.names, fit_twfe.beta, fit_valid.beta,
                                     fit_twfe.se, fit_valid.se):
        print(f"{name:>12} {b1:14.6f} {b2:14.6f} {abs(b1 - b2):14.2e} "
              f"{s1:12.6f} {s2:12.6f} {abs(s1 - s2):12.2e}")

    max_beta_diff = np.max(np.abs(fit_twfe.beta - fit_valid.beta))
    max_se_diff = np.max(np.abs(fit_twfe.se - fit_valid.se))
    print("\nMax abs diff (beta):", max_beta_diff)
    print("Max abs diff (se):  ", max_se_diff)
    print("Sample size n:      ", fit_valid.n)
    print("Number of clusters g:", fit_valid.g)
    print("Internal design k_all:", prep_valid["X"].shape[1], " exposed k_real:", prep_valid["k_real"])
    print("Influence shape:    ", fit_valid.influence.shape)
