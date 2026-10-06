"""Estimators: the method layer. The solver only sees ``fit(mask) -> Fit``."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Tuple

import numpy as np


@dataclass
class Fit:
    """Point estimates + influence ingredients for one regression on a mask."""
    names: Tuple[str, ...]
    beta: np.ndarray
    se: np.ndarray
    n: int
    g: int                                  # number of clusters
    df: int                                 # cluster dof = g - 1
    # influence ingredients (rows are the KEPT subset, in source order):
    X: np.ndarray                           # (n_kept, k) demeaned design
    resid: np.ndarray                       # (n_kept,)
    xtx_inv: np.ndarray                     # (k, k)
    ssr: float
    cluster_code: np.ndarray                # (n_kept,) integer cluster codes
    full_index: np.ndarray                  # (n_kept,) positions into the source frame
    meat: np.ndarray                        # (k, k) cluster sandwich meat
    score_cluster: np.ndarray               # (n_clusters, k) per-cluster score
    scale: float                            # finite-sample scale factor
    n_clusters: int = 0                     # full-data cluster count
    influence: np.ndarray = None            # (n_kept, k) per-row influence on beta

    @property
    def t(self) -> np.ndarray:
        return self.beta / self.se


def _demean_two_way(values: np.ndarray, keep: np.ndarray, codes: Tuple[np.ndarray, np.ndarray],
                    sizes: Tuple[int, int], iters: int = 24) -> np.ndarray:
    out = np.where(keep, values, 0.0)
    mask = keep.astype(float)
    for _ in range(iters):
        for code, size in zip(codes, sizes):
            total = np.bincount(code, weights=out, minlength=size)
            count = np.bincount(code, weights=mask, minlength=size)
            mean = np.divide(total, count, out=np.zeros(size), where=count > 0)
            out = out - mean[code] * mask
    return out


class TWFE:
    """Two-way fixed effects (entity + time), cluster-robust SE."""

    def __init__(self, y_col: str, regressors: Sequence[str], entity_col: str,
                 time_col: str, cluster_col: str, absorb_time: bool = True):
        self.y_col = y_col
        self.regressors = tuple(regressors)
        self.entity_col = entity_col
        self.time_col = time_col
        self.cluster_col = cluster_col
        self.absorb_time = absorb_time

    def required_columns(self) -> Tuple[str, ...]:
        return (self.y_col, *self.regressors, self.entity_col,
                self.time_col, self.cluster_col)

    def prepare(self, data) -> dict:
        y = data[self.y_col].to_numpy(float)
        X = np.column_stack([data[c].to_numpy(float) for c in self.regressors])
        entity_code = _codes(data[self.entity_col].to_numpy())
        time_code = _codes(data[self.time_col].to_numpy())
        cluster_code = _codes(data[self.cluster_col].to_numpy())
        groups = ((entity_code, time_code) if self.absorb_time else (entity_code,))
        sizes = (entity_code.max() + 1, time_code.max() + 1) if self.absorb_time else (entity_code.max() + 1,)
        return {"y": y, "X": X, "cluster_code": cluster_code, "groups": groups,
                "sizes": sizes, "n_clusters": int(cluster_code.max()) + 1}

    def fit(self, prepared: dict, keep: np.ndarray) -> Fit:
        y, X, cluster_code = prepared["y"], prepared["X"], prepared["cluster_code"]
        groups, sizes = prepared["groups"], prepared["sizes"]
        yd = _demean_two_way(y, keep, groups, sizes)
        Xd = np.column_stack([_demean_two_way(X[:, j], keep, groups, sizes)
                              for j in range(X.shape[1])])
        sub = keep
        ys, Xs = yd[sub], Xd[sub]
        beta, *_ = np.linalg.lstsq(Xs, ys, rcond=None)
        resid = ys - Xs @ beta
        xtx_inv = np.linalg.pinv(Xs.T @ Xs)
        n, k = Xs.shape
        code = cluster_code[sub]
        size = int(code.max()) + 1
        agg = np.empty((size, k))
        for j in range(k):
            agg[:, j] = np.bincount(code, weights=Xs[:, j] * resid, minlength=size)
        meat = agg.T @ agg
        g = int(np.unique(code).size)
        scale = (g / max(g - 1, 1)) * ((n - 1) / max(n - k, 1))
        vcov = xtx_inv @ meat @ xtx_inv * scale
        se = np.sqrt(np.clip(np.diag(vcov), 1e-30, None))
        h = np.clip(np.sum((Xs @ xtx_inv) * Xs, axis=1), 0, 0.9999)
        infl = (xtx_inv @ Xs.T) * (resid / (1.0 - h))   # (k, n) -> per-row influence
        return Fit(names=tuple(self.regressors), beta=beta, se=se, n=n, g=g, df=g - 1,
                   X=Xs, resid=resid, xtx_inv=xtx_inv, ssr=float(resid @ resid),
                   cluster_code=code, full_index=np.flatnonzero(keep),
                   meat=meat, score_cluster=agg, scale=scale,
                   n_clusters=prepared.get("n_clusters", size), influence=infl.T)


class OLS:
    """Pooled OLS (optional time absorb), cluster-robust SE."""

    def __init__(self, y_col, regressors, cluster_col, absorb=None):
        self.y_col = y_col
        self.regressors = tuple(regressors)
        self.cluster_col = cluster_col
        self.absorb = absorb

    def required_columns(self):
        return (self.y_col, *self.regressors, self.cluster_col, *(self.absorb or ()))

    def prepare(self, data):
        y = data[self.y_col].to_numpy(float)
        X = np.column_stack([data[c].to_numpy(float) for c in self.regressors])
        cluster_code = _codes(data[self.cluster_col].to_numpy())
        return {"y": y, "X": X, "cluster_code": cluster_code,
                "n_clusters": int(cluster_code.max()) + 1}

    def fit(self, prepared, keep):
        y, X, cluster_code = prepared["y"], prepared["X"], prepared["cluster_code"]
        ys, Xs = y[keep], X[keep]
        beta, *_ = np.linalg.lstsq(Xs, ys, rcond=None)
        resid = ys - Xs @ beta
        xtx_inv = np.linalg.pinv(Xs.T @ Xs)
        n, k = Xs.shape
        code = cluster_code[keep]
        size = int(code.max()) + 1
        agg = np.empty((size, k))
        for j in range(k):
            agg[:, j] = np.bincount(code, weights=Xs[:, j] * resid, minlength=size)
        meat = agg.T @ agg
        g = int(np.unique(code).size)
        scale = (g / max(g - 1, 1)) * ((n - 1) / max(n - k, 1))
        vcov = xtx_inv @ meat @ xtx_inv * scale
        se = np.sqrt(np.clip(np.diag(vcov), 1e-30, None))
        h = np.clip(np.sum((Xs @ xtx_inv) * Xs, axis=1), 0, 0.9999)
        infl = (xtx_inv @ Xs.T) * (resid / (1.0 - h))
        return Fit(names=tuple(self.regressors), beta=beta, se=se, n=n, g=g, df=g - 1,
                   X=Xs, resid=resid, xtx_inv=xtx_inv, ssr=float(resid @ resid),
                   cluster_code=code, full_index=np.flatnonzero(keep),
                   meat=meat, score_cluster=agg, scale=scale,
                   n_clusters=prepared.get("n_clusters", size), influence=infl.T)


def _codes(values: np.ndarray) -> np.ndarray:
    _, inv = np.unique(values, return_inverse=True)
    return inv.astype(np.int64)


class IV2SLS:
    """Two-stage least squares with cluster-robust SE (method-agnostic example).

    Structural: y = endog*beta + exog*gamma + eps, instruments = Z (incl. exog).
    Influence is the first-order structural analogue of the OLS LOO formula with
    the first-stage fitted regressors X_hat; enough for ranking + exact refit.
    """

    def __init__(self, y_col, endog, exog, instruments, cluster_col):
        self.y_col = y_col
        self.endog = tuple(endog)
        self.exog = tuple(exog)
        self.instruments = tuple(instruments)
        self.cluster_col = cluster_col
        self.regressors = self.endog + self.exog

    def required_columns(self):
        return (self.y_col, *self.endog, *self.exog, *self.instruments, self.cluster_col)

    def prepare(self, data):
        y = data[self.y_col].to_numpy(float)
        X = np.column_stack([data[c].to_numpy(float) for c in self.regressors])
        Z = np.column_stack([data[c].to_numpy(float)
                             for c in (*self.instruments, *self.exog)])
        cluster_code = _codes(data[self.cluster_col].to_numpy())
        return {"y": y, "X": X, "Z": Z, "cluster_code": cluster_code,
                "n_clusters": int(cluster_code.max()) + 1}

    def fit(self, prepared, keep):
        y, X, Z, cluster_code = prepared["y"], prepared["X"], prepared["Z"], prepared["cluster_code"]
        ys, Xs, Zs = y[keep], X[keep], Z[keep]
        Pz = Zs @ np.linalg.pinv(Zs.T @ Zs) @ Zs.T
        Xh = Pz @ Xs
        XhXh_inv = np.linalg.pinv(Xh.T @ Xh)
        beta = XhXh_inv @ Xh.T @ ys
        u = ys - Xs @ beta
        n, k = Xs.shape
        code = cluster_code[keep]
        size = int(code.max()) + 1
        agg = np.empty((size, k))
        for j in range(k):
            agg[:, j] = np.bincount(code, weights=Xh[:, j] * u, minlength=size)
        meat = agg.T @ agg
        g = int(np.unique(code).size)
        scale = (g / max(g - 1, 1)) * ((n - 1) / max(n - k, 1))
        vcov = XhXh_inv @ meat @ XhXh_inv * scale
        se = np.sqrt(np.clip(np.diag(vcov), 1e-30, None))
        h = np.clip(np.sum((Xh @ XhXh_inv) * Xh, axis=1), 0, 0.9999)
        infl = (XhXh_inv @ Xh.T) * (u / (1.0 - h))
        return Fit(names=self.regressors, beta=beta, se=se, n=n, g=g, df=g - 1,
                   X=Xs, resid=u, xtx_inv=XhXh_inv, ssr=float(u @ u),
                   cluster_code=code, full_index=np.flatnonzero(keep),
                   meat=meat, score_cluster=agg, scale=scale,
                   n_clusters=prepared.get("n_clusters", size), influence=infl.T)
