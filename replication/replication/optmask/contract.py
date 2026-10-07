"""Method-agnostic contract types for the constrained subsample solver."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Target:
    """One coefficient constraint.

    sign = -1 / 0 / +1.  sign=0 means "significant, either direction".
    kind = "significant" (|t| >= t_crit) or "null" (|t| <= t_crit).

    ``weights`` (optional) turns the target into a LINEAR COMBINATION
    ``sum_j w_j * beta_j`` (e.g. cumulative event-study effect, coefficient
    difference, RDD jump).  When None, the target is the single ``column``.
    ``column`` is then just the target's name.
    """
    column: str
    sign: int = -1
    alpha: float = 0.05
    kind: str = "significant"
    weights: dict = None

    def __post_init__(self) -> None:
        if int(self.sign) not in (-1, 0, 1):
            raise ValueError("sign must be -1, 0 or 1")
        if not (0.0 < float(self.alpha) < 1.0):
            raise ValueError("alpha must be in (0, 1)")
        if self.kind not in ("significant", "null"):
            raise ValueError("kind must be 'significant' or 'null'")

    def linear(self) -> dict:
        return dict(self.weights) if self.weights else {self.column: 1.0}


@dataclass(frozen=True)
class Spec:
    """One regression specification = an estimator recipe + its coefficient targets."""
    name: str
    y: str
    x: str
    controls: Tuple[str, ...] = ()
    entity: str = ""
    time: str = ""
    cluster: str = ""
    targets: Tuple[Target, ...] = ()

    @property
    def regressors(self) -> Tuple[str, ...]:
        return (self.x, *self.controls)


@dataclass(frozen=True)
class Constraints:
    """Structural loss ceilings (fractions)."""
    max_loss: float = 0.5
    max_entity_loss: float = 0.5
    max_cluster_loss: float = 0.5
    stratum_cols: Tuple[str, ...] = ()
    max_stratum_loss: float = 0.5
    min_rows: int = 50
    min_entities: int = 0
    min_clusters: int = 0


@dataclass
class Solution:
    keep: np.ndarray
    n_deleted: int
    feasible: bool
    margins: dict                    # spec_name -> {column: margin}
    fits: list = field(default_factory=list)
    iterations: int = 0

    @property
    def mask(self) -> np.ndarray:
        return self.keep


STATUS_FEASIBLE = "FEASIBLE"
STATUS_INFEASIBLE = "INFEASIBLE"
STATUS_BUDGET = "BUDGET_EXHAUSTED"
