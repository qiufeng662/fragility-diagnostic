# Replication package

This directory reproduces the empirical results of the paper from anonymized data
and a minimal implementation of the diagnostic.

## Contents

- `optmask/` — the diagnostic implementation used in the paper:
  - `certify.py` — determinant/adjugate evaluation of the post-deletion margin
  - `diagnostics.py` — the diagnostic report (Algorithm 1)
  - `search.py` — the branch-and-bound minimal-deletion search (Algorithm 2)
  - `cluster_valid.py`, `estimators.py`, `influence.py`, `contract.py` — supporting
    two-way fixed-effects estimator and target abstraction
- `data/` — the two anonymized panels:
  - `panel_firm.csv` — firm-level panel (Section 8, robust conclusion)
  - `panel_listed.csv` — listed-firm panel (Section 8, fragile conclusion)
- `reproduce.py` — reproduces the Section 8 numbers.
- `make_figs.py`, `redraw_figs.py`, `make_compare_figs.py`, `make_mc_compare.py` —
  generate the nine paper figures (empirical panels and simulations).
- `numeric_compare.py`, `pruning_recall.py` — the two numerical checks in Section 2
  and Section 7 (determinant/adjugate vs. direct solves; candidate-pruning recall).

## Anonymization

The underlying data are third-party research data. Firm identifiers are replaced by
random integers and variable names by generic labels (`y`, `x1`, ..., `year`); all
numeric values are preserved, so the reported results reproduce exactly.

## Environment

The reported numbers were produced with Python 3.12, numpy 2.5.0, scipy, and pandas,
on an 11th-generation Intel Core i5-11300H @ 3.10GHz, using numpy/scipy's default
LAPACK linear-algebra backend.

## Reproducibility details

- Random seed: `20261005` for the anonymization mapping; each simulation script
  fixes its own seed.
- Algorithm parameters: leverage threshold $\tau = 0.5$ in the diagnostic report;
  the branch-and-bound search uses candidate pruning to high-leverage or
  high-influence clusters plus an incumbent upper bound, and evaluates every leaf by
  exact re-fitting.
- Terminology. The paper distinguishes four objects: the exact determinant/adjugate
  evaluator (valid for fixed designs), exact re-fitting (used when fixed effects are
  re-absorbed), the pruned branch-and-bound search (which returns feasible solutions,
  i.e. upper bounds on the minimum), and a certified optimum (obtained only by the
  unpruned exhaustive search). The deletion sets reported in Section 8 are pruned-search
  feasible solutions, not certified optima.

## Run

```bash
python reproduce.py            # Section 8 empirical numbers
python make_figs.py            # fig1/2/3 (firm-level panel)
python redraw_figs.py          # sim1/2/3 + fig_mc1
python make_compare_figs.py    # sim4
python make_mc_compare.py      # sim5
python numeric_compare.py      # Section 2 numerical check
python pruning_recall.py       # Section 7 pruning check
```

Requirements: Python 3.9+ with `numpy`, `scipy`, `pandas`, `matplotlib`.
