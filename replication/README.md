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
- `make_figs.py`, `make_sim_fig1.py`, `redraw_figs.py`, `make_compare_figs.py`, `make_mc_compare.py`, `make_listed_fig.py` —
  generate the paper figures (empirical panels, the unified Figure 1 simulation
  panel, and the remaining simulations). `redraw_figs.py` reads
  `figures/mc_results.csv`, which `mc_study.py` (fixed seed, relative paths)
  generates into the same directory.
- `mc_study.py` — the Monte-Carlo study behind `figures/mc_results.csv`
  (planted-leverage recall / false-positive rates; fixed per-replicate seeds).
- `numeric_compare.py`, `pruning_recall.py`, `mc_eval.py`, `mc_scalability.py` — the numerical checks
  (determinant/adjugate vs. direct solves; candidate-pruning recall; Monte-Carlo
  method evaluation with threshold sensitivity and optimality gap; pruned-search
  scalability table).
- `freeze_x2.py` — the fragile-coefficient evidence of Section 8 (the x2 coefficient
  whose t-statistic falls from 14.50 to -0.05 after deleting one 5-row cluster).
- `diag_panel2.py` — the full diagnostic (Algorithm 1) applied to the listed-firm
  panel: leverage profile, box certificate, first-order distortion, search
  trajectory, and an informal AMIP (first-order influence) comparison.
- `detadj_refit_gap.py` — quantifies the determinant/adjugate vs exact re-fit
  discrepancy under fixed-effect re-absorption (Section 2).
- `make_listed_fig.py` — generates the listed-firm diagnostic figure (fig4).
- `reproduce_all.py` — runs every step above in order.
- `test_diagnostic.py` — minimal unit tests (coordinate invariance, determinant/adjugate
  vs re-fit equivalence, knapsack bound). Run `python -m pytest test_diagnostic.py`.

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
python reproduce_all.py        # runs everything below, in order

# individual steps:
python reproduce.py            # Section 8 empirical numbers
python make_figs.py            # fig1/2/3 (firm-level panel)
python make_sim_fig1.py        # fig_sim_combined (Figure 1)
python mc_study.py            # figures/mc_results.csv (needed by redraw_figs.py)
python redraw_figs.py          # remaining simulation figures
python make_compare_figs.py    # sim4
python make_mc_compare.py      # sim5
python make_listed_fig.py      # fig4 (listed-firm panel)
python numeric_compare.py      # Section 2 numerical check
python pruning_recall.py       # Section 7 pruning check
python mc_eval.py              # Section 7 Monte-Carlo method evaluation
python mc_scalability.py       # Section 7 pruned-search scalability table
python freeze_x2.py            # Section 8 fragile-coefficient evidence
python diag_panel2.py          # Section 8 listed-firm full diagnostic
python detadj_refit_gap.py     # Section 2 det/adjugate vs re-fit gap
python -m pytest test_diagnostic.py   # unit tests
```

## Expected output (tolerances)

The key reported numbers and the tolerance to which they should reproduce
(floating-point differences across BLAS/LAPACK builds are at most a few units in
the last shown digit):

- Panel 1 (firm-level): $\hat\beta(x_6)=1.4077$, $t=5.114$; top leverage
  $h_g = 0.997, 0.714, 0.631$; sign flip at $734$ rows ($87$ clusters).
- Panel 2 (listed-firm): $\hat\beta(x_1)=-0.0023$, $t=-0.508$; sign flip at $6$
  rows ($1$ cluster); significance ($|t|>2$) at $55$ rows ($9$ clusters),
  $t=2.110$.
- Fragile coefficient (panel 1, x2): $t$ falls from $14.50$ to $-0.05$ after
  deleting one $5$-row cluster.
- Determinant/adjugate vs re-fit gap (Section 2): on the firm panel the gap is
  $\le 0.05$ across the top-$15$ leverage clusters and $\det Q(x)$ becomes
  non-positive-definite at $\approx 190$ rows; on the listed-firm panel the gap
  at the $55$-row significance set is $\le 10^{-3}$.

Requirements: Python 3.9+ with `numpy`, `scipy`, `pandas`, `matplotlib`.
