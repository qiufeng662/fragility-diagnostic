"""Run the full replication pipeline end-to-end, in dependency order.

Each step is a standalone script; this runner executes them sequentially and
stops at the first failure.  Expected wall-clock is a few minutes on a
typical laptop.
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

STEPS = [
    "reproduce.py",            # Section 8 empirical numbers (both panels)
    "make_figs.py",            # fig1/2/3 (firm-level panel)
    "make_sim_fig1.py",        # fig_sim_combined (Figure 1, 2x2 simulation)
    "mc_study.py",             # figures/mc_results.csv (Monte-Carlo CSV for redraw_figs)
    "redraw_figs.py",          # remaining simulation figures
    "make_compare_figs.py",    # sim4
    "make_mc_compare.py",      # sim5
    "make_listed_fig.py",      # fig4 (listed-firm panel)
    "numeric_compare.py",      # Section 2 numerical check
    "pruning_recall.py",       # Section 7 pruning check
    "mc_eval.py",              # Section 7 Monte-Carlo method evaluation
    "mc_eval_sig.py",          # Section 7 full-significance (CR1 margin) evaluation
    "mc_scalability.py",       # Section 7 pruned-search scalability table
    "mc_method_compare.py",    # Section 7 method comparison table
    "freeze_x2.py",            # Section 8 fragile-coefficient evidence
]

for s in STEPS:
    t0 = time.time()
    print(f"\n===== {s} =====", flush=True)
    r = subprocess.run([sys.executable, os.path.join(HERE, s)], cwd=HERE)
    if r.returncode != 0:
        print(f"FAILED: {s} (exit {r.returncode})", flush=True)
        sys.exit(r.returncode)
    print(f"----- {s} done in {time.time() - t0:.1f}s -----", flush=True)

print("\nAll replication steps completed.")
