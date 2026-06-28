"""Compare all four query strategies on every dataset.

Run with:
    python examples/run_comparison.py

No extra dependencies — uses only the active_canvas_lab package.
Writes per-strategy HTML reports to examples/out/.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from active_canvas_lab.experiment import run_experiment
from active_canvas_lab.metrics import summarise
from active_canvas_lab.reports import write_report

DATASETS   = ["blobs", "diagonal", "islands", "rings"]
STRATEGIES = ["random", "uncertainty", "margin", "diversity"]
BUDGET     = 60
SEED       = 42
N_SAMPLES  = 300

out_dir = os.path.join(os.path.dirname(__file__), "out")

header = f"{'Dataset':<12} {'Strategy':<14} {'Init':>6} {'Final':>7} {'Peak':>6} {'AUC':>7}"
sep    = "-" * len(header)

print()
print("ActiveCanvas Lab — Strategy × Dataset Comparison")
print(f"budget={BUDGET}  seed={SEED}  n_samples={N_SAMPLES}")
print()
print(header)
print(sep)

for dataset in DATASETS:
    for strategy in STRATEGIES:
        result = run_experiment(
            dataset=dataset,
            strategy=strategy,
            budget=BUDGET,
            seed=SEED,
            n_samples=N_SAMPLES,
        )
        s = summarise(result["history"])

        report_path = os.path.join(out_dir, f"{dataset}_{strategy}.html")
        write_report(result, report_path)

        print(
            f"{dataset:<12} {strategy:<14}"
            f" {s['initial_accuracy']:>6.3f}"
            f" {s['final_accuracy']:>7.3f}"
            f" {s['peak_accuracy']:>6.3f}"
            f" {s['auc']:>7.4f}"
        )
    print(sep)

print()
print(f"Reports written to: {out_dir}/")
print()
