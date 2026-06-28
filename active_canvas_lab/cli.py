"""Command-line interface for ActiveCanvas Lab.

Usage:
    python -m active_canvas_lab [options]

Run `python -m active_canvas_lab --help` for full option list.
"""
import argparse
import sys
from typing import Optional, List

from .datasets import DATASETS
from .experiment import run_experiment
from .metrics import summarise
from .reports import write_report
from .strategies import STRATEGIES


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="active_canvas_lab",
        description=(
            "ActiveCanvas Lab — active learning on synthetic 2D data.\n"
            "Train a logistic regression model using different query strategies\n"
            "and compare how quickly each reaches high accuracy."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--dataset", choices=sorted(DATASETS), default="blobs",
        metavar="NAME",
        help=f"Synthetic dataset. Choices: {sorted(DATASETS)}  (default: blobs)",
    )
    p.add_argument(
        "--strategy", choices=sorted(STRATEGIES), default="uncertainty",
        metavar="NAME",
        help=f"Query strategy. Choices: {sorted(STRATEGIES)}  (default: uncertainty)",
    )
    p.add_argument(
        "--budget", type=int, default=40,
        help="Maximum new labels to query beyond the seed set (default: 40)",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for full reproducibility (default: 42)",
    )
    p.add_argument(
        "--n-samples", type=int, default=300, dest="n_samples",
        help="Total dataset size to generate (default: 300)",
    )
    p.add_argument(
        "--seed-size", type=int, default=10, dest="seed_size",
        help="Initial labeled seed set size (default: 10)",
    )
    p.add_argument(
        "--batch", type=int, default=5,
        help="Number of points queried per round (default: 5)",
    )
    p.add_argument(
        "--report", default=None, metavar="PATH",
        help="Write a self-contained HTML report to this path (optional)",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    result = run_experiment(
        dataset=args.dataset,
        strategy=args.strategy,
        budget=args.budget,
        seed=args.seed,
        n_samples=args.n_samples,
        seed_size=args.seed_size,
        batch_size=args.batch,
    )

    stats = summarise(result["history"])

    print()
    print("ActiveCanvas Lab")
    print(f"  dataset   : {args.dataset}")
    print(f"  strategy  : {args.strategy}")
    print(f"  seed      : {args.seed}")
    print(f"  budget    : {args.budget}")
    print()
    print(f"  initial accuracy : {stats.get('initial_accuracy', 0.0):.3f}")
    print(f"  final accuracy   : {stats.get('final_accuracy',   0.0):.3f}")
    print(f"  peak accuracy    : {stats.get('peak_accuracy',    0.0):.3f}")
    print(f"  AUC              : {stats.get('auc',              0.0):.3f}")
    print(f"  labels used      : {stats.get('n_labeled_final',    0)}")
    print(f"  rounds           : {stats.get('n_rounds',           0)}")

    if args.report:
        path = write_report(result, args.report)
        print(f"\n  report           : {path}")

    print()
    return 0
