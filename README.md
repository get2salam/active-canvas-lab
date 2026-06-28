# ActiveCanvas Lab

**A tiny, dependency-free active learning playground** built from scratch in pure Python.

Train simple models on synthetic 2D data, compare query strategies, and generate a self-contained visual HTML report — all with zero third-party dependencies.

## Why active learning?

Labeling data is expensive. Active learning lets a model *ask* which examples are most informative, achieving high accuracy with far fewer labeled samples than random labeling. This repo makes those dynamics visible.

## Install

```bash
git clone https://github.com/get2salam/active-canvas-lab
cd active-canvas-lab
pip install -e .
```

Requires Python ≥ 3.9. No external packages needed.

## Quick start

```bash
# Run uncertainty sampling on the islands dataset
python -m active_canvas_lab --dataset islands --strategy uncertainty --budget 40 --seed 42 --report out/report.html

# Compare diversity sampling
python -m active_canvas_lab --dataset blobs --strategy diversity --budget 30 --seed 7 --report out/diversity.html

# See all options
python -m active_canvas_lab --help
```

## Sample output

```
ActiveCanvas Lab
  dataset   : islands
  strategy  : uncertainty
  seed      : 42
  budget    : 40

  initial accuracy : 0.653
  final accuracy   : 0.880
  peak accuracy    : 0.880
  AUC              : 0.812
  labels used      : 50
  rounds           : 9
```

The `--report` flag writes a self-contained HTML file with:
- Scatter plot of training data with the labeled subset highlighted
- Learning curve (accuracy vs. labeled samples)
- Summary statistics table

## Datasets

| Name | Description | Notes |
|------|-------------|-------|
| `blobs` | Two Gaussian clusters | Linearly separable |
| `rings` | Concentric rings | Not linearly separable |
| `diagonal` | Linear boundary along x+y=0 | Easy for logistic regression |
| `islands` | Six micro-clusters (3 per class) | Challenging geography |

## Query strategies

| Strategy | Idea | When it shines |
|----------|------|----------------|
| `random` | Sample uniformly at random | Baseline |
| `uncertainty` | Pick samples with predicted probability closest to 0.5 | Most uncertain examples |
| `margin` | Minimise &#124;2p−1&#124; (same as uncertainty for binary) | Binary classification |
| `diversity` | Greedy farthest-first from already-labeled set | Exploration / coverage |

## Run tests

```bash
python -m unittest discover -s tests -v
```

## Run the comparison example

```bash
python examples/run_comparison.py
```

## Architecture

```
active_canvas_lab/
├── datasets.py    # Deterministic synthetic 2D datasets (pure Python, Box-Muller)
├── models.py      # From-scratch logistic regression (SGD + L2)
├── strategies.py  # Random / uncertainty / margin / diversity query strategies
├── experiment.py  # Active learning loop — seed → query → train → evaluate
├── metrics.py     # Accuracy, confusion matrix, AUC of learning curve
├── reports.py     # Self-contained HTML reports with inline SVG charts
├── cli.py         # argparse CLI
└── __main__.py    # python -m active_canvas_lab entry point
```

## How it works

1. A dataset is generated deterministically from a seed.
2. A small **seed set** (default 10) is labeled randomly.
3. A logistic regression model is trained on the seed set.
4. The **query strategy** picks the next `batch` most informative unlabeled points.
5. Their labels are revealed and added to the training set.
6. Steps 3–5 repeat until the label budget is exhausted.
7. A learning curve shows how accuracy improves with each labeling round.

## License

MIT
