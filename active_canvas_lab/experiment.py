"""Active learning experiment loop.

Manages the labeled/unlabeled split, queries the strategy for the next
batch, trains the model, and records accuracy at each round.
"""
import random
from typing import Dict, List, Any

from .datasets import load_dataset
from .models import LogisticRegression
from .strategies import get_strategy


def _split(X: list, y: list, test_ratio: float, rng: random.Random):
    idx = list(range(len(X)))
    rng.shuffle(idx)
    n_test = max(2, int(len(X) * test_ratio))
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return (
        [X[i] for i in train_idx], [y[i] for i in train_idx],
        [X[i] for i in test_idx],  [y[i] for i in test_idx],
    )


class ActiveLearningExperiment:
    """Runs the active learning loop on a fixed train/test split."""

    def __init__(
        self,
        X_train: list, y_train: list,
        X_test: list, y_test: list,
        model, strategy,
        seed_size: int = 10,
        batch_size: int = 5,
        budget: int = 50,
        seed: int = 0,
    ):
        if budget > 0 and seed_size >= len(X_train):
            raise ValueError(
                f"seed_size ({seed_size}) leaves no samples in the pool for "
                f"a training set of size {len(X_train)}, so the requested "
                f"budget ({budget}) can never be spent. This can happen "
                f"even when seed_size < n_samples, because the test split "
                f"is carved out of n_samples before the train/pool split. "
                f"Reduce --seed-size or increase --n-samples."
            )
        rng = random.Random(seed)
        idx = list(range(len(X_train)))
        rng.shuffle(idx)
        seed_idx, pool_idx = idx[:seed_size], idx[seed_size:]

        self.X_lab  = [X_train[i] for i in seed_idx]
        self.y_lab  = [y_train[i] for i in seed_idx]
        self.X_pool = [X_train[i] for i in pool_idx]
        self.y_pool = [y_train[i] for i in pool_idx]

        self.X_test  = X_test
        self.y_test  = y_test
        self.model   = model
        self.strategy = strategy
        self.batch   = batch_size
        self.budget  = budget
        self.rng     = rng
        self.history: List[Dict[str, Any]] = []

    def run(self) -> List[Dict[str, Any]]:
        """Execute the full label-budget loop; return history."""
        queried = 0
        while queried < self.budget and self.X_pool:
            self.model.fit(self.X_lab, self.y_lab)
            self.history.append({
                "n_labeled": len(self.X_lab),
                "accuracy":  self.model.score(self.X_test, self.y_test),
            })
            batch = min(self.batch, len(self.X_pool), self.budget - queried)
            if batch <= 0:
                break
            chosen = self.strategy.query(
                self.model, self.X_pool, batch,
                X_labeled=self.X_lab, rng=self.rng,
            )
            for i in sorted(chosen, reverse=True):
                self.X_lab.append(self.X_pool.pop(i))
                self.y_lab.append(self.y_pool.pop(i))
            queried += len(chosen)

        # Final evaluation after the last batch has been added
        if self.X_lab:
            self.model.fit(self.X_lab, self.y_lab)
            self.history.append({
                "n_labeled": len(self.X_lab),
                "accuracy":  self.model.score(self.X_test, self.y_test),
            })
        return self.history


def run_experiment(
    dataset: str = "blobs",
    strategy: str = "uncertainty",
    budget: int = 40,
    seed: int = 42,
    n_samples: int = 300,
    seed_size: int = 10,
    batch_size: int = 5,
    test_ratio: float = 0.25,
) -> Dict[str, Any]:
    """Convenience wrapper that creates everything from config and runs.

    Returns a result dict with history, data splits, labeled set, and
    metadata — everything reports.py needs to generate a visual.
    """
    rng = random.Random(seed)
    X, y = load_dataset(dataset, n_samples=n_samples, seed=seed)
    X_train, y_train, X_test, y_test = _split(X, y, test_ratio, rng)

    model  = LogisticRegression()
    strat  = get_strategy(strategy)
    exp    = ActiveLearningExperiment(
        X_train, y_train, X_test, y_test,
        model=model, strategy=strat,
        seed_size=seed_size, batch_size=batch_size,
        budget=budget, seed=seed,
    )
    history = exp.run()
    return {
        "history":   history,
        "X_train":   X_train,
        "y_train":   y_train,
        "X_test":    X_test,
        "y_test":    y_test,
        "X_labeled": exp.X_lab,
        "y_labeled": exp.y_lab,
        "model":     model,
        "strategy":  strategy,
        "dataset":   dataset,
        "seed":      seed,
        "budget":    budget,
    }
