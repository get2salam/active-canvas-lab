"""Tests for the active learning experiment loop."""
import unittest
from active_canvas_lab.datasets import load_dataset
from active_canvas_lab.experiment import ActiveLearningExperiment, run_experiment
from active_canvas_lab.models import LogisticRegression
from active_canvas_lab.strategies import get_strategy


def _make_exp(strategy="uncertainty", budget=12, seed=0):
    X, y = load_dataset("blobs", n_samples=80, seed=seed)
    X_train, y_train = X[:60], y[:60]
    X_test,  y_test  = X[60:], y[60:]
    return ActiveLearningExperiment(
        X_train, y_train, X_test, y_test,
        model=LogisticRegression(),
        strategy=get_strategy(strategy),
        seed_size=6, batch_size=4, budget=budget, seed=seed,
    )


class TestActiveExperiment(unittest.TestCase):

    def test_history_has_entries(self):
        exp = _make_exp()
        history = exp.run()
        self.assertGreater(len(history), 0)

    def test_history_has_two_keys(self):
        history = _make_exp().run()
        for h in history:
            self.assertIn("n_labeled", h)
            self.assertIn("accuracy", h)

    def test_n_labeled_monotone(self):
        history = _make_exp().run()
        ns = [h["n_labeled"] for h in history]
        self.assertEqual(ns, sorted(ns))

    def test_accuracy_in_range(self):
        history = _make_exp().run()
        for h in history:
            self.assertGreaterEqual(h["accuracy"], 0.0)
            self.assertLessEqual(h["accuracy"], 1.0)

    def test_budget_respected(self):
        exp = _make_exp(budget=8)
        history = exp.run()
        initial_seed = 6
        final_labeled = history[-1]["n_labeled"]
        self.assertLessEqual(final_labeled - initial_seed, 8 + 4)  # within one batch

    def test_all_strategies(self):
        for st in ("random", "uncertainty", "margin", "diversity"):
            with self.subTest(strategy=st):
                history = _make_exp(strategy=st).run()
                self.assertGreater(len(history), 0)

    def test_run_experiment_function(self):
        result = run_experiment(
            dataset="diagonal", strategy="margin",
            budget=15, seed=1, n_samples=80,
        )
        self.assertIn("history", result)
        self.assertIn("X_labeled", result)
        self.assertIn("X_train", result)
        self.assertIn("X_test", result)
        self.assertGreater(len(result["history"]), 0)

    def test_run_experiment_all_datasets(self):
        for ds in ("blobs", "rings", "diagonal", "islands"):
            with self.subTest(dataset=ds):
                result = run_experiment(
                    dataset=ds, budget=10, seed=0, n_samples=60,
                )
                self.assertGreater(len(result["history"]), 0)

    def test_deterministic(self):
        r1 = run_experiment(dataset="blobs", strategy="uncertainty",
                            budget=20, seed=7, n_samples=100)
        r2 = run_experiment(dataset="blobs", strategy="uncertainty",
                            budget=20, seed=7, n_samples=100)
        self.assertEqual(r1["history"], r2["history"])

    def test_seed_size_ge_train_size_raises_with_positive_budget(self):
        # n_samples=10 with the default 0.25 test_ratio reserves 2 samples
        # for testing, leaving only 8 for train/pool. seed_size=9 satisfies
        # the CLI's "seed_size < n_samples" check yet still swallows the
        # entire train set, leaving nothing to query.
        with self.assertRaises(ValueError):
            run_experiment(
                dataset="blobs", strategy="uncertainty",
                budget=40, seed=42, n_samples=10, seed_size=9,
            )

    def test_seed_size_ge_train_size_allowed_with_zero_budget(self):
        # A zero budget means no querying was ever going to happen, so an
        # empty pool is not misleading and should not be rejected.
        result = run_experiment(
            dataset="blobs", strategy="uncertainty",
            budget=0, seed=42, n_samples=10, seed_size=9,
        )
        self.assertGreater(len(result["history"]), 0)

    def test_active_learning_experiment_raises_on_empty_pool(self):
        X, y = load_dataset("blobs", n_samples=10, seed=0)
        X_train, y_train = X[:8], y[:8]
        X_test, y_test = X[8:], y[8:]
        with self.assertRaises(ValueError):
            ActiveLearningExperiment(
                X_train, y_train, X_test, y_test,
                model=LogisticRegression(),
                strategy=get_strategy("uncertainty"),
                seed_size=8, batch_size=4, budget=10, seed=0,
            )
