"""Tests for active learning query strategies."""
import random
import unittest
from active_canvas_lab.models import LogisticRegression
from active_canvas_lab.strategies import (
    DiversityStrategy,
    MarginStrategy,
    RandomStrategy,
    UncertaintyStrategy,
    get_strategy,
    STRATEGIES,
)


def _pool_and_model():
    X_lab = [[-1.0, 0.0], [1.0, 0.0]] * 5
    y_lab = [0, 1] * 5
    model = LogisticRegression(lr=0.2, n_epochs=100).fit(X_lab, y_lab)
    X_pool = [[float(i), float(j)] for i in range(-4, 5) for j in range(-4, 5)]
    return model, X_pool, X_lab


class TestStrategies(unittest.TestCase):

    def setUp(self):
        self.model, self.X_pool, self.X_lab = _pool_and_model()
        self.rng = random.Random(42)

    def _assert_valid(self, chosen, n_query):
        self.assertEqual(len(chosen), n_query, "wrong number of queries")
        self.assertEqual(len(set(chosen)), n_query, "duplicate indices")
        self.assertTrue(all(0 <= i < len(self.X_pool) for i in chosen))

    def test_random_valid(self):
        chosen = RandomStrategy().query(
            self.model, self.X_pool, 5, rng=self.rng
        )
        self._assert_valid(chosen, 5)

    def test_uncertainty_valid(self):
        chosen = UncertaintyStrategy().query(self.model, self.X_pool, 5)
        self._assert_valid(chosen, 5)

    def test_margin_valid(self):
        chosen = MarginStrategy().query(self.model, self.X_pool, 5)
        self._assert_valid(chosen, 5)

    def test_diversity_with_labeled(self):
        chosen = DiversityStrategy().query(
            self.model, self.X_pool, 5,
            X_labeled=self.X_lab, rng=self.rng,
        )
        self._assert_valid(chosen, 5)

    def test_diversity_no_labeled_falls_back(self):
        chosen = DiversityStrategy().query(
            self.model, self.X_pool, 5,
            X_labeled=None, rng=self.rng,
        )
        self._assert_valid(chosen, 5)

    def test_query_fewer_than_pool(self):
        pool = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]
        for Cls in (RandomStrategy, UncertaintyStrategy, MarginStrategy):
            chosen = Cls().query(self.model, pool, 10, rng=self.rng)
            self.assertLessEqual(len(chosen), len(pool))

    def test_get_strategy_returns_strategy(self):
        for name in STRATEGIES:
            strat = get_strategy(name)
            self.assertTrue(hasattr(strat, "query"))
            self.assertTrue(callable(strat.query))

    def test_get_strategy_unknown_raises(self):
        with self.assertRaises(ValueError):
            get_strategy("bogus")

    def test_random_is_deterministic(self):
        r1 = RandomStrategy().query(
            self.model, self.X_pool, 5, rng=random.Random(7)
        )
        r2 = RandomStrategy().query(
            self.model, self.X_pool, 5, rng=random.Random(7)
        )
        self.assertEqual(r1, r2)

    def test_uncertainty_picks_uncertain_points(self):
        """The points closest to the decision boundary should be chosen."""
        chosen = UncertaintyStrategy().query(self.model, self.X_pool, 3)
        probs = self.model.predict_proba(self.X_pool)
        uncertainty = [abs(probs[i] - 0.5) for i in chosen]
        not_chosen = [abs(probs[i] - 0.5) for i in range(len(self.X_pool)) if i not in chosen]
        self.assertLessEqual(max(uncertainty), min(not_chosen) + 1e-9)
