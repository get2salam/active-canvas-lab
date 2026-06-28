"""Tests for the from-scratch logistic regression model."""
import unittest
from active_canvas_lab.models import LogisticRegression


def _easy_data():
    """Trivially linearly separable dataset."""
    X = [[-2.0, 0.0], [-1.5, 0.0], [1.5, 0.0], [2.0, 0.0]] * 12
    y = [0, 0, 1, 1] * 12
    return X, y


class TestLogisticRegression(unittest.TestCase):

    def test_predict_before_fit_returns_neutral(self):
        model = LogisticRegression()
        probs = model.predict_proba([[0.0, 0.0], [1.0, 1.0]])
        self.assertTrue(all(abs(p - 0.5) < 1e-9 for p in probs))

    def test_fit_returns_self(self):
        X, y = _easy_data()
        model = LogisticRegression()
        result = model.fit(X, y)
        self.assertIs(result, model)

    def test_fit_empty_is_safe(self):
        model = LogisticRegression()
        result = model.fit([], [])
        self.assertIs(result, model)

    def test_predict_shape(self):
        X, y = _easy_data()
        model = LogisticRegression().fit(X, y)
        preds = model.predict(X)
        self.assertEqual(len(preds), len(X))

    def test_predict_values_binary(self):
        X, y = _easy_data()
        model = LogisticRegression().fit(X, y)
        preds = model.predict(X)
        self.assertTrue(all(p in (0, 1) for p in preds))

    def test_proba_in_unit_interval(self):
        X, y = _easy_data()
        model = LogisticRegression().fit(X, y)
        probs = model.predict_proba(X)
        self.assertTrue(all(0.0 <= p <= 1.0 for p in probs))

    def test_convergence_easy_data(self):
        X, y = _easy_data()
        model = LogisticRegression(lr=0.1, n_epochs=300).fit(X, y)
        self.assertGreater(model.score(X, y), 0.95)

    def test_score_empty_returns_zero(self):
        model = LogisticRegression()
        self.assertEqual(model.score([], []), 0.0)

    def test_incremental_improvement(self):
        """More epochs on easy data should not decrease accuracy."""
        X, y = _easy_data()
        m1 = LogisticRegression(lr=0.05, n_epochs=10).fit(X, y)
        m2 = LogisticRegression(lr=0.05, n_epochs=200).fit(X, y)
        self.assertGreaterEqual(m2.score(X, y), m1.score(X, y) - 0.05)

    def test_weight_vector_length_matches_features(self):
        X, y = _easy_data()
        model = LogisticRegression().fit(X, y)
        self.assertEqual(len(model.w), 2)
