"""Tests for metrics module."""
import unittest
from active_canvas_lab.metrics import (
    accuracy,
    auc_learning_curve,
    confusion_matrix,
    summarise,
)


class TestAccuracy(unittest.TestCase):

    def test_perfect(self):
        self.assertAlmostEqual(accuracy([0, 1, 0, 1], [0, 1, 0, 1]), 1.0)

    def test_zero(self):
        self.assertAlmostEqual(accuracy([0, 1], [1, 0]), 0.0)

    def test_half(self):
        self.assertAlmostEqual(accuracy([0, 0, 1, 1], [0, 1, 0, 1]), 0.5)

    def test_empty(self):
        self.assertAlmostEqual(accuracy([], []), 0.0)


class TestConfusionMatrix(unittest.TestCase):

    def test_perfect_predictions(self):
        cm = confusion_matrix([0, 1, 0, 1], [0, 1, 0, 1])
        self.assertEqual(cm[(0, 0)], 2)
        self.assertEqual(cm[(1, 1)], 2)
        self.assertEqual(cm[(0, 1)], 0)
        self.assertEqual(cm[(1, 0)], 0)

    def test_all_wrong(self):
        cm = confusion_matrix([0, 1], [1, 0])
        self.assertEqual(cm[(0, 1)], 1)
        self.assertEqual(cm[(1, 0)], 1)


class TestAUC(unittest.TestCase):

    def test_constant_accuracy_is_that_value(self):
        history = [
            {"n_labeled": 10, "accuracy": 0.8},
            {"n_labeled": 20, "accuracy": 0.8},
        ]
        self.assertAlmostEqual(auc_learning_curve(history), 0.8)

    def test_rising_curve(self):
        history = [
            {"n_labeled": 10, "accuracy": 0.5},
            {"n_labeled": 20, "accuracy": 1.0},
        ]
        # Trapezoid: 0.5*(0.5+1.0)*1 = 0.75
        self.assertAlmostEqual(auc_learning_curve(history), 0.75)

    def test_single_point(self):
        history = [{"n_labeled": 10, "accuracy": 0.6}]
        self.assertAlmostEqual(auc_learning_curve(history), 0.6)

    def test_empty(self):
        self.assertAlmostEqual(auc_learning_curve([]), 0.0)


class TestSummarise(unittest.TestCase):

    def test_keys_present(self):
        history = [
            {"n_labeled": 10, "accuracy": 0.5},
            {"n_labeled": 15, "accuracy": 0.7},
            {"n_labeled": 20, "accuracy": 0.65},
        ]
        s = summarise(history)
        for key in ("initial_accuracy", "final_accuracy", "peak_accuracy",
                    "n_labeled_final", "auc", "n_rounds"):
            self.assertIn(key, s)

    def test_values_correct(self):
        history = [
            {"n_labeled": 10, "accuracy": 0.5},
            {"n_labeled": 20, "accuracy": 0.9},
        ]
        s = summarise(history)
        self.assertAlmostEqual(s["initial_accuracy"], 0.5)
        self.assertAlmostEqual(s["final_accuracy"],   0.9)
        self.assertAlmostEqual(s["peak_accuracy"],    0.9)
        self.assertEqual(s["n_labeled_final"], 20)
        self.assertEqual(s["n_rounds"], 2)

    def test_empty_returns_empty(self):
        self.assertEqual(summarise([]), {})
