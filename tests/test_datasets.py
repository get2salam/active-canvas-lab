"""Tests for synthetic dataset generation."""
import unittest
from active_canvas_lab.datasets import load_dataset, DATASETS


class TestDatasets(unittest.TestCase):

    def test_all_datasets_correct_size(self):
        for name in DATASETS:
            with self.subTest(dataset=name):
                X, y = load_dataset(name, n_samples=60, seed=0)
                self.assertEqual(len(X), 60)
                self.assertEqual(len(y), 60)

    def test_all_points_are_2d(self):
        for name in DATASETS:
            X, _ = load_dataset(name, n_samples=40, seed=0)
            self.assertTrue(all(len(p) == 2 for p in X))

    def test_labels_are_binary(self):
        for name in DATASETS:
            with self.subTest(dataset=name):
                _, y = load_dataset(name, n_samples=100, seed=1)
                self.assertTrue(all(label in (0, 1) for label in y))

    def test_both_classes_present(self):
        for name in DATASETS:
            with self.subTest(dataset=name):
                _, y = load_dataset(name, n_samples=100, seed=2)
                self.assertIn(0, y)
                self.assertIn(1, y)

    def test_reproducible_same_seed(self):
        for name in DATASETS:
            X1, y1 = load_dataset(name, n_samples=50, seed=99)
            X2, y2 = load_dataset(name, n_samples=50, seed=99)
            self.assertEqual(X1, X2)
            self.assertEqual(y1, y2)

    def test_different_seeds_differ(self):
        X1, _ = load_dataset("blobs", n_samples=50, seed=0)
        X2, _ = load_dataset("blobs", n_samples=50, seed=1)
        self.assertNotEqual(X1, X2)

    def test_unknown_dataset_raises(self):
        with self.assertRaises(ValueError):
            load_dataset("does_not_exist")

    def test_large_sample_count(self):
        X, y = load_dataset("islands", n_samples=500, seed=0)
        self.assertEqual(len(X), 500)
        self.assertTrue(all(label in (0, 1) for label in y))
