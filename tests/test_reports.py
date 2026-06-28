"""Tests for HTML report generation."""
import os
import tempfile
import unittest
from active_canvas_lab.experiment import run_experiment
from active_canvas_lab.reports import generate_html, write_report


class TestReports(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.result = run_experiment(
            dataset="blobs", strategy="uncertainty",
            budget=10, seed=0, n_samples=60,
        )

    def test_generate_returns_string(self):
        html = generate_html(self.result)
        self.assertIsInstance(html, str)

    def test_html_doctype(self):
        html = generate_html(self.result)
        self.assertTrue(html.strip().startswith("<!DOCTYPE html>"))

    def test_contains_svg(self):
        html = generate_html(self.result)
        self.assertIn("<svg", html)

    def test_contains_strategy_name(self):
        html = generate_html(self.result)
        self.assertIn("uncertainty", html)

    def test_contains_dataset_name(self):
        html = generate_html(self.result)
        self.assertIn("blobs", html)

    def test_contains_title(self):
        html = generate_html(self.result)
        self.assertIn("ActiveCanvas Lab", html)

    def test_write_creates_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "nested", "dir", "report.html")
            write_report(self.result, path)
            self.assertTrue(os.path.exists(path))

    def test_written_content_is_html(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "r.html")
            write_report(self.result, path)
            with open(path, encoding="utf-8") as fh:
                content = fh.read()
            self.assertIn("<!DOCTYPE html>", content)
            self.assertIn("<svg", content)

    def test_all_datasets_generate(self):
        for ds in ("blobs", "rings", "diagonal", "islands"):
            with self.subTest(dataset=ds):
                r = run_experiment(dataset=ds, budget=8, seed=0, n_samples=50)
                html = generate_html(r)
                self.assertIn("<svg", html)
