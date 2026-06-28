"""Smoke tests for the CLI entry point."""
import os
import tempfile
import unittest
from active_canvas_lab.cli import main


class TestCLI(unittest.TestCase):

    def test_basic_run_returns_zero(self):
        code = main([
            "--dataset", "blobs", "--strategy", "uncertainty",
            "--budget", "10", "--seed", "0", "--n-samples", "80",
        ])
        self.assertEqual(code, 0)

    def test_run_with_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "report.html")
            code = main([
                "--dataset", "blobs", "--strategy", "random",
                "--budget", "10", "--seed", "7",
                "--n-samples", "80", "--report", path,
            ])
            self.assertEqual(code, 0)
            self.assertTrue(os.path.exists(path))

    def test_all_datasets(self):
        for ds in ("blobs", "rings", "diagonal", "islands"):
            with self.subTest(dataset=ds):
                code = main([
                    "--dataset", ds, "--budget", "8",
                    "--seed", "42", "--n-samples", "60",
                ])
                self.assertEqual(code, 0)

    def test_all_strategies(self):
        for st in ("random", "uncertainty", "margin", "diversity"):
            with self.subTest(strategy=st):
                code = main([
                    "--strategy", st, "--budget", "8",
                    "--seed", "42", "--n-samples", "60",
                ])
                self.assertEqual(code, 0)

    def test_invalid_dataset_exits(self):
        with self.assertRaises(SystemExit):
            main(["--dataset", "bogus"])

    def test_invalid_strategy_exits(self):
        with self.assertRaises(SystemExit):
            main(["--strategy", "bogus"])

    def test_deterministic_output(self):
        import io, sys
        buf1 = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf1
        main(["--dataset", "islands", "--strategy", "uncertainty",
              "--budget", "20", "--seed", "42", "--n-samples", "120"])
        sys.stdout = old_stdout

        buf2 = io.StringIO()
        sys.stdout = buf2
        main(["--dataset", "islands", "--strategy", "uncertainty",
              "--budget", "20", "--seed", "42", "--n-samples", "120"])
        sys.stdout = old_stdout

        self.assertEqual(buf1.getvalue(), buf2.getvalue())

    def test_report_nested_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "a", "b", "c", "out.html")
            code = main([
                "--dataset", "blobs", "--budget", "8",
                "--seed", "0", "--n-samples", "60", "--report", path,
            ])
            self.assertEqual(code, 0)
            self.assertTrue(os.path.exists(path))
