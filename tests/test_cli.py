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


class TestCLIArgumentValidation(unittest.TestCase):
    """A negative/zero numeric argument must be rejected at the CLI boundary
    rather than silently producing a degenerate or misleading result (see
    _validate_args in cli.py).
    """

    def test_negative_budget_exits(self):
        with self.assertRaises(SystemExit):
            main(["--dataset", "blobs", "--n-samples", "60", "--budget", "-5"])

    def test_zero_n_samples_exits(self):
        with self.assertRaises(SystemExit):
            main(["--dataset", "blobs", "--n-samples", "0"])

    def test_negative_n_samples_exits(self):
        with self.assertRaises(SystemExit):
            main(["--dataset", "blobs", "--n-samples", "-10"])

    def test_zero_seed_size_exits(self):
        with self.assertRaises(SystemExit):
            main(["--dataset", "blobs", "--n-samples", "60", "--seed-size", "0"])

    def test_negative_seed_size_exits(self):
        with self.assertRaises(SystemExit):
            main(["--dataset", "blobs", "--n-samples", "60", "--seed-size", "-100"])

    def test_seed_size_ge_n_samples_exits(self):
        with self.assertRaises(SystemExit):
            main(["--dataset", "blobs", "--n-samples", "60", "--seed-size", "60"])

    def test_zero_batch_exits(self):
        with self.assertRaises(SystemExit):
            main(["--dataset", "blobs", "--n-samples", "60", "--batch", "0"])

    def test_negative_batch_exits(self):
        with self.assertRaises(SystemExit):
            main(["--dataset", "blobs", "--n-samples", "60", "--batch", "-1"])

    def test_zero_budget_is_allowed(self):
        code = main(["--dataset", "blobs", "--n-samples", "60", "--budget", "0"])
        self.assertEqual(code, 0)

    def test_seed_size_just_below_train_size_is_allowed(self):
        # n_samples=60 with the default 0.25 test split reserves 15 samples
        # for testing, leaving a 45-sample train set. seed_size=44 is the
        # largest value that still leaves one sample in the pool.
        code = main([
            "--dataset", "blobs", "--n-samples", "60",
            "--seed-size", "44", "--budget", "5",
        ])
        self.assertEqual(code, 0)

    def test_seed_size_below_n_samples_but_ge_train_size_exits(self):
        # --seed-size 59 satisfies the "seed_size < n_samples" check on its
        # own, but the 0.25 test split shrinks the actual train set to 45
        # samples, so the seed set would swallow the whole pool and the
        # requested budget could never be spent. This must fail loudly
        # instead of silently reporting a zero-round "success".
        with self.assertRaises(SystemExit):
            main([
                "--dataset", "blobs", "--n-samples", "60",
                "--seed-size", "59", "--budget", "5",
            ])
