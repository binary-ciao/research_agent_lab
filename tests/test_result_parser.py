from __future__ import annotations

from unittest import TestCase, main

from agents.result_parser import parse_experiment_output


class ResultParserTest(TestCase):
    def test_parses_ade_fde_colon_format(self):
        stdout = (
            "epoch 1: train loss 0.123\n"
            "epoch 2: train loss 0.098\n"
            "eval results: ADE: 0.35  FDE: 0.72  MR: 0.15\n"
        )
        r = parse_experiment_output(
            experiment_id="exp_1",
            stdout=stdout,
            stderr="",
            returncode=0,
            command="python main_led_nba.py --cfg led_virat_debug",
            duration_seconds=30.0,
        )
        self.assertEqual(r.status, "passed")
        self.assertTrue(r.smoke_passed)
        self.assertAlmostEqual(r.metrics.get("ade"), 0.35)
        self.assertAlmostEqual(r.metrics.get("fde"), 0.72)
        self.assertAlmostEqual(r.metrics.get("mr"), 0.15)

    def test_parses_equals_format(self):
        stdout = (
            "Training complete\n"
            "minADE = 0.28\n"
            "minFDE = 0.55\n"
        )
        r = parse_experiment_output(
            experiment_id="exp_2",
            stdout=stdout,
            stderr="",
            returncode=0,
            command="python train.py",
            duration_seconds=60.0,
        )
        self.assertEqual(r.status, "passed")
        self.assertAlmostEqual(r.metrics.get("minade"), 0.28)
        self.assertAlmostEqual(r.metrics.get("minfde"), 0.55)

    def test_error_on_nonzero_returncode(self):
        r = parse_experiment_output(
            experiment_id="exp_err",
            stdout="training started...",
            stderr="CUDA out of memory",
            returncode=1,
            command="python train.py",
            duration_seconds=2.0,
        )
        self.assertEqual(r.status, "error")
        self.assertFalse(r.smoke_passed)
        self.assertIn("CUDA", r.error_message)

    def test_failure_pattern_detected(self):
        stdout = "RuntimeError: shape mismatch at line 42\nTraceback (most recent call last):\n..."
        r = parse_experiment_output(
            experiment_id="exp_fail",
            stdout=stdout,
            stderr="",
            returncode=0,
            command="python train.py",
            duration_seconds=5.0,
        )
        self.assertEqual(r.status, "failed")
        self.assertFalse(r.smoke_passed)

    def test_unparsed_when_no_metrics(self):
        stdout = "all systems nominal\nprocessing complete\nhave a nice day\n"
        r = parse_experiment_output(
            experiment_id="exp_no_metrics",
            stdout=stdout,
            stderr="",
            returncode=0,
            command="python train.py",
            duration_seconds=1.0,
        )
        self.assertEqual(r.status, "unparsed")
        self.assertTrue(r.smoke_passed)

    def test_log_tail_truncated(self):
        long_stdout = "line " * 3000
        r = parse_experiment_output(
            experiment_id="exp_tail",
            stdout=long_stdout,
            stderr="",
            returncode=0,
            command="python train.py",
            duration_seconds=1.0,
        )
        self.assertLessEqual(len(r.log_tail), 2500)
        self.assertIn("line", r.log_tail)

    def test_parses_dash_prefix_format(self):
        stdout = (
            "--ADE(1s): 0.2417\t--FDE(1s): 0.1259\n"
            "--ADE(2s): 0.2502\t--FDE(2s): 0.1427\n"
        )
        r = parse_experiment_output(
            experiment_id="exp_dash",
            stdout=stdout,
            stderr="",
            returncode=0,
            command="python train.py",
            expected_metrics=["ADE", "FDE"],
            duration_seconds=1.0,
        )
        self.assertEqual(r.status, "passed")
        self.assertAlmostEqual(r.metrics.get("ade"), 0.2417)

    def test_unparsed_when_metrics_dont_match_expected(self):
        stdout = "loss: 0.123\taccuracy: 0.95"
        r = parse_experiment_output(
            experiment_id="exp_mismatch",
            stdout=stdout,
            stderr="",
            returncode=0,
            command="python train.py",
            expected_metrics=["ADE", "FDE"],
            duration_seconds=1.0,
        )
        self.assertEqual(r.status, "unparsed")

    def test_duplicate_metric_keeps_first(self):
        stdout = "ADE: 0.40\nADE: 0.30\n"
        r = parse_experiment_output(
            experiment_id="exp_dup",
            stdout=stdout,
            stderr="",
            returncode=0,
            command="python train.py",
            duration_seconds=1.0,
        )
        self.assertAlmostEqual(r.metrics["ade"], 0.40)


if __name__ == "__main__":
    main()
