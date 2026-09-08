from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout

from src.prioritization.pipeline import calculate_priority, main


class PrioritizationCliTests(unittest.TestCase):
    def test_cli_accepts_real_numeric_inputs_and_prints_priority_score(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            main([
                "--fault-risk", "0.6",
                "--degradation-rate", "0.1",
                "--access-cost", "0.3",
                "--criticality", "0.7",
            ])
        output = buffer.getvalue().strip()
        self.assertTrue(output.startswith("PriorityScore="))

    def test_cli_invokes_existing_prioritization_logic_not_a_reimplementation(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            main([
                "--fault-risk", "0.6",
                "--degradation-rate", "0.1",
                "--access-cost", "0.3",
                "--criticality", "0.7",
            ])
        printed_score = float(buffer.getvalue().strip().split("=", 1)[1])

        expected_score = calculate_priority(
            fault_risk=0.6, degradation_rate=0.1, access_cost=0.3, criticality=0.7
        )
        self.assertAlmostEqual(printed_score, expected_score)

    def test_cli_respects_custom_weight_overrides(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            main([
                "--fault-risk", "0.6",
                "--degradation-rate", "0.1",
                "--access-cost", "0.3",
                "--criticality", "0.7",
                "--w-fault-risk", "2.0",
            ])
        printed_score = float(buffer.getvalue().strip().split("=", 1)[1])
        # w1=2.0 overridden, w2/w3/w4 remain the PrioritizationWeights default (1.0)
        self.assertAlmostEqual(printed_score, 2.0 * 0.6 + 0.1 - 0.3 + 0.7)

    def test_cli_does_not_fall_back_to_synthetic_example_values(self):
        # All four component flags are required; omitting any of them must
        # fail argument parsing rather than silently using a default value
        # (unlike examples/prioritization_example.py's synthetic dataset).
        with self.assertRaises(SystemExit):
            main(["--degradation-rate", "0.1", "--access-cost", "0.3", "--criticality", "0.7"])

    def test_cli_rejects_nan_input_via_existing_validation(self):
        with self.assertRaisesRegex(ValueError, "NaN"):
            main([
                "--fault-risk", "nan",
                "--degradation-rate", "0.1",
                "--access-cost", "0.3",
                "--criticality", "0.7",
            ])

    def test_cli_rejects_non_numeric_input_via_argparse(self):
        with self.assertRaises(SystemExit):
            main([
                "--fault-risk", "not-a-number",
                "--degradation-rate", "0.1",
                "--access-cost", "0.3",
                "--criticality", "0.7",
            ])


if __name__ == "__main__":
    unittest.main()
