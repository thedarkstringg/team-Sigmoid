from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.prioritization import aggregate_predictions


class AggregationTests(unittest.TestCase):
    def test_basic_six_hour_trailing_aggregation_matches_independent_rolling_mean(self):
        probabilities = np.linspace(0.02, 1.0, 40)

        result = aggregate_predictions(probabilities, horizon_hours=6.0, timestep_hours=1 / 6)
        expected = pd.Series(probabilities).rolling(window=36, min_periods=1).mean().to_numpy()

        np.testing.assert_allclose(result, expected)

    def test_constant_probabilities_remain_constant(self):
        probabilities = [0.4] * 10

        result = aggregate_predictions(probabilities, horizon_hours=6.0, timestep_hours=1 / 6)

        np.testing.assert_allclose(result, [0.4] * 10)

    def test_step_change_is_smoothed_progressively_without_future_leakage(self):
        probabilities = [0.0] * 5 + [1.0] * 5

        result = aggregate_predictions(probabilities, horizon_hours=0.5, timestep_hours=1 / 6)

        # window = 3 samples; index 5 is the first "1.0" - it must only see
        # itself plus the two preceding zeros, never the later 1.0 values.
        np.testing.assert_allclose(result[:5], [0.0] * 5)
        self.assertAlmostEqual(result[5], 1 / 3)
        self.assertAlmostEqual(result[6], 2 / 3)
        np.testing.assert_allclose(result[7:], [1.0] * 3)

    def test_twelve_hour_horizon_matches_independent_rolling_mean(self):
        probabilities = [0.1, 0.2, 0.3, 0.4, 0.5]

        result = aggregate_predictions(probabilities, horizon_hours=12.0, timestep_hours=1 / 6)
        expected = pd.Series(probabilities).rolling(window=72, min_periods=1).mean().to_numpy()

        np.testing.assert_allclose(result, expected)

    def test_rejects_invalid_probabilities(self):
        invalid_cases = ([0.2, np.nan], [0.2, np.inf], [-0.1, 0.2], [0.2, 1.1], "not a sequence")
        for probabilities in invalid_cases:
            with self.subTest(probabilities=probabilities):
                with self.assertRaises(ValueError):
                    aggregate_predictions(probabilities)

    def test_rejects_empty_input(self):
        with self.assertRaises(ValueError):
            aggregate_predictions([])

    def test_rejects_multidimensional_input(self):
        with self.assertRaises(ValueError):
            aggregate_predictions(np.zeros((3, 2)))

    def test_rejects_nan_and_inf(self):
        with self.assertRaises(ValueError):
            aggregate_predictions([0.1, np.nan, 0.3])
        with self.assertRaises(ValueError):
            aggregate_predictions([0.1, np.inf, 0.3])

    def test_rejects_values_outside_unit_interval(self):
        with self.assertRaises(ValueError):
            aggregate_predictions([0.1, -0.01, 0.3])
        with self.assertRaises(ValueError):
            aggregate_predictions([0.1, 1.01, 0.3])

    def test_rejects_invalid_horizon_and_timestep(self):
        invalid_kwargs = (
            {"horizon_hours": 0.0},
            {"horizon_hours": -1.0},
            {"horizon_hours": np.nan},
            {"horizon_hours": np.inf},
            {"timestep_hours": 0.0},
            {"timestep_hours": -0.5},
            {"timestep_hours": np.nan},
        )
        for kwargs in invalid_kwargs:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    aggregate_predictions([0.1, 0.2, 0.3], **kwargs)

    def test_does_not_mutate_input_array(self):
        probabilities = np.array([0.1, 0.5, 0.2, 0.9, 0.3])
        original = probabilities.copy()

        aggregate_predictions(probabilities, horizon_hours=1.0, timestep_hours=1 / 6)

        np.testing.assert_array_equal(probabilities, original)

    def test_output_length_matches_input_length(self):
        for n in (1, 2, 5, 50):
            probabilities = np.linspace(0.0, 1.0, n)
            for horizon_hours, timestep_hours in ((6.0, 1 / 6), (12.0, 1 / 6), (1.0, 1.0)):
                with self.subTest(n=n, horizon_hours=horizon_hours, timestep_hours=timestep_hours):
                    result = aggregate_predictions(
                        probabilities, horizon_hours=horizon_hours, timestep_hours=timestep_hours
                    )
                    self.assertEqual(len(result), n)

    def test_beginning_of_sequence_expands_instead_of_zero_padding(self):
        probabilities = [0.2, 0.4, 0.6, 0.8, 1.0]

        # window = 5 samples (30 min horizon / 6 min timestep), longer than the
        # available history at the start, so early outputs average only what
        # has been observed so far rather than assuming missing zeros.
        result = aggregate_predictions(probabilities, horizon_hours=0.5, timestep_hours=0.1)

        self.assertAlmostEqual(result[0], 0.2)
        self.assertAlmostEqual(result[1], (0.2 + 0.4) / 2)
        self.assertAlmostEqual(result[2], (0.2 + 0.4 + 0.6) / 3)

    def test_horizon_below_one_sample_is_a_no_op(self):
        probabilities = [0.1, 0.9, 0.2, 0.7]

        result = aggregate_predictions(probabilities, horizon_hours=0.01, timestep_hours=1.0)

        np.testing.assert_allclose(result, probabilities)


if __name__ == "__main__":
    unittest.main()
