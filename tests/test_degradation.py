from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.prioritization import calculate_degradation_rate


class DegradationRateTests(unittest.TestCase):
    def test_fits_linear_trend_per_hour_with_explicit_timestamps(self):
        timestamps = pd.date_range("2024-01-01", periods=4, freq="2h")
        probabilities = [0.1, 0.2, 0.3, 0.4]

        self.assertAlmostEqual(
            calculate_degradation_rate(probabilities, timestamps),
            0.05,
        )

    def test_uses_ten_minute_gru_cadence_by_default(self):
        probabilities = [0.1, 0.2, 0.3]

        self.assertAlmostEqual(calculate_degradation_rate(probabilities), 0.6)

    def test_supports_numeric_time_points(self):
        probabilities = np.array([0.2, 0.4, 0.8])
        time_hours = np.array([3.0, 5.0, 8.0])

        self.assertAlmostEqual(
            calculate_degradation_rate(probabilities, time_hours),
            0.12105263157894737,
        )

    def test_rejects_insufficient_or_invalid_probabilities(self):
        invalid_cases = ([0.2], [], [0.2, np.nan], [0.2, np.inf], [0.2, 1.1])
        for probabilities in invalid_cases:
            with self.subTest(probabilities=probabilities):
                with self.assertRaises(ValueError):
                    calculate_degradation_rate(probabilities)

    def test_rejects_invalid_time_inputs(self):
        probabilities = [0.2, 0.4, 0.6]
        invalid_cases = (
            ([0.0, 1.0], {}),
            ([0.0, 1.0, 1.0], {}),
            ([0.0, np.nan, 2.0], {}),
            (None, {"timestep_hours": 0.0}),
        )
        for time_points, kwargs in invalid_cases:
            with self.subTest(time_points=time_points, kwargs=kwargs):
                with self.assertRaises(ValueError):
                    calculate_degradation_rate(probabilities, time_points, **kwargs)

    def test_constant_probability_yields_zero_slope(self):
        self.assertAlmostEqual(calculate_degradation_rate([0.3, 0.3, 0.3, 0.3]), 0.0)

    def test_decreasing_probability_yields_negative_slope(self):
        self.assertAlmostEqual(calculate_degradation_rate([0.4, 0.3, 0.2, 0.1]), -0.6)

    def test_default_cadence_matches_explicit_ten_minute_timestamps(self):
        probabilities = [0.05, 0.15, 0.25, 0.35]
        timestamps = pd.date_range("2024-03-01", periods=4, freq="10min")

        default_rate = calculate_degradation_rate(probabilities)
        explicit_rate = calculate_degradation_rate(probabilities, timestamps)

        self.assertAlmostEqual(default_rate, explicit_rate)
        self.assertAlmostEqual(default_rate, 0.6)

    def test_slope_scales_inversely_with_timestep_hours(self):
        probabilities = [0.1, 0.2, 0.3]

        default_rate = calculate_degradation_rate(probabilities)
        hourly_rate = calculate_degradation_rate(probabilities, timestep_hours=1.0)

        self.assertAlmostEqual(hourly_rate * 6.0, default_rate)

    def test_does_not_mutate_input_arrays(self):
        probabilities = np.array([0.1, 0.2, 0.15, 0.4])
        time_hours = np.array([0.0, 1.0, 2.0, 3.0])
        probabilities_copy = probabilities.copy()
        time_hours_copy = time_hours.copy()

        calculate_degradation_rate(probabilities, time_hours)

        np.testing.assert_array_equal(probabilities, probabilities_copy)
        np.testing.assert_array_equal(time_hours, time_hours_copy)

    def test_accepts_pandas_series_probabilities_with_custom_index(self):
        probabilities = pd.Series([0.1, 0.2, 0.3], index=["a", "b", "c"])
        timestamps = pd.Series(pd.date_range("2024-01-01", periods=3, freq="2h"))

        self.assertAlmostEqual(calculate_degradation_rate(probabilities, timestamps), 0.05)

    def test_boundary_probabilities_zero_and_one(self):
        self.assertAlmostEqual(calculate_degradation_rate([0.0, 1.0]), 6.0)

    def test_accepts_iso_datetime_strings_as_time_points(self):
        probabilities = [0.1, 0.2, 0.3]
        iso_strings = ["2024-01-01T00:00:00", "2024-01-01T02:00:00", "2024-01-01T04:00:00"]

        self.assertAlmostEqual(calculate_degradation_rate(probabilities, iso_strings), 0.05)

    def test_rejects_non_numeric_non_datetime_time_points(self):
        with self.assertRaises(ValueError):
            calculate_degradation_rate([0.1, 0.2, 0.3], ["a", "b", "c"])


if __name__ == "__main__":
    unittest.main()