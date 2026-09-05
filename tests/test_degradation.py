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


if __name__ == "__main__":
    unittest.main()