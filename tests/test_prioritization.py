from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.prioritization import (
    PrioritizationWeights,
    PriorityCalculator,
    calculate_priority_score,
    validate_prioritization_inputs,
)


class PrioritizationCoreTests(unittest.TestCase):
    def test_scalar_score_calculation_with_default_weights(self):
        # PriorityScore = 1.0 * 0.8 + 1.0 * 0.2 - 1.0 * 0.3 + 1.0 * 1.5 = 2.2
        score = calculate_priority_score(
            fault_risk=0.8,
            degradation_rate=0.2,
            access_cost=0.3,
            criticality=1.5,
        )
        self.assertIsInstance(score, float)
        self.assertAlmostEqual(score, 2.2)

    def test_custom_weights_and_kwargs(self):
        # PriorityScore = 2.0 * 0.8 + 1.5 * 0.2 - 0.5 * 0.3 + 3.0 * 1.5
        # = 1.6 + 0.3 - 0.15 + 4.5 = 6.25
        weights = PrioritizationWeights(
            w1_fault_risk=2.0,
            w2_degradation_rate=1.5,
            w3_access_cost=0.5,
            w4_criticality=3.0,
        )
        score = calculate_priority_score(
            fault_risk=0.8,
            degradation_rate=0.2,
            access_cost=0.3,
            criticality=1.5,
            weights=weights,
        )
        self.assertAlmostEqual(score, 6.25)

        # Test kwarg override w1=1.0 -> 1.0*0.8 + 1.5*0.2 - 0.5*0.3 + 3.0*1.5 = 5.45
        score_kwarg = calculate_priority_score(
            fault_risk=0.8,
            degradation_rate=0.2,
            access_cost=0.3,
            criticality=1.5,
            weights=weights,
            w1=1.0,
        )
        self.assertAlmostEqual(score_kwarg, 5.45)

    def test_numpy_array_and_broadcasting(self):
        fr = np.array([0.9, 0.4])
        dr = np.array([0.1, 0.5])
        ac = 0.2  # Broadcast scalar to array
        cr = np.array([1.0, 2.0])

        # Score 0: 1*0.9 + 1*0.1 - 1*0.2 + 1*1.0 = 1.8
        # Score 1: 1*0.4 + 1*0.5 - 1*0.2 + 1*2.0 = 2.7
        scores = calculate_priority_score(
            fault_risk=fr,
            degradation_rate=dr,
            access_cost=ac,
            criticality=cr,
        )
        self.assertIsInstance(scores, np.ndarray)
        np.testing.assert_allclose(scores, np.array([1.8, 2.7]))

    def test_pandas_series_and_dataframe_integration(self):
        df = pd.DataFrame(
            {
                "asset_id": [1, 2],
                "fault_risk": [0.7, 0.3],
                "degradation_rate": [0.5, 0.1],
                "access_cost": [0.4, 0.1],
                "criticality": [1.0, 1.0],
            }
        )
        calculator = PriorityCalculator()
        result_df = calculator.compute_dataframe(df)

        self.assertIn("priority_score", result_df.columns)
        # Expected:
        # Turbine 1: 0.7 + 0.5 - 0.4 + 1.0 = 1.8
        # Turbine 2: 0.3 + 0.1 - 0.1 + 1.0 = 1.3
        np.testing.assert_allclose(result_df["priority_score"].to_numpy(), [1.8, 1.3])

        # Test single pandas Series return type
        series_res = calculate_priority_score(
            df["fault_risk"], df["degradation_rate"], df["access_cost"], df["criticality"]
        )
        self.assertIsInstance(series_res, pd.Series)
        self.assertEqual(series_res.name, "priority_score")

    def test_validation_missing_inputs(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            calculate_priority_score(None, 0.1, 0.1, 1.0)

    def test_validation_nan_and_inf(self):
        with self.assertRaisesRegex(ValueError, "NaN"):
            calculate_priority_score(np.nan, 0.1, 0.1, 1.0)

        with self.assertRaisesRegex(ValueError, "NaN"):
            calculate_priority_score(np.array([0.5, np.nan]), 0.1, 0.1, 1.0)

        with self.assertRaisesRegex(ValueError, "infinite"):
            calculate_priority_score(0.5, np.inf, 0.1, 1.0)

    def test_validation_invalid_weights(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            PrioritizationWeights(w1_fault_risk=-1.0).validate()

        with self.assertRaisesRegex(ValueError, "NaN"):
            PrioritizationWeights(w2_degradation_rate=np.nan).validate()

        with self.assertRaisesRegex(ValueError, "infinite"):
            PrioritizationWeights(w3_access_cost=np.inf).validate()

    def test_validation_incompatible_shapes(self):
        fr = np.array([0.5, 0.6, 0.7])
        ac = np.array([0.1, 0.2])  # Length 2 vs length 3
        with self.assertRaisesRegex(ValueError, "Incompatible input shapes"):
            calculate_priority_score(fr, 0.1, ac, 1.0)

    def test_determinism_and_immutability(self):
        weights = PrioritizationWeights()
        score1 = calculate_priority_score(0.5, 0.2, 0.1, 1.0, weights=weights)
        score2 = calculate_priority_score(0.5, 0.2, 0.1, 1.0, weights=weights)
        self.assertEqual(score1, score2)

        # Immutability check
        with self.assertRaises(AttributeError):
            weights.w1_fault_risk = 2.0  # dataclass is frozen


if __name__ == "__main__":
    unittest.main()
