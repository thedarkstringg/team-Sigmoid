from __future__ import annotations

import unittest

import numpy as np

from src.prioritization import (
    CriticalityCalculator,
    CriticalityConfig,
    PrioritizationPipeline,
    calculate_criticality,
)


class CriticalityCalculationTests(unittest.TestCase):
    def test_capacity_only_calculation(self):
        # factor = 2.5 / 1.0 = 2.5; weight_capacity default 1.0 -> 2.5
        self.assertAlmostEqual(calculate_criticality(capacity=2.5), 2.5)

    def test_output_only_calculation(self):
        # factor = 0.4 / 1.0 = 0.4; weight_output default 1.0 -> 0.4
        self.assertAlmostEqual(calculate_criticality(output=0.4), 0.4)

    def test_combined_capacity_and_output_calculation(self):
        # capacity_term = 1.0*2.0 = 2.0; output_term = 1.0*0.5 = 0.5 -> 2.5
        self.assertAlmostEqual(calculate_criticality(capacity=2.0, output=0.5), 2.5)

    def test_custom_reference_capacity(self):
        config = CriticalityConfig(reference_capacity=5.0)
        # factor = 2.5 / 5.0 = 0.5
        self.assertAlmostEqual(calculate_criticality(capacity=2.5, config=config), 0.5)

    def test_custom_reference_output(self):
        config = CriticalityConfig(reference_output=2.0)
        # factor = 1.0 / 2.0 = 0.5
        self.assertAlmostEqual(calculate_criticality(output=1.0, config=config), 0.5)

    def test_custom_combination_weights(self):
        config = CriticalityConfig(weight_capacity=3.0, weight_output=0.5)
        # capacity_term = 3.0*2.0 = 6.0; output_term = 0.5*4.0 = 2.0 -> 8.0
        self.assertAlmostEqual(calculate_criticality(capacity=2.0, output=4.0, config=config), 8.0)

    def test_higher_capacity_yields_higher_criticality(self):
        low = calculate_criticality(capacity=1.0)
        high = calculate_criticality(capacity=5.0)
        self.assertGreater(high, low)

    def test_higher_output_yields_higher_criticality(self):
        low = calculate_criticality(output=0.2)
        high = calculate_criticality(output=0.9)
        self.assertGreater(high, low)

    def test_deterministic_repeated_calculation(self):
        first = calculate_criticality(capacity=2.0, output=0.5)
        second = calculate_criticality(capacity=2.0, output=0.5)
        self.assertEqual(first, second)

    def test_scalar_input_returns_float(self):
        result = calculate_criticality(capacity=3.0)
        self.assertIsInstance(result, float)

    def test_vectorized_capacity_only(self):
        result = calculate_criticality(capacity=[1.0, 2.0, 3.0])
        self.assertIsInstance(result, np.ndarray)
        np.testing.assert_allclose(result, [1.0, 2.0, 3.0])

    def test_vectorized_combined(self):
        result = calculate_criticality(capacity=[1.0, 2.0], output=[0.1, 0.2])
        np.testing.assert_allclose(result, [1.1, 2.2])

    def test_calculator_class_reuses_configured_config(self):
        config = CriticalityConfig(weight_capacity=2.0)
        calculator = CriticalityCalculator(config=config)
        self.assertAlmostEqual(calculator.compute(capacity=2.0), 4.0)

    def test_zero_capacity_boundary(self):
        self.assertAlmostEqual(calculate_criticality(capacity=0.0), 0.0)

    def test_zero_output_boundary(self):
        self.assertAlmostEqual(calculate_criticality(output=0.0), 0.0)

    def test_does_not_mutate_input_arrays(self):
        capacity = np.array([1.0, 2.0, 3.0])
        original = capacity.copy()

        calculate_criticality(capacity=capacity)

        np.testing.assert_array_equal(capacity, original)

    def test_pipeline_accepts_computed_criticality(self):
        criticality = calculate_criticality(capacity=2.0, output=0.5)
        pipeline = PrioritizationPipeline()
        score = pipeline.calculate(fault_risk=0.5, degradation_rate=0.1, access_cost=0.2, criticality=criticality)
        self.assertAlmostEqual(score, 0.5 + 0.1 - 0.2 + criticality)


class CriticalityValidationTests(unittest.TestCase):
    def test_rejects_missing_capacity_and_output(self):
        with self.assertRaises(ValueError):
            calculate_criticality()

    def test_rejects_negative_capacity(self):
        with self.assertRaises(ValueError):
            calculate_criticality(capacity=-1.0)

    def test_rejects_negative_output(self):
        with self.assertRaises(ValueError):
            calculate_criticality(output=-1.0)

    def test_rejects_nan(self):
        with self.assertRaises(ValueError):
            calculate_criticality(capacity=np.nan)

    def test_rejects_infinity(self):
        with self.assertRaises(ValueError):
            calculate_criticality(output=np.inf)

    def test_rejects_invalid_reference_values(self):
        with self.assertRaises(ValueError):
            CriticalityConfig(reference_capacity=0.0).validate()
        with self.assertRaises(ValueError):
            CriticalityConfig(reference_output=-1.0).validate()

    def test_rejects_invalid_weights(self):
        with self.assertRaises(ValueError):
            CriticalityConfig(weight_capacity=-1.0).validate()
        with self.assertRaises(ValueError):
            CriticalityConfig(weight_output=-2.0).validate()

    def test_rejects_both_weights_zero(self):
        with self.assertRaises(ValueError):
            CriticalityConfig(weight_capacity=0.0, weight_output=0.0).validate()

    def test_rejects_incompatible_array_shapes(self):
        with self.assertRaises(ValueError):
            calculate_criticality(capacity=[1.0, 2.0, 3.0], output=[1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
