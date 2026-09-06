from __future__ import annotations

import unittest

import numpy as np

from src.prioritization import (
    AccessCostCalculator,
    AccessCostConfig,
    PrioritizationPipeline,
    calculate_access_cost,
)


class AccessCostCalculationTests(unittest.TestCase):
    def test_onshore_default_calculation(self):
        # raw = 1.0 + 0.0 + 0.0 = 1.0; onshore premium is 1.0 -> 1.0
        self.assertAlmostEqual(calculate_access_cost("onshore"), 1.0)

    def test_offshore_default_calculation(self):
        # raw = 1.0; offshore premium 1.5 -> 1.5
        self.assertAlmostEqual(calculate_access_cost("offshore"), 1.5)

    def test_case_insensitive_and_whitespace_tolerant_farm_type(self):
        self.assertAlmostEqual(calculate_access_cost(" OFFSHORE "), 1.5)

    def test_offshore_exceeds_otherwise_equivalent_onshore_cost(self):
        onshore_cost = calculate_access_cost("onshore")
        offshore_cost = calculate_access_cost("offshore")
        self.assertGreater(offshore_cost, onshore_cost)

    def test_custom_configuration(self):
        config = AccessCostConfig(
            base_access_cost=2.0,
            transport_cost=1.0,
            logistics_cost=0.5,
            offshore_premium=2.0,
            multiplier=3.0,
        )
        # raw = 3.5
        self.assertAlmostEqual(calculate_access_cost("onshore", config=config), 10.5)
        self.assertAlmostEqual(calculate_access_cost("offshore", config=config), 21.0)

    def test_access_cost_calculator_reuses_configured_config(self):
        config = AccessCostConfig(base_access_cost=2.0, offshore_premium=2.0)
        calculator = AccessCostCalculator(config=config)
        self.assertAlmostEqual(calculator.compute("onshore"), 2.0)
        self.assertAlmostEqual(calculator.compute("offshore"), 4.0)

    def test_vectorized_farm_types(self):
        result = calculate_access_cost(["onshore", "offshore", "onshore"])
        self.assertIsInstance(result, np.ndarray)
        np.testing.assert_allclose(result, [1.0, 1.5, 1.0])

    def test_deterministic_repeated_calculation(self):
        first = calculate_access_cost("offshore")
        second = calculate_access_cost("offshore")
        self.assertEqual(first, second)

    def test_boundary_zero_costs(self):
        config = AccessCostConfig(base_access_cost=0.0, transport_cost=0.0, logistics_cost=0.0)
        self.assertAlmostEqual(calculate_access_cost("onshore", config=config), 0.0)
        self.assertAlmostEqual(calculate_access_cost("offshore", config=config), 0.0)

    def test_does_not_mutate_input_farm_type_array(self):
        farm_types = np.array(["onshore", "offshore"])
        original = farm_types.copy()

        calculate_access_cost(farm_types)

        np.testing.assert_array_equal(farm_types, original)

    def test_pipeline_accepts_computed_access_cost_without_duplicating_logic(self):
        access_cost = calculate_access_cost("offshore")
        pipeline = PrioritizationPipeline()
        score = pipeline.calculate(fault_risk=0.5, degradation_rate=0.1, access_cost=access_cost, criticality=0.5)
        self.assertAlmostEqual(score, 0.5 + 0.1 - access_cost + 0.5)


class AccessCostValidationTests(unittest.TestCase):
    def test_rejects_invalid_farm_type(self):
        with self.assertRaises(ValueError):
            calculate_access_cost("underwater")

    def test_rejects_invalid_farm_type_in_vector(self):
        with self.assertRaises(ValueError):
            calculate_access_cost(["onshore", "underwater"])

    def test_rejects_negative_base_cost(self):
        with self.assertRaises(ValueError):
            AccessCostConfig(base_access_cost=-1.0).validate()
        with self.assertRaises(ValueError):
            calculate_access_cost("onshore", config=AccessCostConfig(base_access_cost=-1.0))

    def test_rejects_negative_transport_or_logistics_cost(self):
        with self.assertRaises(ValueError):
            AccessCostConfig(transport_cost=-0.5).validate()
        with self.assertRaises(ValueError):
            AccessCostConfig(logistics_cost=-0.5).validate()

    def test_rejects_invalid_multiplier(self):
        with self.assertRaises(ValueError):
            AccessCostConfig(multiplier=0.0).validate()
        with self.assertRaises(ValueError):
            AccessCostConfig(multiplier=-2.0).validate()
        with self.assertRaises(ValueError):
            AccessCostConfig(offshore_premium=0.0).validate()

    def test_rejects_nan_and_infinite_components(self):
        with self.assertRaises(ValueError):
            AccessCostConfig(base_access_cost=np.nan).validate()
        with self.assertRaises(ValueError):
            AccessCostConfig(multiplier=np.inf).validate()

    def test_rejects_empty_or_missing_farm_type(self):
        with self.assertRaises(ValueError):
            calculate_access_cost(None)
        with self.assertRaises(ValueError):
            calculate_access_cost([])


if __name__ == "__main__":
    unittest.main()
