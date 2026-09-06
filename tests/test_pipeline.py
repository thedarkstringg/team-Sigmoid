from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from src.prioritization import (
    PrioritizationPipeline,
    PrioritizationWeights,
    aggregate_predictions,
    calculate_degradation_rate,
    calculate_priority,
)


class CalculatePriorityFunctionTests(unittest.TestCase):
    def test_basic_end_to_end_scalar_calculation(self):
        # 1.0*0.5 + 1.0*0.2 - 1.0*0.3 + 1.0*0.8 = 1.2
        score = calculate_priority(
            fault_risk=0.5, degradation_rate=0.2, access_cost=0.3, criticality=0.8
        )
        self.assertAlmostEqual(score, 1.2)

    def test_matches_priority_score_formula_with_custom_weights(self):
        weights = PrioritizationWeights(
            w1_fault_risk=2.0, w2_degradation_rate=3.0, w3_access_cost=0.5, w4_criticality=1.5
        )
        # 2*0.5 + 3*0.2 - 0.5*0.3 + 1.5*0.8 = 1 + 0.6 - 0.15 + 1.2 = 2.65
        score = calculate_priority(
            fault_risk=0.5, degradation_rate=0.2, access_cost=0.3, criticality=0.8, weights=weights
        )
        self.assertAlmostEqual(score, 2.65)

    def test_fault_risk_contributes_positively(self):
        low = calculate_priority(0.2, 0.0, 0.0, 0.0)
        high = calculate_priority(0.9, 0.0, 0.0, 0.0)
        self.assertGreater(high, low)
        self.assertAlmostEqual(high - low, 0.7)

    def test_degradation_rate_contributes_positively(self):
        low = calculate_priority(0.0, 0.1, 0.0, 0.0)
        high = calculate_priority(0.0, 0.6, 0.0, 0.0)
        self.assertGreater(high, low)
        self.assertAlmostEqual(high - low, 0.5)

    def test_access_cost_subtracts_from_score(self):
        cheap = calculate_priority(0.0, 0.0, 0.1, 0.0)
        expensive = calculate_priority(0.0, 0.0, 0.9, 0.0)
        self.assertLess(expensive, cheap)
        self.assertAlmostEqual(cheap - expensive, 0.8)

    def test_criticality_adds_to_score(self):
        low = calculate_priority(0.0, 0.0, 0.0, 0.2)
        high = calculate_priority(0.0, 0.0, 0.0, 0.9)
        self.assertGreater(high, low)
        self.assertAlmostEqual(high - low, 0.7)

    def test_vectorized_inputs_produce_vectorized_output(self):
        fault_risk = np.array([0.1, 0.9])
        degradation_rate = np.array([0.05, -0.2])
        access_cost = np.array([0.2, 0.3])
        criticality = np.array([0.5, 1.0])

        result = calculate_priority(fault_risk, degradation_rate, access_cost, criticality)

        expected = fault_risk + degradation_rate - access_cost + criticality
        self.assertIsInstance(result, np.ndarray)
        np.testing.assert_allclose(result, expected)

    def test_rejects_mismatched_input_lengths(self):
        with self.assertRaises(ValueError):
            calculate_priority(
                fault_risk=np.array([0.1, 0.2, 0.3]),
                degradation_rate=0.1,
                access_cost=np.array([0.1, 0.2]),
                criticality=1.0,
            )

    def test_rejects_nan_and_inf_inputs(self):
        with self.assertRaises(ValueError):
            calculate_priority(np.nan, 0.1, 0.1, 1.0)
        with self.assertRaises(ValueError):
            calculate_priority(0.5, np.inf, 0.1, 1.0)

    def test_rejects_non_numeric_inputs(self):
        with self.assertRaises(ValueError):
            calculate_priority("bad", 0.1, 0.1, 1.0)

    def test_does_not_mutate_input_arrays(self):
        fault_risk = np.array([0.1, 0.2, 0.3])
        degradation_rate = np.array([0.01, 0.02, 0.03])
        access_cost = np.array([0.1, 0.1, 0.1])
        criticality = np.array([1.0, 1.0, 1.0])
        originals = [arr.copy() for arr in (fault_risk, degradation_rate, access_cost, criticality)]

        calculate_priority(fault_risk, degradation_rate, access_cost, criticality)

        for arr, original in zip(
            (fault_risk, degradation_rate, access_cost, criticality), originals
        ):
            np.testing.assert_array_equal(arr, original)

    def test_delegates_scoring_to_core_calculate_priority_score(self):
        with patch("src.prioritization.pipeline.calculate_priority_score") as mock_score:
            mock_score.return_value = 42.0
            result = calculate_priority(0.5, 0.2, 0.3, 0.8, weights=None)

        mock_score.assert_called_once_with(
            fault_risk=0.5, degradation_rate=0.2, access_cost=0.3, criticality=0.8, weights=None
        )
        self.assertEqual(result, 42.0)


class PrioritizationPipelineTests(unittest.TestCase):
    def test_calculate_matches_calculate_priority_function(self):
        pipeline = PrioritizationPipeline()
        direct = calculate_priority(0.4, 0.1, 0.2, 0.6)
        via_pipeline = pipeline.calculate(0.4, 0.1, 0.2, 0.6)
        self.assertAlmostEqual(direct, via_pipeline)

    def test_default_weights_are_preserved(self):
        pipeline = PrioritizationPipeline()
        self.assertEqual(pipeline.weights, PrioritizationWeights())

    def test_custom_weights_are_respected(self):
        weights = PrioritizationWeights(w1_fault_risk=5.0)
        pipeline = PrioritizationPipeline(weights=weights)
        score = pipeline.calculate(1.0, 0.0, 0.0, 0.0)
        self.assertAlmostEqual(score, 5.0)

    def test_fault_risk_from_predictions_uses_existing_aggregation_module(self):
        probabilities = [0.1, 0.2, 0.3, 0.4]
        pipeline = PrioritizationPipeline(horizon_hours=0.5, timestep_hours=1 / 6)

        with patch(
            "src.prioritization.pipeline.aggregate_predictions",
            wraps=aggregate_predictions,
        ) as mock_aggregate:
            fault_risk = pipeline.fault_risk_from_predictions(probabilities)

        mock_aggregate.assert_called_once_with(probabilities, horizon_hours=0.5, timestep_hours=1 / 6)
        expected_signal = aggregate_predictions(probabilities, horizon_hours=0.5, timestep_hours=1 / 6)
        self.assertAlmostEqual(fault_risk, float(expected_signal[-1]))

    def test_degradation_rate_from_predictions_uses_existing_degradation_module(self):
        probabilities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        pipeline = PrioritizationPipeline(horizon_hours=0.5, timestep_hours=1 / 6)

        with patch(
            "src.prioritization.pipeline.calculate_degradation_rate",
            wraps=calculate_degradation_rate,
        ) as mock_degradation:
            rate = pipeline.degradation_rate_from_predictions(probabilities)

        smoothed_signal = aggregate_predictions(probabilities, horizon_hours=0.5, timestep_hours=1 / 6)
        mock_degradation.assert_called_once()
        called_args, called_kwargs = mock_degradation.call_args
        np.testing.assert_allclose(called_args[0], smoothed_signal)
        self.assertEqual(called_kwargs["timestep_hours"], 1 / 6)
        expected_rate = calculate_degradation_rate(smoothed_signal, timestep_hours=1 / 6)
        self.assertAlmostEqual(rate, expected_rate)

    def test_calculate_from_predictions_end_to_end(self):
        probabilities = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7]
        pipeline = PrioritizationPipeline(horizon_hours=0.5, timestep_hours=1 / 6)

        result = pipeline.calculate_from_predictions(probabilities, access_cost=0.2, criticality=0.5)

        expected_signal = aggregate_predictions(probabilities, horizon_hours=0.5, timestep_hours=1 / 6)
        expected_fault_risk = float(expected_signal[-1])
        expected_degradation_rate = calculate_degradation_rate(expected_signal, timestep_hours=1 / 6)
        expected_score = calculate_priority(
            expected_fault_risk, expected_degradation_rate, 0.2, 0.5
        )
        self.assertAlmostEqual(result, expected_score)


if __name__ == "__main__":
    unittest.main()
