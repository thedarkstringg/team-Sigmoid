"""
Integration tests proving the real aggregation, degradation, core scoring, and
pipeline orchestration modules work together end-to-end. These tests do not
mock any prioritization module; they exercise the actual production code.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.prioritization import (
    PrioritizationPipeline,
    PrioritizationWeights,
    aggregate_predictions,
    calculate_degradation_rate,
    calculate_priority,
)


HORIZON_HOURS = 0.5
TIMESTEP_HOURS = 1 / 6  # 10-minute cadence, matches the project's GRU sequences


class EndToEndPredictionToPriorityTests(unittest.TestCase):
    def test_end_to_end_flow_matches_manually_verified_aggregation(self):
        probabilities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        # Hand-verified trailing mean, window=3 samples (0.5h / 10min):
        # [0.1, .15, .2, .3, .4, .5]
        expected_aggregated = np.array([0.1, 0.15, 0.2, 0.3, 0.4, 0.5])
        np.testing.assert_allclose(
            aggregate_predictions(probabilities, horizon_hours=HORIZON_HOURS, timestep_hours=TIMESTEP_HOURS),
            expected_aggregated,
        )

        weights = PrioritizationWeights(
            w1_fault_risk=1.0, w2_degradation_rate=1.0, w3_access_cost=1.0, w4_criticality=1.0
        )
        expected_fault_risk = expected_aggregated[-1]
        expected_degradation = calculate_degradation_rate(expected_aggregated, timestep_hours=TIMESTEP_HOURS)
        access_cost, criticality = 0.2, 0.7
        expected_score = calculate_priority(
            expected_fault_risk, expected_degradation, access_cost, criticality, weights=weights
        )

        pipeline = PrioritizationPipeline(weights=weights, horizon_hours=HORIZON_HOURS, timestep_hours=TIMESTEP_HOURS)
        actual_score = pipeline.calculate_from_predictions(probabilities, access_cost=access_cost, criticality=criticality)

        self.assertAlmostEqual(actual_score, expected_score, places=9)


class AggregationToFaultRiskTests(unittest.TestCase):
    def test_pipeline_uses_smoothed_signal_not_the_raw_final_probability(self):
        probabilities = [0.0, 0.0, 0.0, 0.0, 0.9]
        pipeline = PrioritizationPipeline(horizon_hours=HORIZON_HOURS, timestep_hours=TIMESTEP_HOURS)

        fault_risk = pipeline.fault_risk_from_predictions(probabilities)
        raw_last_probability = probabilities[-1]

        # window=3 -> mean([0.0, 0.0, 0.9]) = 0.3, not the raw spike of 0.9.
        self.assertAlmostEqual(fault_risk, 0.3)
        self.assertNotAlmostEqual(fault_risk, raw_last_probability)

        degradation_rate = pipeline.degradation_rate_from_predictions(probabilities)
        access_cost, criticality = 0.0, 0.0
        actual_score = pipeline.calculate_from_predictions(probabilities, access_cost, criticality)
        expected_score = pipeline.calculate(fault_risk, degradation_rate, access_cost, criticality)
        self.assertAlmostEqual(actual_score, expected_score)


class AggregationToDegradationTests(unittest.TestCase):
    def test_positive_trend_produces_positive_degradation_through_the_real_signal(self):
        probabilities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        pipeline = PrioritizationPipeline(horizon_hours=HORIZON_HOURS, timestep_hours=TIMESTEP_HOURS)

        aggregated_signal = aggregate_predictions(
            probabilities, horizon_hours=HORIZON_HOURS, timestep_hours=TIMESTEP_HOURS
        )
        expected_degradation = calculate_degradation_rate(aggregated_signal, timestep_hours=TIMESTEP_HOURS)
        actual_degradation = pipeline.degradation_rate_from_predictions(probabilities)

        self.assertGreater(actual_degradation, 0.0)
        self.assertAlmostEqual(actual_degradation, expected_degradation)

        access_cost, criticality = 0.1, 0.2
        score = pipeline.calculate_from_predictions(probabilities, access_cost, criticality)
        score_without_degradation = pipeline.calculate(
            pipeline.fault_risk_from_predictions(probabilities), 0.0, access_cost, criticality
        )
        self.assertGreater(score, score_without_degradation)

    def test_negative_trend_produces_negative_degradation_through_the_real_signal(self):
        probabilities = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        pipeline = PrioritizationPipeline(horizon_hours=HORIZON_HOURS, timestep_hours=TIMESTEP_HOURS)

        actual_degradation = pipeline.degradation_rate_from_predictions(probabilities)
        self.assertLess(actual_degradation, 0.0)

        access_cost, criticality = 0.1, 0.2
        score = pipeline.calculate_from_predictions(probabilities, access_cost, criticality)
        score_without_degradation = pipeline.calculate(
            pipeline.fault_risk_from_predictions(probabilities), 0.0, access_cost, criticality
        )
        self.assertLess(score, score_without_degradation)


class NoFutureLeakageTests(unittest.TestCase):
    def test_appending_a_future_spike_does_not_change_earlier_aggregated_values(self):
        predictions_before = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
        predictions_after = predictions_before + [0.95, 0.97, 0.99]

        aggregated_before = aggregate_predictions(
            predictions_before, horizon_hours=HORIZON_HOURS, timestep_hours=TIMESTEP_HOURS
        )
        aggregated_after = aggregate_predictions(
            predictions_after, horizon_hours=HORIZON_HOURS, timestep_hours=TIMESTEP_HOURS
        )

        np.testing.assert_array_equal(aggregated_before, aggregated_after[: len(predictions_before)])

        # Degradation computed from the identical prefix must also agree,
        # proving the causal property survives into the next module too.
        degradation_before = calculate_degradation_rate(aggregated_before, timestep_hours=TIMESTEP_HOURS)
        degradation_prefix_of_after = calculate_degradation_rate(
            aggregated_after[: len(predictions_before)], timestep_hours=TIMESTEP_HOURS
        )
        self.assertEqual(degradation_before, degradation_prefix_of_after)

        pipeline = PrioritizationPipeline(horizon_hours=HORIZON_HOURS, timestep_hours=TIMESTEP_HOURS)
        fault_risk_before = pipeline.fault_risk_from_predictions(predictions_before)
        self.assertEqual(fault_risk_before, aggregated_after[len(predictions_before) - 1])


class SixVersusTwelveHourAggregationTests(unittest.TestCase):
    def test_both_horizons_run_and_differ_on_a_monotonic_trend(self):
        probabilities = np.linspace(0.0, 0.9, 91)

        pipeline_6h = PrioritizationPipeline(horizon_hours=6.0, timestep_hours=TIMESTEP_HOURS)
        pipeline_12h = PrioritizationPipeline(horizon_hours=12.0, timestep_hours=TIMESTEP_HOURS)

        fault_risk_6h = pipeline_6h.fault_risk_from_predictions(probabilities)
        fault_risk_12h = pipeline_12h.fault_risk_from_predictions(probabilities)

        self.assertAlmostEqual(
            fault_risk_6h,
            float(aggregate_predictions(probabilities, horizon_hours=6.0, timestep_hours=TIMESTEP_HOURS)[-1]),
        )
        self.assertAlmostEqual(
            fault_risk_12h,
            float(aggregate_predictions(probabilities, horizon_hours=12.0, timestep_hours=TIMESTEP_HOURS)[-1]),
        )
        # On a monotonically increasing sequence, the longer 12h window
        # averages in more of the smaller historical values.
        self.assertGreater(fault_risk_6h, fault_risk_12h)


class CustomWeightsEndToEndTests(unittest.TestCase):
    def test_custom_weights_apply_correct_signs_to_every_component(self):
        probabilities = [0.2, 0.3, 0.25, 0.4, 0.5, 0.6]
        weights = PrioritizationWeights(
            w1_fault_risk=2.0, w2_degradation_rate=3.0, w3_access_cost=4.0, w4_criticality=5.0
        )
        pipeline = PrioritizationPipeline(weights=weights, horizon_hours=HORIZON_HOURS, timestep_hours=TIMESTEP_HOURS)

        fault_risk = pipeline.fault_risk_from_predictions(probabilities)
        degradation_rate = pipeline.degradation_rate_from_predictions(probabilities)
        access_cost, criticality = 0.3, 0.6

        expected = (
            2.0 * fault_risk + 3.0 * degradation_rate - 4.0 * access_cost + 5.0 * criticality
        )
        actual = pipeline.calculate_from_predictions(probabilities, access_cost, criticality)
        self.assertAlmostEqual(actual, expected)


class AccessCostEffectTests(unittest.TestCase):
    def test_higher_access_cost_lowers_priority_score(self):
        probabilities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        pipeline = PrioritizationPipeline(horizon_hours=HORIZON_HOURS, timestep_hours=TIMESTEP_HOURS)
        criticality = 0.5

        low_cost_score = pipeline.calculate_from_predictions(probabilities, access_cost=0.1, criticality=criticality)
        high_cost_score = pipeline.calculate_from_predictions(probabilities, access_cost=0.9, criticality=criticality)

        self.assertLess(high_cost_score, low_cost_score)
        self.assertAlmostEqual(low_cost_score - high_cost_score, 0.8)  # default w3 = 1.0


class CriticalityEffectTests(unittest.TestCase):
    def test_higher_criticality_raises_priority_score(self):
        probabilities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        pipeline = PrioritizationPipeline(horizon_hours=HORIZON_HOURS, timestep_hours=TIMESTEP_HOURS)
        access_cost = 0.3

        low_crit_score = pipeline.calculate_from_predictions(probabilities, access_cost=access_cost, criticality=0.2)
        high_crit_score = pipeline.calculate_from_predictions(probabilities, access_cost=access_cost, criticality=0.9)

        self.assertGreater(high_crit_score, low_crit_score)
        self.assertAlmostEqual(high_crit_score - low_crit_score, 0.7)  # default w4 = 1.0


class RiskEffectTests(unittest.TestCase):
    def test_higher_risk_sequence_produces_higher_priority_contribution(self):
        low_risk_sequence = [0.1] * 6
        high_risk_sequence = [0.8] * 6
        pipeline = PrioritizationPipeline(horizon_hours=HORIZON_HOURS, timestep_hours=TIMESTEP_HOURS)
        access_cost, criticality = 0.2, 0.5

        # Constant sequences -> zero degradation both sides, isolating FaultRisk.
        self.assertAlmostEqual(pipeline.degradation_rate_from_predictions(low_risk_sequence), 0.0)
        self.assertAlmostEqual(pipeline.degradation_rate_from_predictions(high_risk_sequence), 0.0)

        low_score = pipeline.calculate_from_predictions(low_risk_sequence, access_cost, criticality)
        high_score = pipeline.calculate_from_predictions(high_risk_sequence, access_cost, criticality)

        self.assertAlmostEqual(high_score - low_score, 0.7)  # default w1 = 1.0


class DegradationEffectTests(unittest.TestCase):
    def test_increasing_versus_decreasing_sequence_changes_degradation_direction(self):
        increasing_sequence = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        decreasing_sequence = [0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        pipeline = PrioritizationPipeline(horizon_hours=HORIZON_HOURS, timestep_hours=TIMESTEP_HOURS)

        degradation_inc = pipeline.degradation_rate_from_predictions(increasing_sequence)
        degradation_dec = pipeline.degradation_rate_from_predictions(decreasing_sequence)
        self.assertGreater(degradation_inc, 0.0)
        self.assertLess(degradation_dec, 0.0)

        fixed_fault_risk, access_cost, criticality = 0.5, 0.2, 0.5
        score_inc = pipeline.calculate(fixed_fault_risk, degradation_inc, access_cost, criticality)
        score_dec = pipeline.calculate(fixed_fault_risk, degradation_dec, access_cost, criticality)

        self.assertGreater(score_inc, score_dec)
        self.assertAlmostEqual(score_inc - score_dec, degradation_inc - degradation_dec)  # default w2 = 1.0


class DeterministicRepeatedExecutionTests(unittest.TestCase):
    def test_identical_inputs_produce_identical_outputs(self):
        probabilities = [0.05, 0.12, 0.2, 0.33, 0.41, 0.5, 0.62]
        pipeline = PrioritizationPipeline(horizon_hours=HORIZON_HOURS, timestep_hours=TIMESTEP_HOURS)

        first_run = pipeline.calculate_from_predictions(probabilities, access_cost=0.3, criticality=0.6)
        second_run = pipeline.calculate_from_predictions(probabilities, access_cost=0.3, criticality=0.6)

        self.assertEqual(first_run, second_run)


class InputImmutabilityTests(unittest.TestCase):
    def test_calculate_from_predictions_does_not_mutate_inputs(self):
        predictions = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5])
        predictions_copy = predictions.copy()
        access_cost, criticality = 0.3, 0.6
        pipeline = PrioritizationPipeline(horizon_hours=HORIZON_HOURS, timestep_hours=TIMESTEP_HOURS)

        pipeline.calculate_from_predictions(predictions, access_cost, criticality)

        np.testing.assert_array_equal(predictions, predictions_copy)


class VectorizedMultiTurbineTests(unittest.TestCase):
    def test_vectorized_scoring_matches_per_turbine_calculate_from_predictions(self):
        turbine_a_predictions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        turbine_b_predictions = [0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        pipeline = PrioritizationPipeline(horizon_hours=HORIZON_HOURS, timestep_hours=TIMESTEP_HOURS)

        fault_risk_vec = np.array(
            [
                pipeline.fault_risk_from_predictions(turbine_a_predictions),
                pipeline.fault_risk_from_predictions(turbine_b_predictions),
            ]
        )
        degradation_vec = np.array(
            [
                pipeline.degradation_rate_from_predictions(turbine_a_predictions),
                pipeline.degradation_rate_from_predictions(turbine_b_predictions),
            ]
        )
        access_cost_vec = np.array([0.2, 0.3])
        criticality_vec = np.array([0.5, 0.7])

        vectorized_scores = pipeline.calculate(fault_risk_vec, degradation_vec, access_cost_vec, criticality_vec)
        self.assertIsInstance(vectorized_scores, np.ndarray)

        score_a = pipeline.calculate_from_predictions(turbine_a_predictions, access_cost=0.2, criticality=0.5)
        score_b = pipeline.calculate_from_predictions(turbine_b_predictions, access_cost=0.3, criticality=0.7)
        np.testing.assert_allclose(vectorized_scores, [score_a, score_b])


class InvalidDataThroughPipelineTests(unittest.TestCase):
    def test_rejects_nan_probabilities(self):
        pipeline = PrioritizationPipeline(horizon_hours=HORIZON_HOURS, timestep_hours=TIMESTEP_HOURS)
        with self.assertRaises(ValueError):
            pipeline.calculate_from_predictions([0.1, np.nan, 0.3], access_cost=0.2, criticality=0.5)

    def test_rejects_infinite_probabilities(self):
        pipeline = PrioritizationPipeline(horizon_hours=HORIZON_HOURS, timestep_hours=TIMESTEP_HOURS)
        with self.assertRaises(ValueError):
            pipeline.calculate_from_predictions([0.1, np.inf, 0.3], access_cost=0.2, criticality=0.5)

    def test_rejects_probabilities_outside_unit_interval(self):
        pipeline = PrioritizationPipeline(horizon_hours=HORIZON_HOURS, timestep_hours=TIMESTEP_HOURS)
        with self.assertRaises(ValueError):
            pipeline.calculate_from_predictions([0.1, 1.5, 0.3], access_cost=0.2, criticality=0.5)

    def test_rejects_invalid_horizon_through_real_pipeline(self):
        pipeline = PrioritizationPipeline(horizon_hours=-1.0, timestep_hours=TIMESTEP_HOURS)
        with self.assertRaises(ValueError):
            pipeline.calculate_from_predictions([0.1, 0.2, 0.3], access_cost=0.2, criticality=0.5)

    def test_rejects_invalid_timestep_through_real_pipeline(self):
        pipeline = PrioritizationPipeline(horizon_hours=HORIZON_HOURS, timestep_hours=0.0)
        with self.assertRaises(ValueError):
            pipeline.calculate_from_predictions([0.1, 0.2, 0.3], access_cost=0.2, criticality=0.5)


class ComponentConsistencyTests(unittest.TestCase):
    def test_manually_composed_flow_matches_pipeline_output(self):
        probabilities = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7]
        weights = PrioritizationWeights(
            w1_fault_risk=1.5, w2_degradation_rate=0.5, w3_access_cost=2.0, w4_criticality=1.2
        )
        access_cost, criticality = 0.25, 0.65

        aggregated_signal = aggregate_predictions(
            probabilities, horizon_hours=HORIZON_HOURS, timestep_hours=TIMESTEP_HOURS
        )
        fault_risk = float(aggregated_signal[-1])
        degradation_rate = calculate_degradation_rate(aggregated_signal, timestep_hours=TIMESTEP_HOURS)
        manual_score = calculate_priority(fault_risk, degradation_rate, access_cost, criticality, weights=weights)

        pipeline = PrioritizationPipeline(weights=weights, horizon_hours=HORIZON_HOURS, timestep_hours=TIMESTEP_HOURS)
        pipeline_score = pipeline.calculate_from_predictions(probabilities, access_cost, criticality)

        self.assertAlmostEqual(manual_score, pipeline_score, places=9)


if __name__ == "__main__":
    unittest.main()
