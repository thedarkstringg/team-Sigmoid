"""
Ranking-level tests for the existing prioritization system.

These tests exercise the REAL public APIs from src/prioritization/ (no mocks)
to verify that multiple turbines rank correctly by PriorityScore and that
changing weights or inputs moves that ranking in the expected direction.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.prioritization import (
    PrioritizationPipeline,
    PrioritizationWeights,
    calculate_priority,
)


BASE = dict(fault_risk=0.5, degradation_rate=0.1, access_cost=0.3, criticality=0.6)


class SingleFactorRankingTests(unittest.TestCase):
    def test_higher_fault_risk_scores_higher(self):
        low = calculate_priority(**{**BASE, "fault_risk": 0.2})
        high = calculate_priority(**{**BASE, "fault_risk": 0.9})
        self.assertGreater(high, low)

    def test_higher_degradation_rate_scores_higher(self):
        low = calculate_priority(**{**BASE, "degradation_rate": 0.0})
        high = calculate_priority(**{**BASE, "degradation_rate": 0.5})
        self.assertGreater(high, low)

    def test_higher_criticality_scores_higher(self):
        low = calculate_priority(**{**BASE, "criticality": 0.2})
        high = calculate_priority(**{**BASE, "criticality": 0.9})
        self.assertGreater(high, low)

    def test_higher_access_cost_scores_lower(self):
        # w3 multiplies AccessCost with a negative sign in the formula.
        low_cost = calculate_priority(**{**BASE, "access_cost": 0.1})
        high_cost = calculate_priority(**{**BASE, "access_cost": 0.9})
        self.assertLess(high_cost, low_cost)


class MultiTurbineRankingTests(unittest.TestCase):
    def test_expected_descending_ranking_order(self):
        # Hand-computed with default weights (all 1.0):
        #   T-A: 0.9 + 0.3 - 0.1 + 0.8 = 1.9
        #   T-B: 0.8 + 0.1 - 0.5 + 0.3 = 0.7
        #   T-C: 0.3 + 0.05 - 0.2 + 0.4 = 0.55
        fault_risk = np.array([0.9, 0.8, 0.3])
        degradation_rate = np.array([0.3, 0.1, 0.05])
        access_cost = np.array([0.1, 0.5, 0.2])
        criticality = np.array([0.8, 0.3, 0.4])
        turbines = ["T-A", "T-B", "T-C"]

        scores = calculate_priority(fault_risk, degradation_rate, access_cost, criticality)
        ranked_order = [turbines[i] for i in np.argsort(-scores, kind="stable")]

        self.assertEqual(ranked_order, ["T-A", "T-B", "T-C"])

    def test_increasing_one_weight_changes_ranking_in_expected_direction(self):
        # Same technical risk; T-accessible is cheaper to access.
        fault_risk = np.array([0.5, 0.5])
        degradation_rate = np.array([0.0, 0.0])
        access_cost = np.array([0.1, 0.9])   # turbine 0 is much cheaper to reach
        criticality = np.array([0.5, 0.5])
        turbines = ["T-accessible", "T-hard-to-reach"]

        default_scores = calculate_priority(fault_risk, degradation_rate, access_cost, criticality)
        default_order = [turbines[i] for i in np.argsort(-default_scores, kind="stable")]
        self.assertEqual(default_order, ["T-accessible", "T-hard-to-reach"])

        # When AccessCost barely matters, the two turbines tie; ranking ties
        # resolve by stable-sort original order. Raising w1 (FaultRisk) then
        # breaks the tie in favor of the slightly riskier turbine.
        cost_insensitive = PrioritizationWeights(
            w1_fault_risk=1.0, w2_degradation_rate=1.0, w3_access_cost=0.0, w4_criticality=1.0
        )
        tie_scores = calculate_priority(
            fault_risk, degradation_rate, access_cost, criticality, weights=cost_insensitive
        )
        np.testing.assert_allclose(tie_scores, [1.0, 1.0])
        tie_order = [turbines[i] for i in np.argsort(-tie_scores, kind="stable")]
        self.assertEqual(tie_order, ["T-accessible", "T-hard-to-reach"])

        risk_weighted = PrioritizationWeights(
            w1_fault_risk=3.0, w2_degradation_rate=1.0, w3_access_cost=1.0, w4_criticality=1.0
        )
        risk_scores = calculate_priority(
            fault_risk, degradation_rate, access_cost, criticality, weights=risk_weighted
        )
        self.assertAlmostEqual(risk_scores[0], 1.9)
        self.assertAlmostEqual(risk_scores[1], 1.1)
        risk_order = [turbines[i] for i in np.argsort(-risk_scores, kind="stable")]
        self.assertEqual(risk_order, ["T-accessible", "T-hard-to-reach"])

    def test_equal_inputs_produce_equal_scores_and_tie_ranking(self):
        fault_risk = np.array([0.5, 0.5])
        degradation_rate = np.array([0.1, 0.1])
        access_cost = np.array([0.3, 0.3])
        criticality = np.array([0.6, 0.6])

        scores = calculate_priority(fault_risk, degradation_rate, access_cost, criticality)
        np.testing.assert_array_equal(scores[0], scores[1])

        order = np.argsort(-scores, kind="stable")
        self.assertEqual(order.tolist(), [0, 1])


class RankingDeterminismAndImmutabilityTests(unittest.TestCase):
    def _inputs(self):
        return (
            np.array([0.9, 0.2, 0.5]),
            np.array([0.1, 0.4, 0.05]),
            np.array([0.3, 0.1, 0.6]),
            np.array([0.7, 0.8, 0.4]),
        )

    def test_repeated_ranking_is_identical(self):
        fr, dr, ac, cr = self._inputs()
        first_scores = calculate_priority(fr, dr, ac, cr)
        second_scores = calculate_priority(fr, dr, ac, cr)

        np.testing.assert_array_equal(first_scores, second_scores)
        first_order = np.argsort(-first_scores, kind="stable")
        second_order = np.argsort(-second_scores, kind="stable")
        np.testing.assert_array_equal(first_order, second_order)

    def test_ranking_does_not_mutate_inputs(self):
        fr, dr, ac, cr = self._inputs()
        originals = [arr.copy() for arr in (fr, dr, ac, cr)]

        calculate_priority(fr, dr, ac, cr)

        for arr, original in zip((fr, dr, ac, cr), originals):
            np.testing.assert_array_equal(arr, original)


class PipelineEndToEndRankingTests(unittest.TestCase):
    def test_pipeline_ranks_two_turbines_by_predictions(self):
        pipeline = PrioritizationPipeline(horizon_hours=0.5, timestep_hours=1 / 6)

        low_risk_predictions = [0.1] * 6
        high_risk_predictions = [0.8] * 6
        access_cost, criticality = 0.2, 0.5

        low_score = pipeline.calculate_from_predictions(low_risk_predictions, access_cost, criticality)
        high_score = pipeline.calculate_from_predictions(high_risk_predictions, access_cost, criticality)

        self.assertGreater(high_score, low_score)
        self.assertAlmostEqual(high_score - low_score, 0.7)  # constant signals -> zero degradation


if __name__ == "__main__":
    unittest.main()
