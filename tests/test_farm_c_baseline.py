from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.baseline.train_farm_c_baseline import (
    flatten_valid_timesteps,
    prediction_frame,
    select_threshold_and_evaluate,
    validate_and_filter_metadata,
)


class FarmCBaselineTests(unittest.TestCase):
    def test_flattening_clips_and_preserves_sequence_timestep_order(self):
        X = np.asarray(
            [
                [[-20.0, 1.0], [2.0, 30.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]],
            ]
        )
        y = np.asarray([[0, 1, 0], [1, 0, 1]], dtype=np.uint8)
        mask = np.asarray([[1, 0, 1], [0, 1, 1]], dtype=np.uint8)

        flat_X, flat_y, selector = flatten_valid_timesteps(X, y, mask)

        np.testing.assert_array_equal(selector, [True, False, True, False, True, True])
        np.testing.assert_allclose(
            flat_X, [[-10.0, 1.0], [3.0, 4.0], [7.0, 8.0], [9.0, 10.0]]
        )
        np.testing.assert_array_equal(flat_y, [0, 0, 0, 1])

    def test_mask_zero_rows_are_excluded_before_sklearn_finiteness_check(self):
        X = np.asarray([[[1.0], [np.nan], [np.inf]]])
        y = np.asarray([[0, 1, 1]], dtype=np.uint8)
        mask = np.asarray([[1, 0, 0]], dtype=np.uint8)

        flat_X, flat_y, _ = flatten_valid_timesteps(X, y, mask)

        np.testing.assert_array_equal(flat_X, [[1.0]])
        np.testing.assert_array_equal(flat_y, [0])

    def test_threshold_is_selected_from_validation_only(self):
        val_y = np.asarray([0, 0, 1, 1], dtype=np.uint8)
        val_probabilities = np.asarray([0.1, 0.4, 0.6, 0.9])
        test_y = np.asarray([0, 1], dtype=np.uint8)

        first, _, _ = select_threshold_and_evaluate(
            val_y, val_probabilities, test_y, np.asarray([0.2, 0.8])
        )
        second, _, _ = select_threshold_and_evaluate(
            val_y, val_probabilities, test_y[::-1], np.asarray([0.99, 0.01])
        )

        self.assertEqual(first, 0.6)
        self.assertEqual(second, first)

    def test_prediction_metadata_uses_exact_flattened_mask_alignment(self):
        y = np.zeros((1, 144), dtype=np.uint8)
        y[0, 5] = 1
        mask = np.zeros_like(y)
        mask[0, [0, 5, 143]] = 1
        metadata = pd.DataFrame(
            {
                "farm": "C",
                "split": "val",
                "sequence_idx": 0,
                "timestep_idx": np.arange(144),
                "asset_id": 7,
                "event_id": 11,
                "window_end": pd.date_range("2024-01-01", periods=144, freq="10min"),
                "fault_time": pd.NaT,
                "hours_to_fault": np.nan,
                "label": y.reshape(-1),
                "mask": mask.reshape(-1),
            }
        )

        filtered = validate_and_filter_metadata(metadata, y, mask, "val")
        predictions = prediction_frame(
            filtered,
            np.asarray([0.1, 0.8, 0.2]),
            threshold=0.5,
            model_name="test_model",
        )

        np.testing.assert_array_equal(predictions["timestep_idx"], [0, 5, 143])
        np.testing.assert_array_equal(predictions["label"], [0, 1, 0])
        np.testing.assert_allclose(predictions["probability"], [0.1, 0.8, 0.2])
        np.testing.assert_array_equal(predictions["prediction"], [0, 1, 0])
        self.assertEqual(predictions["sequence_idx"].tolist(), [0, 0, 0])


if __name__ == "__main__":
    unittest.main()
