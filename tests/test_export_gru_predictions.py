from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.eval.export_gru_predictions import infer_model_shape, validate_metadata
from src.model.model import TemporalRiskModel


class PredictionExportContractTests(unittest.TestCase):
    def test_infers_checkpoint_model_dimensions(self):
        model = TemporalRiskModel(input_size=10, hidden_size=7, num_layers=3)
        self.assertEqual(infer_model_shape(model.state_dict()), (10, 7, 3))

    def test_metadata_alignment_is_proved(self):
        y = np.zeros((2, 144), dtype=np.uint8)
        mask = np.ones_like(y)
        metadata = pd.DataFrame(
            {
                "sequence_idx": np.repeat(np.arange(2), 144),
                "timestep_idx": np.tile(np.arange(144), 2),
                "asset_id": 1,
                "event_id": 2,
                "window_end": pd.date_range("2024-01-01", periods=288, freq="10min"),
                "label": y.reshape(-1),
                "mask": mask.reshape(-1),
            }
        )
        validate_metadata(metadata, y, mask)
        metadata.loc[144, "timestep_idx"] = 99
        with self.assertRaisesRegex(ValueError, "timestep_idx"):
            validate_metadata(metadata, y, mask)


if __name__ == "__main__":
    unittest.main()
