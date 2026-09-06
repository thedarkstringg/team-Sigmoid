from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.eval.cross_farm_evaluate import check_compatibility
from src.model.model import TemporalRiskModel


def _write_checkpoint(directory: Path, input_size: int, hidden_size: int = 32, num_layers: int = 2) -> Path:
    model = TemporalRiskModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
    path = directory / "checkpoint.pt"
    torch.save(
        {"model_state_dict": model.state_dict(), "epoch": 5, "best_val_loss": 0.42},
        path,
    )
    return path


def _write_sequences(directory: Path, input_size: int, n_sequences: int = 2) -> None:
    rng = np.random.default_rng(seed=42)
    x = rng.random((n_sequences, 144, input_size), dtype=np.float32)
    y = rng.integers(0, 2, size=(n_sequences, 144), dtype=np.uint8)
    mask = np.ones((n_sequences, 144), dtype=np.uint8)
    np.save(directory / "val_X.npy", x)
    np.save(directory / "val_y.npy", y)
    np.save(directory / "val_mask.npy", mask)


class CrossFarmCompatibilityTests(unittest.TestCase):
    def test_compatible_ten_feature_checkpoint_and_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = _write_checkpoint(root, input_size=10)
            _write_sequences(root, input_size=10)

            ckpt_size, data_size, compatible = check_compatibility(checkpoint, root)

            self.assertEqual(ckpt_size, 10)
            self.assertEqual(data_size, 10)
            self.assertTrue(compatible)

    def test_compatible_eleven_feature_checkpoint_and_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = _write_checkpoint(root, input_size=11)
            _write_sequences(root, input_size=11)

            ckpt_size, data_size, compatible = check_compatibility(checkpoint, root)

            self.assertEqual(ckpt_size, 11)
            self.assertEqual(data_size, 11)
            self.assertTrue(compatible)

    def test_incompatible_feature_dimensions_raise_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = _write_checkpoint(root, input_size=10)
            _write_sequences(root, input_size=11)

            with self.assertRaisesRegex(ValueError, "input_size=10.*input_size=11"):
                check_compatibility(checkpoint, root)

    def test_missing_checkpoint_raises_file_not_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_sequences(root, input_size=10)

            with self.assertRaises(FileNotFoundError):
                check_compatibility(root / "nonexistent.pt", root)

    def test_missing_sequence_files_raise_file_not_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = _write_checkpoint(root, input_size=10)

            with self.assertRaisesRegex(FileNotFoundError, "val_X.npy"):
                check_compatibility(checkpoint, root)

    def test_deterministic_repeated_check_returns_identical_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = _write_checkpoint(root, input_size=10)
            _write_sequences(root, input_size=10)

            first = check_compatibility(checkpoint, root)
            second = check_compatibility(checkpoint, root)

            self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
