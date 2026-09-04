from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.data.build_physical_manifest import build_manifest
from src.data.export_physical_sequences import prepare_power_curve
from src.data.physical_features import (
    compute_physical_features,
    grid_power_factor,
    load_mapping,
    phase_imbalance,
    required_sensor_ids,
    wrap_to_180,
)
from src.data.sequence_utils import (
    SEQ_LEN,
    apply_scaler,
    build_timestep_metadata,
    fit_train_scaler,
    load_scaler,
    save_scaler,
)


CONFIG_PATH = Path("configs/physical_sensor_mapping.yaml")


class PhysicalFormulaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_mapping(CONFIG_PATH)

    def test_yaw_wraps_to_half_open_interval(self):
        values = np.asarray([-540, -181, -180, 179, 180, 181, 540], dtype=float)
        actual = wrap_to_180(values)
        expected = np.asarray([-180, 179, -180, 179, -180, -179, -180], dtype=float)
        np.testing.assert_array_equal(actual, expected)
        self.assertTrue(np.all(actual >= -180))
        self.assertTrue(np.all(actual < 180))

    def test_current_imbalance_formula(self):
        values = pd.DataFrame([[9.0, 10.0, 11.0], [4.0, 4.0, 4.0]])
        result, valid = phase_imbalance(values, epsilon=1e-6)
        self.assertAlmostEqual(result.iloc[0], 1.0 / (10.0 + 1e-6))
        self.assertEqual(result.iloc[1], 0.0)
        self.assertTrue(valid.all())

    def test_voltage_imbalance_formula(self):
        values = pd.DataFrame([[220.0, 230.0, 240.0]])
        result, _ = phase_imbalance(values, epsilon=1e-6)
        self.assertAlmostEqual(result.iloc[0], 10.0 / (230.0 + 1e-6))

    def test_grid_power_factor(self):
        result, valid = grid_power_factor(pd.Series([-3.0, 0.0]), pd.Series([4.0, 0.0]))
        self.assertAlmostEqual(result.iloc[0], 0.6)
        self.assertTrue(valid.iloc[0])
        self.assertFalse(valid.iloc[1])

    def farm_a_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame(
            {sensor: [1.0, 1.0] for sensor in required_sensor_ids(self.config, "A")}
        )
        frame["sensor_0"] = 10.0
        frame["sensor_12"] = 50.0
        frame["sensor_11"] = 60.0
        frame["sensor_18"] = 20.0
        frame["sensor_52"] = [0.0, 2.0]
        frame["power_30"] = 3.0
        frame["sensor_31"] = 4.0
        frame[["sensor_23", "sensor_24", "sensor_25"]] = [9.0, 10.0, 11.0]
        frame[["sensor_32", "sensor_33", "sensor_34"]] = [220.0, 230.0, 240.0]
        frame["sensor_26"] = 50.0
        return frame

    def test_speed_ratio_mask_and_thermal_deltas(self):
        features, valid = compute_physical_features(
            self.farm_a_frame(), "A", self.config, frequency_validated=True
        )
        self.assertTrue(np.isnan(features.loc[0, "generator_rotor_speed_ratio"]))
        self.assertFalse(valid.loc[0, "generator_rotor_speed_ratio"])
        self.assertEqual(features.loc[1, "generator_rotor_speed_ratio"], 10.0)
        self.assertTrue(valid.loc[1, "generator_rotor_speed_ratio"])
        self.assertTrue((features["gearbox_oil_rise_C"] == 40.0).all())
        self.assertTrue((features["gearbox_bearing_hotspot_over_oil_C"] == 10.0).all())

    def test_feature_order_and_manifest(self):
        features, _ = compute_physical_features(
            self.farm_a_frame(), "A", self.config, frequency_validated=True
        )
        self.assertEqual(features.columns.tolist(), self.config["strict_feature_order"])
        manifest = build_manifest(self.config)
        self.assertEqual(manifest["feature_name"].tolist(), self.config["strict_feature_order"])
        self.assertEqual(len(manifest), 10)


class MetadataAndLeakageTests(unittest.TestCase):
    def test_metadata_flattening_uses_actual_timestamps(self):
        index = pd.date_range("2024-01-01", periods=SEQ_LEN, freq="10min")
        labels = np.zeros(SEQ_LEN, dtype=np.uint8)
        mask = np.ones(SEQ_LEN, dtype=np.uint8)
        metadata = build_timestep_metadata(
            farm="C", split="train", sequence_idx=3, index=index, asset_id=7,
            event_id=11, event_label="normal", event_end=index[-1], labels=labels, mask=mask,
        )
        self.assertEqual(len(metadata), SEQ_LEN)
        np.testing.assert_array_equal(metadata["timestep_idx"], np.arange(SEQ_LEN))
        self.assertEqual(metadata["window_end"].nunique(), SEQ_LEN)
        self.assertTrue(metadata["window_end"].is_monotonic_increasing)

    def test_scaler_uses_train_valid_timesteps_only(self):
        train = np.asarray([[[1.0, 10.0], [3.0, 30.0], [1000.0, 1000.0]]], dtype=np.float32)
        mask = np.asarray([[1, 1, 0]], dtype=np.uint8)
        external = np.asarray([[[100.0, 200.0]]], dtype=np.float32)
        mean, std = fit_train_scaler(train, mask)
        np.testing.assert_allclose(mean, [2.0, 20.0])
        np.testing.assert_allclose(std, [1.0, 10.0])
        np.testing.assert_allclose(apply_scaler(external, mean, std), [[[98.0, 18.0]]])

    def test_scaler_provenance_and_order_are_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scaler.npz"
            save_scaler(path, np.zeros(2), np.ones(2), ["a", "b"])
            load_scaler(path, ["a", "b"])
            with self.assertRaises(ValueError):
                load_scaler(path, ["b", "a"])
            bad = Path(directory) / "bad.npz"
            np.savez(
                bad, mean=np.zeros(2), std=np.ones(2), feature_names=np.asarray(["a", "b"]),
                source_farm=np.asarray("A"), source_split=np.asarray("test"),
            )
            with self.assertRaisesRegex(ValueError, "Farm C train"):
                load_scaler(bad, ["a", "b"])

    def test_target_power_calibration_requires_explicit_flag(self):
        config = load_mapping(CONFIG_PATH)
        args = Namespace(
            include_power_residual=True,
            power_curve=None,
            farm="A",
            allow_target_power_calibration=False,
            raw_dir=Path("unused"),
            output_dir=Path("unused"),
        )
        with self.assertRaisesRegex(ValueError, "not zero-shot DG"):
            prepare_power_curve(args, config, pd.DataFrame(), {})


if __name__ == "__main__":
    unittest.main()

