from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.data import audit_crossfarm
from src.data.build_physical_manifest import build_manifest
from src.data.audit_crossfarm import audit_farm_c_temporal_boundaries
from src.data.export_physical_sequences import build_event_sequences, prepare_power_curve
from src.data.physical_features import (
    compute_physical_features,
    fit_binned_power_curve,
    grid_power_factor,
    load_mapping,
    phase_imbalance,
    required_sensor_ids,
    resolve_average_columns,
    wrap_to_180,
)
from src.data.sequence_utils import (
    SEQ_LEN,
    apply_scaler,
    build_timestep_metadata,
    detect_care_delimiter,
    fit_train_scaler,
    load_scaler,
    read_care_csv,
    resolve_event_boundaries,
    save_scaler,
)


CONFIG_PATH = Path("configs/physical_sensor_mapping.yaml")


class CareCsvReaderTests(unittest.TestCase):
    def test_comma_and_semicolon_files_expose_the_same_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            comma_path = root / "comma.csv"
            semicolon_path = root / "semicolon.csv"
            comma_path.write_text("event_id,asset_id,value\n1,A,10\n2,B,20\n", encoding="utf-8")
            semicolon_path.write_text(
                "event_id;asset_id;value\n1;A;10\n2;B;20\n", encoding="utf-8"
            )

            self.assertEqual(detect_care_delimiter(comma_path), ",")
            self.assertEqual(detect_care_delimiter(semicolon_path), ";")
            comma = read_care_csv(comma_path)
            semicolon = read_care_csv(semicolon_path)

            self.assertEqual(comma.columns.tolist(), ["event_id", "asset_id", "value"])
            self.assertEqual(semicolon.columns.tolist(), comma.columns.tolist())
            pd.testing.assert_frame_equal(semicolon, comma)

    def test_tab_delimiter_and_nrows_are_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tab.csv"
            path.write_text("event_id\tasset_id\tvalue\n1\tA\t10\n2\tB\t20\n", encoding="utf-8")

            self.assertEqual(detect_care_delimiter(path), "\t")
            frame = read_care_csv(path, nrows=1)

            self.assertEqual(len(frame), 1)
            self.assertEqual(frame.iloc[0].to_dict(), {"event_id": 1, "asset_id": "A", "value": 10})

    def test_usecols_is_forwarded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "semicolon.csv"
            path.write_text(
                "event_id;asset_id;value\n1;A;10\n2;B;20\n", encoding="utf-8"
            )

            frame = read_care_csv(path, usecols=["asset_id", "value"])

            self.assertEqual(frame.columns.tolist(), ["asset_id", "value"])


class EventBoundaryTests(unittest.TestCase):
    @staticmethod
    def write_raw(path: Path, timestamps, train_test) -> None:
        pd.DataFrame(
            {"time_stamp": timestamps, "train_test": train_test, "asset_id": 7}
        ).to_csv(path, index=False)

    def test_farm_c_resolves_integer_like_ids_from_original_row_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "comma_4.csv"
            self.write_raw(
                path,
                ["2024-01-10", "2023-07-31 10:00:00", "2023-08-19 10:00:00"],
                ["train", " Prediction ", "PREDICTION"],
            )
            event = {
                "event_start": "2020-01-01",
                "event_end": "2020-01-02",
                "event_start_id": "1.0",
                "event_end_id": 2.0,
            }

            boundaries = resolve_event_boundaries(path, event, "C")

            self.assertEqual(boundaries.event_start, pd.Timestamp("2023-07-31 10:00:00"))
            self.assertEqual(boundaries.event_end, pd.Timestamp("2023-08-19 10:00:00"))
            self.assertEqual(boundaries.source, "raw_row_ids")

    def test_corrupt_farm_c_metadata_does_not_override_valid_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "comma_11.csv"
            self.write_raw(
                path,
                ["2023-10-17 12:30:00", "2023-11-08 10:30:00"],
                ["prediction", "prediction"],
            )
            event_info = pd.DataFrame(
                [
                    {
                        "event_id": 11,
                        "event_start": "not-a-timestamp",
                        "event_end": "1900-01-01",
                        "event_start_id": 0,
                        "event_end_id": 1,
                    }
                ]
            )

            boundaries = resolve_event_boundaries(path, event_info.iloc[0], "C")
            checks, failures, mismatches = audit_farm_c_temporal_boundaries(root, event_info)

            self.assertEqual(boundaries.event_end, pd.Timestamp("2023-11-08 10:30:00"))
            self.assertTrue(checks[0]["valid"])
            self.assertEqual(failures, [])
            self.assertEqual(mismatches[0]["classification"], "metadata_mismatch")
            self.assertTrue(mismatches[0]["raw_boundary_valid"])
            self.assertEqual(
                {item["field"] for item in mismatches[0]["mismatches"]},
                {"event_start", "event_end"},
            )

    def test_farm_c_out_of_range_id_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "comma_70.csv"
            self.write_raw(path, ["2023-11-10", "2023-11-30"], ["prediction"] * 2)
            event = {
                "event_start": "ignored",
                "event_end": "ignored",
                "event_start_id": 0,
                "event_end_id": 2,
            }

            with self.assertRaisesRegex(ValueError, "event_end_id=2 is out of range"):
                resolve_event_boundaries(path, event, "C")

    def test_farm_c_missing_non_integer_and_reversed_ids_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "comma_70.csv"
            self.write_raw(path, ["2023-11-10", "2023-11-30"], ["prediction"] * 2)
            base = {
                "event_start": "ignored",
                "event_end": "ignored",
                "event_start_id": 0,
                "event_end_id": 1,
            }
            cases = [
                ({key: value for key, value in base.items() if key != "event_start_id"}, "missing event_start_id"),
                ({**base, "event_start_id": 0.5}, "event_start_id must be an integer-like"),
                ({**base, "event_start_id": 1, "event_end_id": 0}, "is greater than"),
            ]

            for event, message in cases:
                with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                    resolve_event_boundaries(path, event, "C")

    def test_farm_c_invalid_or_reversed_resolved_timestamps_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "comma_70.csv"
            event = {
                "event_start_id": 0,
                "event_end_id": 1,
            }
            cases = [
                (["invalid", "2023-11-30"], "does not contain a valid timestamp"),
                (["2023-12-01", "2023-11-30"], "is after event_end"),
            ]

            for timestamps, message in cases:
                with self.subTest(message=message):
                    self.write_raw(path, timestamps, ["prediction"] * 2)
                    with self.assertRaisesRegex(ValueError, message):
                        resolve_event_boundaries(path, event, "C")

    def test_farm_c_non_prediction_boundary_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "comma_70.csv"
            self.write_raw(path, ["2023-11-10", "2023-11-30"], ["train", "prediction"])
            event = {
                "event_start": "ignored",
                "event_end": "ignored",
                "event_start_id": 0,
                "event_end_id": 1,
            }

            with self.assertRaisesRegex(ValueError, "both have train_test=prediction"):
                resolve_event_boundaries(path, event, "C")

    def test_farm_a_and_b_continue_to_use_metadata_timestamps(self):
        event = {"event_start": "2024-02-01 01:00", "event_end": "2024-02-02 02:00"}
        missing_raw = Path("raw-file-is-not-consulted-for-external-farms.csv")

        for farm in ("A", "B"):
            with self.subTest(farm=farm):
                boundaries = resolve_event_boundaries(missing_raw, event, farm)
                self.assertEqual(boundaries.event_start, pd.Timestamp(event["event_start"]))
                self.assertEqual(boundaries.event_end, pd.Timestamp(event["event_end"]))
                self.assertEqual(boundaries.source, "event_metadata")

    def test_sequence_labels_and_fault_time_use_resolved_farm_c_end(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "comma_4.csv"
            timestamps = pd.date_range("2024-01-01", periods=151, freq="10min")
            self.write_raw(path, timestamps, ["prediction"] * len(timestamps))
            event = pd.Series(
                {
                    "event_id": 4,
                    "event_label": "anomaly",
                    "event_start": "2030-01-01",
                    "event_end": "2023-01-01",
                    "event_start_id": 144,
                    "event_end_id": 150,
                }
            )

            def fake_features(raw, farm, config, **kwargs):
                features = pd.DataFrame({"feature": 1.0}, index=raw.index)
                valid = pd.DataFrame({"feature": True}, index=raw.index)
                return features, valid

            with patch(
                "src.data.export_physical_sequences.compute_physical_features",
                side_effect=fake_features,
            ):
                sequences = build_event_sequences(
                    raw_path=path,
                    event_row=event,
                    farm="C",
                    split="train",
                    config={},
                    sensor_ids=[],
                    power_curve=None,
                )

            self.assertEqual(len(sequences), 1)
            sequence = sequences[0]
            resolved_end = timestamps[150]
            self.assertEqual(sequence["event_end"], resolved_end)
            self.assertTrue(sequence["y"].all())
            metadata = build_timestep_metadata(
                farm="C",
                split="train",
                sequence_idx=0,
                index=sequence["index"],
                asset_id=sequence["asset_id"],
                event_id=sequence["event_id"],
                event_label=sequence["event_label"],
                event_end=sequence["event_end"],
                labels=sequence["y"],
                mask=sequence["mask"],
            )
            self.assertTrue((metadata["fault_time"] == resolved_end).all())
            self.assertTrue(metadata["label"].all())

    def test_farm_c_audit_is_not_export_ready_when_boundary_ids_are_invalid(self):
        config = load_mapping(CONFIG_PATH)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "audit.json"
            pd.DataFrame(
                [
                    {
                        "event_id": 4,
                        "event_label": "anomaly",
                        "event_start": "bad metadata",
                        "event_end": "bad metadata",
                        "event_start_id": 0,
                        "event_end_id": 2,
                    }
                ]
            ).to_csv(root / "comma_event_info.csv", index=False)
            sensors = required_sensor_ids(config, "C")
            raw = pd.DataFrame({sensor: [1.0, 1.0] for sensor in sensors})
            frequency_sensor = config["features"]["grid_frequency_deviation_Hz"]["farms"]["C"][
                "grid_frequency"
            ]
            raw[frequency_sensor] = 50.0
            raw["time_stamp"] = ["2024-01-01 00:00", "2024-01-01 00:10"]
            raw["train_test"] = "prediction"
            raw["asset_id"] = 7
            raw.to_csv(root / "comma_4.csv", index=False)

            argv = [
                "audit_crossfarm.py",
                "--farm",
                "C",
                "--raw-dir",
                str(root),
                "--config",
                str(CONFIG_PATH),
                "--output",
                str(output),
            ]
            with patch("sys.argv", argv):
                audit_crossfarm.main()

            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertFalse(report["strict_export_ready"])
            self.assertEqual(len(report["temporal_boundary_failures"]), 1)
            self.assertIn("out of range", report["temporal_boundary_failures"][0]["reason"])


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

    def test_column_resolution_selects_only_average_compatible_names(self):
        resolved = resolve_average_columns(
            ["sensor_1", "sensor_1_avg", "sensor_1_max", "sensor_2_average"],
            ["sensor_1", "sensor_2"],
        )
        self.assertEqual(resolved, {"sensor_1": "sensor_1_avg", "sensor_2": "sensor_2_average"})
        with self.assertRaises(ValueError):
            resolve_average_columns(["sensor_3_max", "sensor_3_std"], ["sensor_3"])

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

    def test_invalid_physical_values_are_nan_for_safe_fill(self):
        frame = self.farm_a_frame()
        frame.loc[1, "wind_speed_3"] = -1.0
        features, valid = compute_physical_features(
            frame, "A", self.config, frequency_validated=True
        )
        self.assertFalse(valid.loc[1, "wind_speed_mps"])
        self.assertTrue(np.isnan(features.loc[1, "wind_speed_mps"]))

    def test_feature_order_and_manifest(self):
        features, _ = compute_physical_features(
            self.farm_a_frame(), "A", self.config, frequency_validated=True
        )
        self.assertEqual(features.columns.tolist(), self.config["strict_feature_order"])
        manifest = build_manifest(self.config)
        self.assertEqual(manifest["feature_name"].tolist(), self.config["strict_feature_order"])
        self.assertEqual(len(manifest), 10)

    def valid_frame(self, farm: str) -> pd.DataFrame:
        frame = pd.DataFrame(
            {sensor: [1.0] for sensor in required_sensor_ids(self.config, farm)}
        )
        definitions = self.config["features"]
        power = definitions["grid_power_factor"]["farms"][farm]
        frame[power["active_power"]] = 3.0
        frame[power["reactive_power"]] = 4.0
        current = definitions["grid_current_imbalance"]["farms"][farm]["phase_currents"]
        frame[current] = [9.0, 10.0, 11.0]
        voltage = definitions["grid_voltage_imbalance"]["farms"][farm]["phase_voltages"]
        frame[voltage] = [220.0, 230.0, 240.0]
        frequency = definitions["grid_frequency_deviation_Hz"]["farms"][farm]["grid_frequency"]
        frame[frequency] = 50.0
        speed = definitions["generator_rotor_speed_ratio"]["farms"][farm]
        frame[speed["rotor_speeds"]] = 2.0
        return frame

    def test_exact_farm_specific_yaw_formulas(self):
        farm_b = self.valid_frame("B")
        farm_b["sensor_4"] = 350.0
        farm_b["sensor_21"] = 10.0
        b_features, _ = compute_physical_features(
            farm_b, "B", self.config, frequency_validated=True
        )
        self.assertEqual(b_features.loc[0, "yaw_misalignment_deg"], 20.0)
        farm_c = self.valid_frame("C")
        farm_c["sensor_124"] = -190.0
        c_features, _ = compute_physical_features(
            farm_c, "C", self.config, frequency_validated=True
        )
        self.assertEqual(c_features.loc[0, "yaw_misalignment_deg"], 170.0)

    def test_farm_c_generator_speed_unit_conversion(self):
        frame = self.valid_frame("C")
        frame["sensor_8"] = 2.0 * np.pi
        frame[["sensor_144", "sensor_145"]] = 2.0
        features, valid = compute_physical_features(
            frame, "C", self.config, frequency_validated=True
        )
        self.assertTrue(valid.loc[0, "generator_rotor_speed_ratio"])
        self.assertAlmostEqual(features.loc[0, "generator_rotor_speed_ratio"], 30.0)

    def test_binned_power_curve_median_and_interpolation(self):
        curve = fit_binned_power_curve(
            pd.Series([0.1, 0.2, 0.3, 1.1, 1.2, 1.3]),
            pd.Series([0.0, 2.0, 100.0, 10.0, 12.0, 1000.0]),
            bin_width=1.0,
            min_bin_count=3,
            provenance={"farm": "C", "setting": "source-domain-train-normal-only"},
        )
        np.testing.assert_allclose(curve.median_power, [2.0, 12.0])
        expected = curve.expected_power(np.asarray([0.75]))
        self.assertGreater(expected[0], 2.0)
        self.assertLess(expected[0], 12.0)


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

    def test_scaler_statistics_avoid_float32_accumulation_error(self):
        valid_count = 100_000
        train = np.empty((1_001, 100, 1), dtype=np.float32)
        train[:500, :, 0] = 1_000_000.0 - 1.0
        train[500:1_000, :, 0] = 1_000_000.0 + 1.0
        train[1_000, :, 0] = 0.0
        mask = np.ones(train.shape[:2], dtype=np.uint8)
        mask[1_000, :] = 0

        mean, std = fit_train_scaler(train, mask)
        scaled = apply_scaler(train, mean, std)
        valid = scaled.reshape(-1, 1)[mask.reshape(-1).astype(bool)]

        self.assertEqual(valid.shape[0], valid_count)
        self.assertEqual(mean.dtype, np.float32)
        self.assertEqual(std.dtype, np.float32)
        np.testing.assert_allclose(valid.mean(axis=0), 0.0, atol=2e-4)
        np.testing.assert_allclose(valid.std(axis=0), 1.0, atol=2e-4)

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
