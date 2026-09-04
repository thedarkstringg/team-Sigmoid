"""Farm-agnostic computation of the physical-v1 feature representation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import yaml


STRICT_SCHEMA_VERSION = "physical-v1"


def load_mapping(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config.get("schema_version") != STRICT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported physical schema: {config.get('schema_version')}")
    order = config.get("strict_feature_order", [])
    if len(order) != 10 or len(order) != len(set(order)):
        raise ValueError("Strict physical-v1 schema must contain 10 unique features")
    missing = set(order) - set(config.get("features", {}))
    if missing:
        raise ValueError(f"Feature definitions missing from mapping: {sorted(missing)}")
    return config


def feature_order(config: Mapping[str, Any], include_power_residual: bool = False) -> list[str]:
    names = list(config["strict_feature_order"])
    if include_power_residual:
        names.append("power_residual")
    return names


def _walk_sensor_ids(value: Any) -> Iterable[str]:
    if isinstance(value, str) and (
        value.startswith("sensor_")
        or value.startswith("wind_speed_")
        or value.startswith("power_")
        or value.startswith("reactive_power_")
    ):
        yield value
    elif isinstance(value, Mapping):
        for child in value.values():
            yield from _walk_sensor_ids(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_sensor_ids(child)


def required_sensor_ids(
    config: Mapping[str, Any], farm: str, include_power_residual: bool = False
) -> list[str]:
    farm = farm.upper()
    definitions = [config["features"][name]["farms"][farm] for name in config["strict_feature_order"]]
    if include_power_residual:
        definitions.append(config["optional_features"]["power_residual"]["farms"][farm])
    return sorted(set(sensor for definition in definitions for sensor in _walk_sensor_ids(definition)))


def resolve_average_columns(
    available_columns: Iterable[str], sensor_ids: Iterable[str]
) -> dict[str, str]:
    """Resolve logical sensors to Avg columns without ever selecting Min/Max/Std."""
    available = set(available_columns)
    resolved: dict[str, str] = {}
    for sensor in sensor_ids:
        candidates = (f"{sensor}_avg", f"{sensor}_average", sensor)
        matches = [candidate for candidate in candidates if candidate in available]
        if not matches:
            raise ValueError(f"No Avg-compatible column found for {sensor}")
        resolved[sensor] = matches[0]
    return resolved


def rename_to_sensor_ids(frame: pd.DataFrame, resolved: Mapping[str, str]) -> pd.DataFrame:
    missing = set(resolved.values()) - set(frame.columns)
    if missing:
        raise ValueError(f"Raw frame is missing resolved columns: {sorted(missing)}")
    return frame.loc[:, list(resolved.values())].rename(
        columns={raw: sensor for sensor, raw in resolved.items()}
    )


def wrap_to_180(values: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    return (values + 180.0) % 360.0 - 180.0


def phase_imbalance(values: pd.DataFrame, epsilon: float = 1.0e-6) -> tuple[pd.Series, pd.Series]:
    mean = values.mean(axis=1)
    numerator = values.sub(mean, axis=0).abs().max(axis=1)
    result = numerator / (mean.abs() + epsilon)
    valid = np.isfinite(values).all(axis=1) & np.isfinite(result)
    return result, valid


def grid_power_factor(active: pd.Series, reactive: pd.Series) -> tuple[pd.Series, pd.Series]:
    apparent = np.sqrt(active.pow(2) + reactive.pow(2))
    result = active.abs() / apparent
    valid = np.isfinite(active) & np.isfinite(reactive) & np.isfinite(result)
    return result, valid


def _mean(frame: pd.DataFrame, sensors: list[str]) -> pd.Series:
    return frame[sensors].mean(axis=1, skipna=False)


def _finite(series: pd.Series) -> pd.Series:
    return pd.Series(np.isfinite(series.to_numpy(dtype=float)), index=series.index)


def compute_physical_features(
    raw_by_sensor: pd.DataFrame,
    farm: str,
    config: Mapping[str, Any],
    *,
    frequency_validated: bool = False,
    power_curve: "PowerCurve | None" = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ordered features and per-feature validity masks.

    ``raw_by_sensor`` must use logical sensor IDs as columns. Invalid values are
    left as NaN and must be filled only inside an approved sequence window.
    """
    farm = farm.upper()
    if farm not in {"A", "B", "C"}:
        raise ValueError(f"Unknown farm: {farm}")
    settings = config["global_validity"]
    features: dict[str, pd.Series] = {}
    validity: dict[str, pd.Series] = {}

    definition = config["features"]["wind_speed_mps"]["farms"][farm]
    wind = raw_by_sensor[definition["wind_speed"]].astype(float)
    features["wind_speed_mps"] = wind
    validity["wind_speed_mps"] = _finite(wind) & (wind >= 0)

    definition = config["features"]["pitch_angle_deg"]["farms"][farm]
    pitch = _mean(raw_by_sensor, definition["pitch_angles"])
    features["pitch_angle_deg"] = pitch
    validity["pitch_angle_deg"] = _finite(pitch)

    definition = config["features"]["yaw_misalignment_deg"]["farms"][farm]
    if definition["mode"] == "direct_relative":
        relative = raw_by_sensor[definition["relative_angle"]].astype(float)
    elif definition["mode"] == "absolute_minus_nacelle":
        relative = (
            raw_by_sensor[definition["absolute_wind_direction"]].astype(float)
            - raw_by_sensor[definition["nacelle_direction"]].astype(float)
        )
    else:
        raise ValueError(f"Unknown yaw formula mode: {definition['mode']}")
    yaw = wrap_to_180(relative).abs()
    features["yaw_misalignment_deg"] = yaw
    validity["yaw_misalignment_deg"] = _finite(yaw) & yaw.between(0, 180, inclusive="both")

    definition = config["features"]["gearbox_oil_rise_C"]["farms"][farm]
    oil = _mean(raw_by_sensor, definition["gearbox_oil_temperatures"])
    oil_rise = oil - raw_by_sensor[definition["ambient_temperature"]].astype(float)
    features["gearbox_oil_rise_C"] = oil_rise
    validity["gearbox_oil_rise_C"] = _finite(oil_rise)

    definition = config["features"]["gearbox_bearing_hotspot_over_oil_C"]["farms"][farm]
    bearing_hotspot = raw_by_sensor[definition["gearbox_bearing_temperatures"]].max(
        axis=1, skipna=False
    )
    bearing_oil = _mean(raw_by_sensor, definition["gearbox_oil_temperatures"])
    hotspot = bearing_hotspot - bearing_oil
    features["gearbox_bearing_hotspot_over_oil_C"] = hotspot
    validity["gearbox_bearing_hotspot_over_oil_C"] = _finite(hotspot)

    definition = config["features"]["generator_rotor_speed_ratio"]["farms"][farm]
    generator = raw_by_sensor[definition["generator_speed"]].astype(float)
    if definition["generator_speed_unit"] == "rad/s":
        generator = generator * 60.0 / (2.0 * np.pi)
    rotor = _mean(raw_by_sensor, definition["rotor_speeds"])
    speed_valid = (
        _finite(generator)
        & _finite(rotor)
        & (rotor >= float(settings["minimum_rotor_speed_rpm"]))
    )
    speed_ratio = pd.Series(np.nan, index=raw_by_sensor.index, dtype=float)
    speed_ratio.loc[speed_valid] = generator.loc[speed_valid] / rotor.loc[speed_valid]
    features["generator_rotor_speed_ratio"] = speed_ratio
    validity["generator_rotor_speed_ratio"] = speed_valid & _finite(speed_ratio)

    definition = config["features"]["grid_power_factor"]["farms"][farm]
    active = raw_by_sensor[definition["active_power"]].astype(float)
    reactive = raw_by_sensor[definition["reactive_power"]].astype(float)
    factor, factor_valid = grid_power_factor(active, reactive)
    apparent = np.sqrt(active.pow(2) + reactive.pow(2))
    factor_valid &= apparent >= float(settings["minimum_apparent_power_kva"])
    factor = factor.where(factor_valid)
    features["grid_power_factor"] = factor
    validity["grid_power_factor"] = factor_valid

    definition = config["features"]["grid_current_imbalance"]["farms"][farm]
    currents = raw_by_sensor[definition["phase_currents"]].astype(float)
    imbalance, imbalance_valid = phase_imbalance(currents, float(settings["epsilon"]))
    imbalance_valid &= currents.mean(axis=1).abs() >= float(settings["minimum_mean_current_a"])
    features["grid_current_imbalance"] = imbalance.where(imbalance_valid)
    validity["grid_current_imbalance"] = imbalance_valid

    definition = config["features"]["grid_voltage_imbalance"]["farms"][farm]
    voltages = raw_by_sensor[definition["phase_voltages"]].astype(float)
    imbalance, imbalance_valid = phase_imbalance(voltages, float(settings["epsilon"]))
    imbalance_valid &= voltages.mean(axis=1).abs() >= float(settings["minimum_mean_voltage_v"])
    features["grid_voltage_imbalance"] = imbalance.where(imbalance_valid)
    validity["grid_voltage_imbalance"] = imbalance_valid

    definition = config["features"]["grid_frequency_deviation_Hz"]["farms"][farm]
    frequency = raw_by_sensor[definition["grid_frequency"]].astype(float)
    deviation = (frequency - float(settings["expected_grid_frequency_hz"])).abs()
    frequency_mask = _finite(deviation) & bool(frequency_validated)
    features["grid_frequency_deviation_Hz"] = deviation.where(frequency_mask)
    validity["grid_frequency_deviation_Hz"] = frequency_mask

    if power_curve is not None:
        optional = config["optional_features"]["power_residual"]["farms"][farm]
        optional_wind = raw_by_sensor[optional["wind_speed"]].astype(float)
        optional_power = raw_by_sensor[optional["active_power"]].astype(float)
        residual = power_curve.residual(optional_wind, optional_power)
        features["power_residual"] = residual
        validity["power_residual"] = _finite(residual)

    ordered = feature_order(config, include_power_residual=power_curve is not None)
    feature_frame = pd.DataFrame(features, index=raw_by_sensor.index)[ordered]
    valid_frame = pd.DataFrame(validity, index=raw_by_sensor.index)[ordered].astype(bool)
    return feature_frame, valid_frame


@dataclass(frozen=True)
class PowerCurve:
    wind_speed_centers: np.ndarray
    median_power: np.ndarray
    power_scale: float
    provenance: dict[str, Any]

    def expected_power(self, wind_speed: pd.Series | np.ndarray) -> np.ndarray:
        values = np.asarray(wind_speed, dtype=float)
        return np.interp(
            values,
            self.wind_speed_centers,
            self.median_power,
            left=self.median_power[0],
            right=self.median_power[-1],
        )

    def residual(self, wind_speed: pd.Series, active_power: pd.Series) -> pd.Series:
        expected = self.expected_power(wind_speed)
        result = (active_power.to_numpy(dtype=float) - expected) / self.power_scale
        return pd.Series(result, index=active_power.index)

    def save(self, path: str | Path) -> None:
        payload = {
            "wind_speed_centers": self.wind_speed_centers.tolist(),
            "median_power": self.median_power.tolist(),
            "power_scale": self.power_scale,
            "provenance": self.provenance,
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "PowerCurve":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            wind_speed_centers=np.asarray(payload["wind_speed_centers"], dtype=float),
            median_power=np.asarray(payload["median_power"], dtype=float),
            power_scale=float(payload["power_scale"]),
            provenance=dict(payload["provenance"]),
        )


def fit_binned_power_curve(
    wind_speed: pd.Series,
    active_power: pd.Series,
    *,
    bin_width: float = 0.5,
    min_bin_count: int = 20,
    provenance: Mapping[str, Any],
) -> PowerCurve:
    """Fit a median binned curve; caller is responsible for enforcing fit scope."""
    values = pd.DataFrame({"wind": wind_speed, "power": active_power}).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    values = values[values["wind"] >= 0]
    if values.empty:
        raise ValueError("No finite wind/power observations for power-curve fitting")
    bins = np.floor(values["wind"] / bin_width).astype(int)
    grouped = values.groupby(bins).agg(wind=("wind", "median"), power=("power", "median"), count=("power", "size"))
    grouped = grouped[grouped["count"] >= min_bin_count].sort_values("wind")
    if len(grouped) < 2:
        raise ValueError("Power curve needs at least two populated wind-speed bins")
    scale = float(values["power"].abs().quantile(0.99))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Power-curve normalization scale is not positive")
    return PowerCurve(
        wind_speed_centers=grouped["wind"].to_numpy(dtype=float),
        median_power=grouped["power"].to_numpy(dtype=float),
        power_scale=scale,
        provenance=dict(provenance),
    )
