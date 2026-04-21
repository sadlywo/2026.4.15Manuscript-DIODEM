from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy.signal import welch


def compute_rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((prediction - target) ** 2)))


def compute_pearson(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=float)
    target = np.asarray(target, dtype=float)
    if prediction.std() < 1e-12 or target.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(prediction, target)[0, 1])


def compute_psd_distance(prediction: np.ndarray, target: np.ndarray, sampling_frequency: float) -> float:
    if len(prediction) < 4:
        return np.nan
    nperseg = min(256, len(prediction))
    freq_pred, psd_pred = welch(prediction, fs=sampling_frequency, nperseg=nperseg)
    freq_target, psd_target = welch(target, fs=sampling_frequency, nperseg=nperseg)
    if len(freq_pred) != len(freq_target):
        return np.nan
    return float(np.sqrt(np.mean((psd_pred - psd_target) ** 2)))


def _nanmean(values: List[float]) -> float:
    array = np.asarray(values, dtype=float)
    if array.size == 0 or np.all(np.isnan(array)):
        return np.nan
    return float(np.nanmean(array))


def _high_band_power(signal: np.ndarray, sampling_frequency: float) -> float:
    if len(signal) < 4:
        return np.nan
    nperseg = min(256, len(signal))
    freqs, psd = welch(signal, fs=sampling_frequency, nperseg=nperseg)
    mask = (freqs >= 5.0) & (freqs <= min(15.0, sampling_frequency / 2.0))
    if not np.any(mask):
        return np.nan
    return float(np.trapz(psd[mask], freqs[mask]))


def _compute_signal_norm(values: np.ndarray, channel_names: List[str], prefix: str) -> np.ndarray:
    indices = [idx for idx, name in enumerate(channel_names) if name.startswith(prefix)]
    if not indices:
        return np.zeros(values.shape[0], dtype=np.float32)
    return np.linalg.norm(values[:, indices], axis=1)


def compute_window_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    input_signal: np.ndarray,
    metadata: Dict[str, Any],
    channels: List[str],
    sampling_frequency: float,
) -> Dict[str, Any]:
    """Compute all evaluation metrics for one cached window in the signal's native units."""
    if prediction.shape != target.shape or prediction.shape != input_signal.shape:
        raise ValueError("Prediction, target, and input windows must share the same shape.")

    row: Dict[str, Any] = dict(metadata)
    per_channel_rmse = []
    per_channel_input_rmse = []
    per_channel_corr = []
    per_channel_psd = []
    for channel_idx, channel_name in enumerate(channels):
        pred_channel = prediction[:, channel_idx]
        target_channel = target[:, channel_idx]
        input_channel = input_signal[:, channel_idx]
        row[f"rmse_{channel_name}"] = compute_rmse(pred_channel, target_channel)
        row[f"pearson_{channel_name}"] = compute_pearson(pred_channel, target_channel)
        row[f"input_rmse_{channel_name}"] = compute_rmse(input_channel, target_channel)
        row[f"psd_distance_{channel_name}"] = compute_psd_distance(
            pred_channel,
            target_channel,
            sampling_frequency,
        )
        per_channel_rmse.append(row[f"rmse_{channel_name}"])
        per_channel_input_rmse.append(row[f"input_rmse_{channel_name}"])
        per_channel_corr.append(row[f"pearson_{channel_name}"])
        per_channel_psd.append(row[f"psd_distance_{channel_name}"])

    pred_acc_norm = _compute_signal_norm(prediction, channels, "acc_")
    target_acc_norm = _compute_signal_norm(target, channels, "acc_")
    pred_gyr_norm = _compute_signal_norm(prediction, channels, "gyr_")
    target_gyr_norm = _compute_signal_norm(target, channels, "gyr_")
    input_acc_norm = _compute_signal_norm(input_signal, channels, "acc_")
    input_gyr_norm = _compute_signal_norm(input_signal, channels, "gyr_")

    row["rmse_mean"] = float(np.mean(per_channel_rmse))
    row["input_rmse_mean"] = float(np.mean(per_channel_input_rmse))
    row["pearson_mean"] = _nanmean(per_channel_corr)
    row["psd_distance_mean"] = _nanmean(per_channel_psd)
    row["acc_norm_rmse"] = compute_rmse(pred_acc_norm, target_acc_norm)
    row["gyr_norm_rmse"] = compute_rmse(pred_gyr_norm, target_gyr_norm)
    row["input_acc_norm_rmse"] = compute_rmse(input_acc_norm, target_acc_norm)
    row["input_gyr_norm_rmse"] = compute_rmse(input_gyr_norm, target_gyr_norm)
    row["acc_channel_rmse_mean"] = _nanmean(
        [row.get(f"rmse_{channel_name}") for channel_name in channels if channel_name.startswith("acc_")]
    )
    row["gyr_channel_rmse_mean"] = _nanmean(
        [row.get(f"rmse_{channel_name}") for channel_name in channels if channel_name.startswith("gyr_")]
    )
    row["input_acc_channel_rmse_mean"] = _nanmean(
        [row.get(f"input_rmse_{channel_name}") for channel_name in channels if channel_name.startswith("acc_")]
    )
    row["input_gyr_channel_rmse_mean"] = _nanmean(
        [row.get(f"input_rmse_{channel_name}") for channel_name in channels if channel_name.startswith("gyr_")]
    )
    row["acc_channel_rmse_improvement"] = float(row["input_acc_channel_rmse_mean"] - row["acc_channel_rmse_mean"])
    row["gyr_channel_rmse_improvement"] = float(row["input_gyr_channel_rmse_mean"] - row["gyr_channel_rmse_mean"])

    input_acc_ratio = (_high_band_power(input_acc_norm, sampling_frequency) + 1e-8) / (
        _high_band_power(target_acc_norm, sampling_frequency) + 1e-8
    )
    pred_acc_ratio = (_high_band_power(pred_acc_norm, sampling_frequency) + 1e-8) / (
        _high_band_power(target_acc_norm, sampling_frequency) + 1e-8
    )
    input_gyr_ratio = (_high_band_power(input_gyr_norm, sampling_frequency) + 1e-8) / (
        _high_band_power(target_gyr_norm, sampling_frequency) + 1e-8
    )
    pred_gyr_ratio = (_high_band_power(pred_gyr_norm, sampling_frequency) + 1e-8) / (
        _high_band_power(target_gyr_norm, sampling_frequency) + 1e-8
    )
    row["hf_ratio_improvement_acc"] = float(abs(input_acc_ratio - 1.0) - abs(pred_acc_ratio - 1.0))
    row["hf_ratio_improvement_gyr"] = float(abs(input_gyr_ratio - 1.0) - abs(pred_gyr_ratio - 1.0))
    row["hf_ratio_improvement_mean"] = float(
        np.mean([row["hf_ratio_improvement_acc"], row["hf_ratio_improvement_gyr"]])
    )
    return row


def _aggregate_metric_frame(frame: pd.DataFrame, group_column: str) -> pd.DataFrame:
    metric_columns = [
        col
        for col in frame.columns
        if col.startswith("rmse_")
        or col.startswith("input_rmse_")
        or col.startswith("pearson_")
        or col.startswith("hf_ratio_")
        or col.startswith("psd_distance_")
        or col.endswith("_channel_rmse_mean")
        or col.endswith("_channel_rmse_improvement")
        or col.endswith("_norm_rmse")
    ]
    return (
        frame.groupby(group_column)[metric_columns]
        .mean(numeric_only=True)
        .reset_index()
        .sort_values(group_column)
    )


def summarize_window_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    inputs: np.ndarray,
    metadata: List[Dict[str, Any]],
    channels: List[str],
    sampling_frequency: float,
) -> Dict[str, Any]:
    """Aggregate window-level metrics overall and by motion / segment / experiment."""
    rows = [
        compute_window_metrics(predictions[idx], targets[idx], inputs[idx], metadata[idx], channels, sampling_frequency)
        for idx in range(len(metadata))
    ]
    frame = pd.DataFrame(rows)

    overall = {
        "num_windows": int(len(frame)),
        "rmse_mean": float(frame["rmse_mean"].mean()),
        "input_rmse_mean": float(frame["input_rmse_mean"].mean()),
        "acc_norm_rmse": float(frame["acc_norm_rmse"].mean()),
        "gyr_norm_rmse": float(frame["gyr_norm_rmse"].mean()),
        "input_acc_norm_rmse": float(frame["input_acc_norm_rmse"].mean()),
        "input_gyr_norm_rmse": float(frame["input_gyr_norm_rmse"].mean()),
        "acc_channel_rmse_mean": float(frame["acc_channel_rmse_mean"].mean()),
        "gyr_channel_rmse_mean": float(frame["gyr_channel_rmse_mean"].mean()),
        "input_acc_channel_rmse_mean": float(frame["input_acc_channel_rmse_mean"].mean()),
        "input_gyr_channel_rmse_mean": float(frame["input_gyr_channel_rmse_mean"].mean()),
        "acc_channel_rmse_improvement": float(frame["acc_channel_rmse_improvement"].mean()),
        "gyr_channel_rmse_improvement": float(frame["gyr_channel_rmse_improvement"].mean()),
        "pearson_mean": float(frame["pearson_mean"].mean()),
        "hf_ratio_improvement_mean": float(frame["hf_ratio_improvement_mean"].mean()),
        "psd_distance_mean": float(frame["psd_distance_mean"].mean()),
    }

    anomaly_mask = (
        frame["is_anomaly_case"].astype(bool)
        if "is_anomaly_case" in frame.columns
        else pd.Series(False, index=frame.index)
    )

    return {
        "overall": overall,
        "per_window": frame,
        "per_motion": _aggregate_metric_frame(frame, "motion_name"),
        "per_segment": _aggregate_metric_frame(frame, "segment_id"),
        "per_experiment": _aggregate_metric_frame(frame, "experiment_id"),
        "anomaly": frame.loc[anomaly_mask].copy(),
    }


def save_metric_summary(summary: Dict[str, Any], output_dir: Path) -> None:
    """Persist metric tables in the format requested by the user."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "overall_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary["overall"], handle, indent=2, sort_keys=True)
    summary["per_motion"].to_csv(output_dir / "per_motion_metrics.csv", index=False)
    summary["per_segment"].to_csv(output_dir / "per_segment_metrics.csv", index=False)
    summary["per_experiment"].to_csv(output_dir / "per_experiment_metrics.csv", index=False)
    if not summary["anomaly"].empty:
        summary["anomaly"].to_csv(output_dir / "anomaly_case_metrics.csv", index=False)
