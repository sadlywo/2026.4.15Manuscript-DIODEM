from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


CHANNELS = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]
PREDICTION_COLUMNS = ["pred_acc_x", "pred_acc_y", "pred_acc_z", "pred_gyr_x", "pred_gyr_y", "pred_gyr_z"]


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_matrix(rows: List[Dict[str, str]], columns: List[str]) -> np.ndarray:
    missing = [column for column in columns if rows and column not in rows[0]]
    if missing:
        raise KeyError(f"CSV rows are missing required columns: {missing}")
    matrix = []
    for row in rows:
        matrix.append([float(row[column]) for column in columns])
    return np.asarray(matrix, dtype=np.float32)


def _compute_rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((prediction - target) ** 2)))


def _compute_pearson(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=float)
    target = np.asarray(target, dtype=float)
    if prediction.std() < 1e-12 or target.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(prediction, target)[0, 1])


def _periodogram_power(signal: np.ndarray, sampling_frequency: float) -> tuple[np.ndarray, np.ndarray]:
    signal = np.asarray(signal, dtype=float)
    if signal.size < 4:
        return np.asarray([]), np.asarray([])
    centered = signal - signal.mean()
    spectrum = np.fft.rfft(centered)
    freqs = np.fft.rfftfreq(signal.size, d=1.0 / sampling_frequency)
    power = (np.abs(spectrum) ** 2) / max(signal.size, 1)
    return freqs, power


def _compute_psd_distance(prediction: np.ndarray, target: np.ndarray, sampling_frequency: float) -> float:
    freq_pred, power_pred = _periodogram_power(prediction, sampling_frequency)
    freq_target, power_target = _periodogram_power(target, sampling_frequency)
    if power_pred.size == 0 or power_target.size == 0 or power_pred.shape != power_target.shape:
        return float("nan")
    return float(np.sqrt(np.mean((power_pred - power_target) ** 2)))


def _high_band_power(signal: np.ndarray, sampling_frequency: float) -> float:
    freqs, power = _periodogram_power(signal, sampling_frequency)
    if power.size == 0:
        return float("nan")
    upper = min(15.0, sampling_frequency / 2.0)
    mask = (freqs >= 5.0) & (freqs <= upper)
    if not np.any(mask):
        return float("nan")
    return float(np.trapz(power[mask], freqs[mask]))


def _compute_norm(values: np.ndarray, start_index: int, end_index: int) -> np.ndarray:
    return np.linalg.norm(values[:, start_index:end_index], axis=1)


def _percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q))


def _fmt(value: float, decimals: int = 4) -> str:
    if value is None or math.isnan(value):
        return ""
    return f"{float(value):.{decimals}f}"


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown_table(path: Path, row: Dict[str, object]) -> None:
    headers = list(row.keys())
    values = [str(row[key]) for key in headers]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        "| " + " | ".join(values) + " |",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize one replay run into a PPT-friendly scientific table.")
    parser.add_argument("--input-csv", type=Path, required=True, help="Replay-ready nonrigid 6-channel CSV.")
    parser.add_argument("--reference-csv", type=Path, required=True, help="Replay-ready rigid 6-channel CSV.")
    parser.add_argument("--prediction-csv", type=Path, required=True, help="Model output CSV from main_realtime_infer.py.")
    parser.add_argument("--sampling-frequency", type=float, default=40.0)
    parser.add_argument("--label", type=str, default="TCN-causal replay demo")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/replay_demo"),
        help="Directory where PPT-ready outputs will be saved.",
    )
    args = parser.parse_args()

    input_rows = _read_csv_rows(args.input_csv.resolve())
    reference_rows = _read_csv_rows(args.reference_csv.resolve())
    prediction_rows = _read_csv_rows(args.prediction_csv.resolve())
    if not input_rows or not reference_rows or not prediction_rows:
        raise ValueError("Input, reference, and prediction CSV files must all contain data rows.")

    input_values = _load_matrix(input_rows, CHANNELS)
    target_values = _load_matrix(reference_rows, CHANNELS)
    prediction_values = _load_matrix(prediction_rows, PREDICTION_COLUMNS)
    latencies = _load_matrix(prediction_rows, ["latency_ms"]).reshape(-1)

    length = min(len(input_values), len(target_values), len(prediction_values), len(latencies))
    input_values = input_values[:length]
    target_values = target_values[:length]
    prediction_values = prediction_values[:length]
    latencies = latencies[:length]

    per_channel_rmse = []
    per_channel_pearson = []
    per_channel_psd = []
    per_channel_input_rmse = []
    for index in range(len(CHANNELS)):
        per_channel_rmse.append(_compute_rmse(prediction_values[:, index], target_values[:, index]))
        per_channel_input_rmse.append(_compute_rmse(input_values[:, index], target_values[:, index]))
        per_channel_pearson.append(_compute_pearson(prediction_values[:, index], target_values[:, index]))
        per_channel_psd.append(_compute_psd_distance(prediction_values[:, index], target_values[:, index], args.sampling_frequency))

    pred_acc_norm = _compute_norm(prediction_values, 0, 3)
    target_acc_norm = _compute_norm(target_values, 0, 3)
    pred_gyr_norm = _compute_norm(prediction_values, 3, 6)
    target_gyr_norm = _compute_norm(target_values, 3, 6)
    input_acc_norm = _compute_norm(input_values, 0, 3)
    input_gyr_norm = _compute_norm(input_values, 3, 6)

    input_acc_ratio = (_high_band_power(input_acc_norm, args.sampling_frequency) + 1e-8) / (
        _high_band_power(target_acc_norm, args.sampling_frequency) + 1e-8
    )
    pred_acc_ratio = (_high_band_power(pred_acc_norm, args.sampling_frequency) + 1e-8) / (
        _high_band_power(target_acc_norm, args.sampling_frequency) + 1e-8
    )
    input_gyr_ratio = (_high_band_power(input_gyr_norm, args.sampling_frequency) + 1e-8) / (
        _high_band_power(target_gyr_norm, args.sampling_frequency) + 1e-8
    )
    pred_gyr_ratio = (_high_band_power(pred_gyr_norm, args.sampling_frequency) + 1e-8) / (
        _high_band_power(target_gyr_norm, args.sampling_frequency) + 1e-8
    )

    rmse_mean = float(np.mean(per_channel_rmse))
    input_rmse_mean = float(np.mean(per_channel_input_rmse))
    acc_rmse_mean = float(np.mean(per_channel_rmse[:3]))
    gyr_rmse_mean = float(np.mean(per_channel_rmse[3:]))
    input_acc_rmse_mean = float(np.mean(per_channel_input_rmse[:3]))
    input_gyr_rmse_mean = float(np.mean(per_channel_input_rmse[3:]))
    rmse_reduction_pct = 100.0 * (input_rmse_mean - rmse_mean) / max(input_rmse_mean, 1e-8)
    acc_reduction_pct = 100.0 * (input_acc_rmse_mean - acc_rmse_mean) / max(input_acc_rmse_mean, 1e-8)
    gyr_reduction_pct = 100.0 * (input_gyr_rmse_mean - gyr_rmse_mean) / max(input_gyr_rmse_mean, 1e-8)
    hf_improvement = float(
        np.mean(
            [
                abs(input_acc_ratio - 1.0) - abs(pred_acc_ratio - 1.0),
                abs(input_gyr_ratio - 1.0) - abs(pred_gyr_ratio - 1.0),
            ]
        )
    )

    summary_row = {
        "Experiment": args.label,
        "Samples": int(length),
        "Sampling Hz": _fmt(float(args.sampling_frequency), decimals=1),
        "Input RMSE": _fmt(input_rmse_mean),
        "Compensated RMSE": _fmt(rmse_mean),
        "RMSE Reduction (%)": _fmt(rmse_reduction_pct, decimals=2),
        "Acc RMSE": _fmt(acc_rmse_mean),
        "Acc Reduction (%)": _fmt(acc_reduction_pct, decimals=2),
        "Gyr RMSE": _fmt(gyr_rmse_mean),
        "Gyr Reduction (%)": _fmt(gyr_reduction_pct, decimals=2),
        "Pearson": _fmt(float(np.nanmean(np.asarray(per_channel_pearson, dtype=float)))),
        "PSD Distance": _fmt(float(np.nanmean(np.asarray(per_channel_psd, dtype=float)))),
        "HF Improve.": _fmt(hf_improvement),
        "Latency mean (ms)": _fmt(float(np.mean(latencies)), decimals=3),
        "Latency p95 (ms)": _fmt(_percentile(latencies, 95), decimals=3),
    }

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "ppt_replay_summary_table.csv"
    md_path = output_dir / "ppt_replay_summary_table.md"
    json_path = output_dir / "ppt_replay_summary_table.json"

    _write_csv(csv_path, [summary_row])
    _write_markdown_table(md_path, summary_row)
    json_path.write_text(json.dumps({"rows": [summary_row]}, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary_row, indent=2, ensure_ascii=False))
    print(f"Saved PPT replay summary table to {csv_path}")
    print(f"Saved PPT markdown table to {md_path}")


if __name__ == "__main__":
    main()
