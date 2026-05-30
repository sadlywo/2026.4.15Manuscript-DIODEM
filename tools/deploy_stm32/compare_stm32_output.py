from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Sequence

import numpy as np

from deploy_common import DEFAULT_OUTPUT_ROOT, PREDICTION_CHANNELS, repo_root_from_file, save_json


def load_prediction_csv(path: Path, columns: Sequence[str] = PREDICTION_CHANNELS) -> np.ndarray:
    rows = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in columns if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"CSV {path} is missing prediction columns: {missing}")
        for row in reader:
            rows.append([float(row[column]) for column in columns])
    return np.asarray(rows, dtype=np.float32)


def compute_error_metrics(
    reference: np.ndarray,
    actual: np.ndarray,
    channel_names: Sequence[str] = PREDICTION_CHANNELS,
) -> Dict[str, object]:
    reference = np.asarray(reference, dtype=np.float32)
    actual = np.asarray(actual, dtype=np.float32)
    if reference.shape != actual.shape:
        raise ValueError(f"Shape mismatch: reference {reference.shape}, actual {actual.shape}")
    if reference.ndim != 2:
        raise ValueError(f"Expected `[N, C]` arrays, got {reference.shape}")
    delta = actual - reference
    per_channel = {}
    for index, name in enumerate(channel_names):
        if index >= delta.shape[1]:
            break
        channel_delta = delta[:, index]
        per_channel[str(name)] = {
            "max_abs_error": float(np.max(np.abs(channel_delta))) if channel_delta.size else 0.0,
            "mean_abs_error": float(np.mean(np.abs(channel_delta))) if channel_delta.size else 0.0,
            "rmse": float(np.sqrt(np.mean(np.square(channel_delta)))) if channel_delta.size else 0.0,
            "bias": float(np.mean(channel_delta)) if channel_delta.size else 0.0,
        }
    return {
        "num_samples": int(reference.shape[0]),
        "num_channels": int(reference.shape[1]),
        "max_abs_error": float(np.max(np.abs(delta))) if delta.size else 0.0,
        "mean_abs_error": float(np.mean(np.abs(delta))) if delta.size else 0.0,
        "rmse": float(np.sqrt(np.mean(np.square(delta)))) if delta.size else 0.0,
        "per_channel": per_channel,
    }


def compare_csv(reference_csv: Path, stm32_csv: Path, output: Path | None = None) -> Dict[str, object]:
    reference = load_prediction_csv(reference_csv)
    actual = load_prediction_csv(stm32_csv)
    metrics = compute_error_metrics(reference, actual)
    if output is not None:
        save_json(metrics, output)
    return metrics


def main() -> None:
    repo_root = repo_root_from_file()
    parser = argparse.ArgumentParser(description="Compare STM32 compensated output against PC FP32 references.")
    parser.add_argument("--reference-csv", type=Path, required=True)
    parser.add_argument("--stm32-csv", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / DEFAULT_OUTPUT_ROOT / "stm32_vs_pc_metrics.json",
    )
    args = parser.parse_args()
    metrics = compare_csv(args.reference_csv, args.stm32_csv, args.output)
    print(metrics)
    print(f"Saved comparison metrics to {args.output.resolve()}")


if __name__ == "__main__":
    main()
