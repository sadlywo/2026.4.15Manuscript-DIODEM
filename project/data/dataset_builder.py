from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from project.data.splits import assign_split_labels
from project.data.window_dataset import fit_normalization_stats, generate_window_start_indices
from project.utils.io import (
    ensure_dir,
    load_csv_table,
    load_yaml_config,
    read_imu_csv,
    save_json,
    save_pickle,
    select_channels,
    split_imu_segments,
)


DEFAULT_SEGMENTS = [f"seg{i}" for i in range(1, 6)]


def build_pair_table(
    metadata_df: pd.DataFrame,
    selected_df: pd.DataFrame | None = None,
    required_sampling_frequency: float = 40.0,
) -> pd.DataFrame:
    """Expand motion-level rigid/nonrigid pairs into segment-level pair records."""
    selected_df = selected_df if selected_df is not None else pd.DataFrame()
    anomaly_mask = pd.Series(False, index=selected_df.index)
    if not selected_df.empty and "selection_reason" in selected_df.columns:
        anomaly_mask = selected_df["selection_reason"].fillna("").str.contains("anomaly_case", case=False)
    anomaly_df = selected_df.loc[anomaly_mask].copy() if not selected_df.empty else pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    group_cols = ["kc_type", "experiment_id", "motion_folder"]
    for group_key, group in metadata_df.groupby(group_cols):
        rigid_rows = group[group["file_type"] == "imu_rigid"]
        nonrigid_rows = group[group["file_type"] == "imu_nonrigid"]
        if rigid_rows.empty or nonrigid_rows.empty:
            continue

        rigid_row = rigid_rows.iloc[0]
        nonrigid_row = nonrigid_rows.iloc[0]
        rigid_fs = float(rigid_row["sampling_frequency"])
        nonrigid_fs = float(nonrigid_row["sampling_frequency"])
        if abs(rigid_fs - required_sampling_frequency) > 1e-6:
            continue
        if abs(nonrigid_fs - required_sampling_frequency) > 1e-6:
            continue
        if abs(rigid_fs - nonrigid_fs) > 1e-6:
            continue

        if int(rigid_row["n_samples"]) != int(nonrigid_row["n_samples"]):
            continue

        kc_type, experiment_id, motion_folder = group_key
        motion_name = rigid_row["motion_name"]
        motion_index = rigid_row.get("motion_index", motion_folder.split("_")[0])
        for segment_id in DEFAULT_SEGMENTS:
            is_anomaly = False
            if not anomaly_df.empty:
                mask = (
                    (anomaly_df["kc_type"] == kc_type)
                    & (anomaly_df["experiment_id"] == experiment_id)
                    & (anomaly_df["motion_folder"] == motion_folder)
                    & (anomaly_df["segment"].str.lower() == segment_id)
                )
                is_anomaly = bool(mask.any())

            rows.append(
                {
                    "kc_type": kc_type,
                    "experiment_id": experiment_id,
                    "motion_folder": motion_folder,
                    "motion_index": motion_index,
                    "motion_name": motion_name,
                    "segment_id": segment_id,
                    "rigid_path": str(rigid_row["path"]),
                    "nonrigid_path": str(nonrigid_row["path"]),
                    "sampling_frequency": rigid_fs,
                    "n_samples": int(rigid_row["n_samples"]),
                    "is_anomaly_case": is_anomaly,
                }
            )

    return pd.DataFrame(rows)


def _resolve_project_paths(config: Dict[str, Any]) -> Dict[str, Path]:
    repo_root = Path(config["repo_root"])
    return {
        "repo_root": repo_root,
        "metadata_summary": (repo_root / config["metadata_summary"]).resolve(),
        "selected_examples": (repo_root / config["selected_examples"]).resolve(),
        "processed_root": ensure_dir((repo_root / config["processed_root"]).resolve()),
        "outputs_root": ensure_dir((repo_root / config["outputs_root"]).resolve()),
    }


def _load_or_cache_csv(csv_path: Path, cache: Dict[Path, Dict[str, Any]]) -> Dict[str, Any]:
    if csv_path not in cache:
        cache[csv_path] = read_imu_csv(csv_path)
    return cache[csv_path]


def _build_windows_for_pair(
    pair_row: pd.Series,
    channels: List[str],
    window_size: int,
    stride: int,
    csv_cache: Dict[Path, Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    rigid_path = Path(pair_row["rigid_path"])
    nonrigid_path = Path(pair_row["nonrigid_path"])
    rigid_loaded = _load_or_cache_csv(rigid_path, csv_cache)
    nonrigid_loaded = _load_or_cache_csv(nonrigid_path, csv_cache)

    if abs(rigid_loaded["sampling_frequency"] - nonrigid_loaded["sampling_frequency"]) > 1e-6:
        return np.empty((0, window_size, len(channels)), dtype=np.float32), np.empty(
            (0, window_size, len(channels)),
            dtype=np.float32,
        ), []

    rigid_segments = split_imu_segments(rigid_loaded["data"])
    nonrigid_segments = split_imu_segments(nonrigid_loaded["data"])
    segment_id = pair_row["segment_id"]
    if segment_id not in rigid_segments or segment_id not in nonrigid_segments:
        return np.empty((0, window_size, len(channels)), dtype=np.float32), np.empty(
            (0, window_size, len(channels)),
            dtype=np.float32,
        ), []

    rigid_array = select_channels(rigid_segments[segment_id], channels)
    nonrigid_array = select_channels(nonrigid_segments[segment_id], channels)
    if rigid_array.shape != nonrigid_array.shape:
        return np.empty((0, window_size, len(channels)), dtype=np.float32), np.empty(
            (0, window_size, len(channels)),
            dtype=np.float32,
        ), []

    starts = generate_window_start_indices(len(rigid_array), window_size, stride)
    inputs: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    metadata: List[Dict[str, Any]] = []
    for start_idx in starts:
        end_idx = start_idx + window_size
        x_window = nonrigid_array[start_idx:end_idx]
        y_window = rigid_array[start_idx:end_idx]
        if x_window.shape != (window_size, len(channels)):
            continue
        if y_window.shape != (window_size, len(channels)):
            continue
        inputs.append(x_window.astype(np.float32))
        targets.append(y_window.astype(np.float32))
        metadata.append(
            {
                "kc_type": pair_row["kc_type"],
                "experiment_id": pair_row["experiment_id"],
                "motion_folder": pair_row["motion_folder"],
                "motion_index": pair_row["motion_index"],
                "motion_name": pair_row["motion_name"],
                "segment_id": segment_id,
                "start_idx": int(start_idx),
                "end_idx": int(end_idx),
                "sampling_frequency": float(pair_row["sampling_frequency"]),
                "is_anomaly_case": bool(pair_row["is_anomaly_case"]),
            }
        )

    if not inputs:
        return np.empty((0, window_size, len(channels)), dtype=np.float32), np.empty(
            (0, window_size, len(channels)),
            dtype=np.float32,
        ), []
    return np.stack(inputs), np.stack(targets), metadata


def build_processed_splits(config: Dict[str, Any]) -> Dict[str, Path]:
    """Build train/val/test caches from the existing DIODEM preprocessing outputs."""
    paths = _resolve_project_paths(config)
    metadata_df = load_csv_table(paths["metadata_summary"])
    selected_df = load_csv_table(paths["selected_examples"]) if paths["selected_examples"].exists() else pd.DataFrame()
    pair_df = build_pair_table(metadata_df, selected_df)

    split_config = {
        "strategy": config["split_strategy"],
        "by_experiment": config.get("by_experiment", {}),
        "by_motion_type": config.get("by_motion_type", {}),
        "by_chain": config.get("by_chain", {}),
        "anomaly": config.get("anomaly", {}),
    }
    labeled_pairs = assign_split_labels(pair_df, split_config)
    labeled_pairs.to_csv(paths["processed_root"] / "pair_table.csv", index=False)

    channels = list(config["input_channels"])
    if channels != list(config["target_channels"]):
        raise ValueError("This baseline project expects input and target channels to match.")

    window_size = int(config["window_size"])
    stride = int(config["stride"])
    csv_cache: Dict[Path, Dict[str, Any]] = {}
    built_paths: Dict[str, Path] = {}
    split_bundles: Dict[str, Dict[str, Any]] = {}

    for split_name in ("train", "val", "test"):
        split_pairs = labeled_pairs[labeled_pairs["split"] == split_name].reset_index(drop=True)
        inputs_list: List[np.ndarray] = []
        targets_list: List[np.ndarray] = []
        metadata_list: List[Dict[str, Any]] = []

        for row in split_pairs.itertuples(index=False):
            x_windows, y_windows, window_metadata = _build_windows_for_pair(
                pd.Series(row._asdict()),
                channels,
                window_size,
                stride,
                csv_cache,
            )
            if len(window_metadata) == 0:
                continue
            inputs_list.append(x_windows)
            targets_list.append(y_windows)
            metadata_list.extend(window_metadata)

        if inputs_list:
            inputs = np.concatenate(inputs_list, axis=0).astype(np.float32)
            targets = np.concatenate(targets_list, axis=0).astype(np.float32)
        else:
            inputs = np.empty((0, window_size, len(channels)), dtype=np.float32)
            targets = np.empty((0, window_size, len(channels)), dtype=np.float32)

        bundle = {
            "inputs": inputs,
            "targets": targets,
            "metadata": metadata_list,
            "channels": channels,
        }
        split_bundles[split_name] = bundle
        bundle_path = paths["processed_root"] / f"{split_name}_samples.pkl"
        save_pickle(bundle, bundle_path)
        built_paths[split_name] = bundle_path

    normalization_mode = config.get("normalization", "none")
    stats = fit_normalization_stats(split_bundles["train"], normalization=normalization_mode)
    serializable_stats = {}
    for key, value in stats.items():
        if isinstance(value, np.ndarray):
            serializable_stats[key] = value.tolist()
        else:
            serializable_stats[key] = value
    save_json(serializable_stats, paths["processed_root"] / "normalization_stats.json")
    return built_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Build cached DIODEM supervised-learning windows.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("project/configs/default.yaml"),
        help="Path to the YAML config file.",
    )
    args = parser.parse_args()
    config = load_yaml_config(args.config.resolve())
    build_processed_splits(config)
    print(f"Processed sample caches saved under {config['processed_root']}")


if __name__ == "__main__":
    main()
