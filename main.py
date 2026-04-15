from __future__ import annotations

import math
import re
import textwrap
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from scipy.signal import welch
from scipy.spatial.transform import Rotation


# Replace this with your local dataset path if needed.
DATASET_ROOT = Path(r"e:\VSCode_Study\2026.4.15Manuscript-DIODEM\dataset")
OUTPUT_ROOT = Path(r"e:\VSCode_Study\2026.4.15Manuscript-DIODEM\outputs")
FIGURES_DIR = OUTPUT_ROOT / "figures"
TABLES_DIR = OUTPUT_ROOT / "tables"

MOTION_KEYWORDS = [
    "pause",
    "slow",
    "fast",
    "shaking",
    "dangle",
    "global",
    "pickandplace",
    "gait",
    "quasistatic",
]

IMU_SIGNAL_PATTERN = re.compile(r"^(seg\d+)_(acc|gyr|mag)_(x|y|z)$", re.IGNORECASE)
OMC_MARKER_PATTERN = re.compile(
    r"^(seg\d+)_marker(\d+)_(x|y|z)$",
    re.IGNORECASE,
)
OMC_QUAT_PATTERN = re.compile(r"^(seg\d+)_quat_([wxyz])$", re.IGNORECASE)


@dataclass
class LoadedCsv:
    data: pd.DataFrame
    sampling_frequency: float
    time_vector: np.ndarray
    warnings: List[str]


def print_stage(message: str) -> None:
    print(f"[DIODEM] {message}")


def sanitize_motion_label(label: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", str(label).strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "unknown"


def infer_file_type_from_name(file_name: str) -> str:
    lower_name = file_name.lower()
    if "imu_nonrigid" in lower_name:
        return "imu_nonrigid"
    if "imu_rigid" in lower_name:
        return "imu_rigid"
    if lower_name.endswith("_omc.csv"):
        return "omc"
    return "unknown"


def parse_motion_folder_name(folder_name: str) -> Dict[str, str]:
    match = re.match(r"^(motion\d+)(?:_(.+))?$", folder_name, re.IGNORECASE)
    if not match:
        return {
            "motion_index": folder_name,
            "motion_name": sanitize_motion_label(folder_name),
        }
    motion_index = match.group(1).lower()
    motion_name = sanitize_motion_label(match.group(2) or motion_index)
    return {"motion_index": motion_index, "motion_name": motion_name}


def segment_sort_key(segment_name: str) -> int:
    match = re.search(r"(\d+)", segment_name)
    return int(match.group(1)) if match else 10**9


def ensure_output_dirs() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def add_warning(warnings_list: List[str], message: str) -> None:
    warnings_list.append(message)
    print(f"[WARN] {message}")


def extract_sampling_frequency(csv_path: Path) -> float:
    with csv_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for _ in range(10):
            line = handle.readline()
            if not line:
                break
            match = re.search(
                r"sampling\s+frequency\s*:\s*([0-9]*\.?[0-9]+)",
                line,
                re.IGNORECASE,
            )
            if match:
                return float(match.group(1))
    raise ValueError(f"Could not parse sampling frequency from {csv_path}")


def read_dataset_csv(csv_path: Path, warnings_list: Optional[List[str]] = None) -> LoadedCsv:
    local_warnings: List[str] = []
    sink = warnings_list if warnings_list is not None else local_warnings
    sampling_frequency = extract_sampling_frequency(csv_path)

    try:
        df = pd.read_csv(csv_path, comment="#")
    except Exception as exc:
        raise RuntimeError(f"Failed to read CSV {csv_path}: {exc}") from exc

    original_columns = list(df.columns)
    kept_columns: List[str] = []
    for column in original_columns:
        normalized = str(column).strip()
        if not normalized or normalized.lower().startswith("unnamed"):
            add_warning(
                sink,
                f"{csv_path.name}: dropping blank/unnamed column {column!r}.",
            )
            continue
        kept_columns.append(normalized)
    df = df.loc[:, [col for col in original_columns if str(col).strip() in kept_columns]].copy()
    df.columns = kept_columns

    for column in df.columns:
        numeric_series = pd.to_numeric(df[column], errors="coerce")
        introduced_nan = int(numeric_series.isna().sum() - df[column].isna().sum())
        if introduced_nan > 0:
            add_warning(
                sink,
                f"{csv_path.name}: column {column} produced {introduced_nan} NaNs during numeric conversion.",
            )
        df[column] = numeric_series.astype(float)

    n_samples = len(df)
    if sampling_frequency <= 0:
        add_warning(sink, f"{csv_path.name}: non-positive sampling frequency {sampling_frequency}.")
        time_vector = np.arange(n_samples, dtype=float)
    else:
        time_vector = np.arange(n_samples, dtype=float) / sampling_frequency

    return LoadedCsv(
        data=df,
        sampling_frequency=sampling_frequency,
        time_vector=time_vector,
        warnings=local_warnings,
    )


def standardize_signal_groups(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    groups: Dict[str, Dict[str, pd.Series]] = {}
    for column in df.columns:
        match = IMU_SIGNAL_PATTERN.match(column)
        if not match:
            continue
        segment, modality, axis = match.groups()
        groups.setdefault(segment.lower(), {})
        groups[segment.lower()][f"{modality.lower()}_{axis.lower()}"] = df[column].astype(float)

    standardized: Dict[str, pd.DataFrame] = {}
    for segment in sorted(groups, key=segment_sort_key):
        signal_map = groups[segment]
        segment_df = pd.DataFrame(index=df.index)
        for modality in ("acc", "gyr", "mag"):
            axes = []
            for axis in ("x", "y", "z"):
                key = f"{modality}_{axis}"
                segment_df[key] = signal_map.get(
                    key,
                    pd.Series(np.nan, index=df.index, dtype=float),
                )
                axes.append(segment_df[key].to_numpy(dtype=float))
            stacked = np.column_stack(axes)
            segment_df[f"{modality}_norm"] = np.linalg.norm(stacked, axis=1)
        standardized[segment] = segment_df
    return standardized


def standardize_omc_groups(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    groups: Dict[str, Dict[str, pd.Series]] = {}
    for column in df.columns:
        marker_match = OMC_MARKER_PATTERN.match(column)
        quat_match = OMC_QUAT_PATTERN.match(column)
        if marker_match:
            segment, marker_idx, axis = marker_match.groups()
            groups.setdefault(segment.lower(), {})
            groups[segment.lower()][f"marker{marker_idx}_{axis.lower()}"] = df[column].astype(float)
        elif quat_match:
            segment, quat_axis = quat_match.groups()
            groups.setdefault(segment.lower(), {})
            groups[segment.lower()][f"quat_{quat_axis.lower()}"] = df[column].astype(float)

    standardized: Dict[str, pd.DataFrame] = {}
    for segment in sorted(groups, key=segment_sort_key):
        standardized[segment] = pd.DataFrame(groups[segment], index=df.index)
    return standardized


def scan_dataset(dataset_root: Path) -> Tuple[pd.DataFrame, Dict[Path, LoadedCsv], List[str]]:
    warnings_list: List[str] = []
    file_cache: Dict[Path, LoadedCsv] = {}
    rows: List[Dict[str, object]] = []

    for csv_path in sorted(dataset_root.rglob("*.csv")):
        try:
            relative = csv_path.relative_to(dataset_root)
        except ValueError:
            add_warning(warnings_list, f"Skipping non-relative path {csv_path}.")
            continue

        if len(relative.parts) < 4:
            add_warning(warnings_list, f"Skipping unexpected path layout: {relative}.")
            continue

        kc_type, experiment_id, motion_folder = relative.parts[:3]
        file_type = infer_file_type_from_name(csv_path.name)
        motion_info = parse_motion_folder_name(motion_folder)

        try:
            loaded = read_dataset_csv(csv_path, warnings_list)
        except Exception as exc:
            add_warning(warnings_list, str(exc))
            continue

        file_cache[csv_path] = loaded
        n_samples = len(loaded.data)
        duration_sec = (
            n_samples / loaded.sampling_frequency if loaded.sampling_frequency > 0 else np.nan
        )
        rows.append(
            {
                "kc_type": kc_type.lower(),
                "experiment_id": experiment_id.lower(),
                "motion_folder": motion_folder,
                "motion_index": motion_info["motion_index"],
                "motion_name": motion_info["motion_name"],
                "file_type": file_type,
                "path": str(csv_path),
                "sampling_frequency": loaded.sampling_frequency,
                "n_samples": n_samples,
                "duration_sec": duration_sec,
            }
        )

    metadata_df = pd.DataFrame(rows)
    if not metadata_df.empty:
        metadata_df = metadata_df.sort_values(
            ["kc_type", "experiment_id", "motion_index", "file_type"]
        ).reset_index(drop=True)
    return metadata_df, file_cache, warnings_list


def build_motion_level_summary(metadata_df: pd.DataFrame) -> pd.DataFrame:
    if metadata_df.empty:
        return pd.DataFrame()
    preferred = metadata_df[metadata_df["file_type"] == "imu_rigid"].copy()
    if preferred.empty:
        preferred = metadata_df.copy()
    motion_df = (
        preferred.sort_values(["kc_type", "experiment_id", "motion_folder"])
        .drop_duplicates(["kc_type", "experiment_id", "motion_folder"])
        .copy()
    )
    return motion_df


def print_dataset_summary(metadata_df: pd.DataFrame) -> None:
    if metadata_df.empty:
        print_stage("No valid CSV files discovered.")
        return

    motion_df = build_motion_level_summary(metadata_df)
    total_experiments = metadata_df["experiment_id"].nunique()
    total_motions = motion_df[["kc_type", "experiment_id", "motion_folder"]].drop_duplicates().shape[0]
    print_stage(f"Total experiments: {total_experiments}")
    print_stage(f"Total motions: {total_motions}")

    file_counts = metadata_df["file_type"].value_counts().to_dict()
    for file_type in ("imu_rigid", "imu_nonrigid", "omc", "unknown"):
        if file_type in file_counts:
            print_stage(f"{file_type} files: {file_counts[file_type]}")

    print_stage("Motion-level sample count and duration summary:")
    motion_summary = (
        motion_df.groupby("motion_name")
        .agg(motion_count=("motion_folder", "count"), total_duration_sec=("duration_sec", "sum"))
        .sort_values(["motion_count", "total_duration_sec"], ascending=False)
    )
    for motion_name, row in motion_summary.iterrows():
        print(
            f"  - {motion_name:20s} count={int(row['motion_count']):3d} "
            f"duration={row['total_duration_sec']:.1f}s"
        )

    print_stage("arm vs gait summary:")
    kc_summary = (
        motion_df.groupby("kc_type")
        .agg(motion_count=("motion_folder", "count"), total_duration_sec=("duration_sec", "sum"))
        .sort_values("motion_count", ascending=False)
    )
    for kc_type, row in kc_summary.iterrows():
        print(
            f"  - {kc_type:6s} motions={int(row['motion_count']):3d} "
            f"duration={row['total_duration_sec']:.1f}s"
        )


def local_linear_alignment(
    rigid_df: pd.DataFrame,
    nonrigid_df: pd.DataFrame,
    time_rigid: np.ndarray,
    time_nonrigid: np.ndarray,
    rigid_fs: float,
    nonrigid_fs: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, float, bool]:
    if len(rigid_df) == 0 or len(nonrigid_df) == 0:
        return rigid_df.copy(), nonrigid_df.copy(), np.array([], dtype=float), min(rigid_fs, nonrigid_fs), False

    same_length = len(rigid_df) == len(nonrigid_df)
    same_time = same_length and np.allclose(time_rigid, time_nonrigid, atol=1e-9, rtol=1e-6)
    if same_time:
        return rigid_df.copy(), nonrigid_df.copy(), time_rigid.copy(), rigid_fs, False

    common_start = max(float(time_rigid[0]), float(time_nonrigid[0]))
    common_end = min(float(time_rigid[-1]), float(time_nonrigid[-1]))
    if common_end <= common_start:
        min_len = min(len(rigid_df), len(nonrigid_df))
        return (
            rigid_df.iloc[:min_len].reset_index(drop=True),
            nonrigid_df.iloc[:min_len].reset_index(drop=True),
            time_rigid[:min_len].copy(),
            min(rigid_fs, nonrigid_fs),
            False,
        )

    target_fs = min(rigid_fs, nonrigid_fs)
    n_samples = max(int(math.floor((common_end - common_start) * target_fs)) + 1, 2)
    common_time = np.linspace(common_start, common_end, n_samples)
    rigid_aligned = interpolate_frame(rigid_df, time_rigid, common_time)
    nonrigid_aligned = interpolate_frame(nonrigid_df, time_nonrigid, common_time)
    return rigid_aligned, nonrigid_aligned, common_time, target_fs, True


def interpolate_frame(df: pd.DataFrame, source_time: np.ndarray, target_time: np.ndarray) -> pd.DataFrame:
    result = pd.DataFrame(index=np.arange(len(target_time)))
    for column in df.columns:
        values = df[column].to_numpy(dtype=float)
        finite_mask = np.isfinite(values)
        if finite_mask.sum() < 2:
            result[column] = np.full(len(target_time), np.nan, dtype=float)
            continue
        result[column] = np.interp(
            target_time,
            source_time[finite_mask],
            values[finite_mask],
        )
    return result


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    finite_mask = np.isfinite(x) & np.isfinite(y)
    if finite_mask.sum() < 3:
        return np.nan
    x_valid = x[finite_mask]
    y_valid = y[finite_mask]
    if np.std(x_valid) < 1e-12 or np.std(y_valid) < 1e-12:
        return np.nan
    return float(np.corrcoef(x_valid, y_valid)[0, 1])


def compute_band_power(freqs: np.ndarray, spectrum: np.ndarray, low: float, high: float) -> float:
    band_mask = (freqs >= low) & (freqs < high)
    if not np.any(band_mask):
        return 0.0
    return float(np.trapz(spectrum[band_mask], freqs[band_mask]))


def compute_signal_metrics(rigid_signal: np.ndarray, nonrigid_signal: np.ndarray, fs: float) -> Dict[str, float]:
    rigid_signal = np.asarray(rigid_signal, dtype=float)
    nonrigid_signal = np.asarray(nonrigid_signal, dtype=float)
    finite_mask = np.isfinite(rigid_signal) & np.isfinite(nonrigid_signal)
    rigid_valid = rigid_signal[finite_mask]
    nonrigid_valid = nonrigid_signal[finite_mask]

    if len(rigid_valid) < 4 or fs <= 0:
        return {
            "rmse": np.nan,
            "pearson_r": np.nan,
            "mean_diff": np.nan,
            "std_ratio": np.nan,
            "rigid_total_power": np.nan,
            "nonrigid_total_power": np.nan,
            "rigid_low_power": np.nan,
            "rigid_mid_power": np.nan,
            "rigid_high_power": np.nan,
            "nonrigid_low_power": np.nan,
            "nonrigid_mid_power": np.nan,
            "nonrigid_high_power": np.nan,
            "high_band_energy_ratio": np.nan,
        }

    residual = nonrigid_valid - rigid_valid
    rmse = float(np.sqrt(np.mean(residual**2)))
    mean_diff = float(np.mean(residual))
    rigid_std = float(np.std(rigid_valid))
    nonrigid_std = float(np.std(nonrigid_valid))
    std_ratio = float(nonrigid_std / rigid_std) if rigid_std > 1e-12 else np.nan
    pearson_r = safe_pearson(rigid_valid, nonrigid_valid)

    nperseg = min(1024, len(rigid_valid))
    freqs_rigid, psd_rigid = welch(rigid_valid, fs=fs, nperseg=nperseg)
    freqs_nonrigid, psd_nonrigid = welch(nonrigid_valid, fs=fs, nperseg=nperseg)

    rigid_total_power = compute_band_power(freqs_rigid, psd_rigid, 0.0, fs / 2.0 + 1e-6)
    nonrigid_total_power = compute_band_power(freqs_nonrigid, psd_nonrigid, 0.0, fs / 2.0 + 1e-6)
    rigid_low_power = compute_band_power(freqs_rigid, psd_rigid, 0.0, 2.0)
    rigid_mid_power = compute_band_power(freqs_rigid, psd_rigid, 2.0, 5.0)
    rigid_high_power = compute_band_power(freqs_rigid, psd_rigid, 5.0, min(15.0, fs / 2.0 + 1e-6))
    nonrigid_low_power = compute_band_power(freqs_nonrigid, psd_nonrigid, 0.0, 2.0)
    nonrigid_mid_power = compute_band_power(freqs_nonrigid, psd_nonrigid, 2.0, 5.0)
    nonrigid_high_power = compute_band_power(
        freqs_nonrigid,
        psd_nonrigid,
        5.0,
        min(15.0, fs / 2.0 + 1e-6),
    )
    high_ratio = float((nonrigid_high_power + 1e-12) / (rigid_high_power + 1e-12))

    return {
        "rmse": rmse,
        "pearson_r": pearson_r,
        "mean_diff": mean_diff,
        "std_ratio": std_ratio,
        "rigid_total_power": rigid_total_power,
        "nonrigid_total_power": nonrigid_total_power,
        "rigid_low_power": rigid_low_power,
        "rigid_mid_power": rigid_mid_power,
        "rigid_high_power": rigid_high_power,
        "nonrigid_low_power": nonrigid_low_power,
        "nonrigid_mid_power": nonrigid_mid_power,
        "nonrigid_high_power": nonrigid_high_power,
        "high_band_energy_ratio": high_ratio,
    }


def compute_pairwise_metrics(
    metadata_df: pd.DataFrame,
    file_cache: Dict[Path, LoadedCsv],
    warnings_list: List[str],
) -> pd.DataFrame:
    pair_rows: List[Dict[str, object]] = []
    if metadata_df.empty:
        return pd.DataFrame()

    grouped = metadata_df.groupby(["kc_type", "experiment_id", "motion_folder"])
    for (kc_type, experiment_id, motion_folder), group in grouped:
        file_map = {row.file_type: Path(row.path) for row in group.itertuples()}
        if "imu_rigid" not in file_map or "imu_nonrigid" not in file_map:
            add_warning(
                warnings_list,
                f"Skipping {kc_type}/{experiment_id}/{motion_folder}: missing rigid or nonrigid IMU file.",
            )
            continue

        rigid_loaded = file_cache[file_map["imu_rigid"]]
        nonrigid_loaded = file_cache[file_map["imu_nonrigid"]]
        rigid_groups = standardize_signal_groups(rigid_loaded.data)
        nonrigid_groups = standardize_signal_groups(nonrigid_loaded.data)
        common_segments = sorted(set(rigid_groups) & set(nonrigid_groups), key=segment_sort_key)
        motion_name = group["motion_name"].iloc[0]
        motion_index = group["motion_index"].iloc[0]

        for segment in common_segments:
            rigid_segment = rigid_groups[segment]
            nonrigid_segment = nonrigid_groups[segment]
            aligned_rigid, aligned_nonrigid, _, aligned_fs, used_resampling = local_linear_alignment(
                rigid_segment,
                nonrigid_segment,
                rigid_loaded.time_vector,
                nonrigid_loaded.time_vector,
                rigid_loaded.sampling_frequency,
                nonrigid_loaded.sampling_frequency,
            )

            for modality in ("acc", "gyr", "mag"):
                metric_row: Dict[str, object] = {
                    "kc_type": kc_type,
                    "experiment_id": experiment_id,
                    "motion_folder": motion_folder,
                    "motion_index": motion_index,
                    "motion_name": motion_name,
                    "segment": segment,
                    "modality": modality,
                    "rigid_path": str(file_map["imu_rigid"]),
                    "nonrigid_path": str(file_map["imu_nonrigid"]),
                    "aligned_sampling_frequency": aligned_fs,
                    "aligned_n_samples": len(aligned_rigid),
                    "used_local_resampling": used_resampling,
                }
                for axis in ("x", "y", "z"):
                    metrics = compute_signal_metrics(
                        aligned_rigid[f"{modality}_{axis}"].to_numpy(dtype=float),
                        aligned_nonrigid[f"{modality}_{axis}"].to_numpy(dtype=float),
                        aligned_fs,
                    )
                    for metric_name, metric_value in metrics.items():
                        metric_row[f"{metric_name}_{axis}"] = metric_value

                norm_metrics = compute_signal_metrics(
                    aligned_rigid[f"{modality}_norm"].to_numpy(dtype=float),
                    aligned_nonrigid[f"{modality}_norm"].to_numpy(dtype=float),
                    aligned_fs,
                )
                for metric_name, metric_value in norm_metrics.items():
                    metric_row[f"{metric_name}_norm"] = metric_value

                axis_rmse = [
                    metric_row.get("rmse_x", np.nan),
                    metric_row.get("rmse_y", np.nan),
                    metric_row.get("rmse_z", np.nan),
                ]
                metric_row["rmse_mean_axes"] = float(np.nanmean(axis_rmse))
                pair_rows.append(metric_row)

    pairwise_df = pd.DataFrame(pair_rows)
    if not pairwise_df.empty:
        pairwise_df = pairwise_df.sort_values(
            ["kc_type", "experiment_id", "motion_index", "segment", "modality"]
        ).reset_index(drop=True)
    return pairwise_df


def rank_percentile(series: pd.Series, ascending: bool = True) -> pd.Series:
    ranked = series.rank(method="average", na_option="keep", ascending=ascending)
    denominator = max(ranked.notna().sum(), 1)
    return ranked / denominator


def build_segment_summary(pairwise_df: pd.DataFrame) -> pd.DataFrame:
    if pairwise_df.empty:
        return pd.DataFrame()

    focus = pairwise_df[pairwise_df["modality"].isin(["acc", "gyr"])].copy()
    if focus.empty:
        return pd.DataFrame()

    index_cols = ["kc_type", "experiment_id", "motion_folder", "motion_index", "motion_name", "segment"]
    metric_cols = [
        "rmse_norm",
        "pearson_r_norm",
        "high_band_energy_ratio_norm",
        "mean_diff_norm",
        "std_ratio_norm",
    ]
    focus = focus[index_cols + ["modality"] + metric_cols]
    pivot = focus.pivot_table(index=index_cols, columns="modality", values=metric_cols, aggfunc="first")
    pivot.columns = [f"{modality}_{metric}" for metric, modality in pivot.columns]
    segment_df = pivot.reset_index()

    for col in [
        "acc_rmse_norm",
        "gyr_rmse_norm",
        "acc_high_band_energy_ratio_norm",
        "gyr_high_band_energy_ratio_norm",
        "acc_pearson_r_norm",
        "gyr_pearson_r_norm",
    ]:
        if col not in segment_df.columns:
            segment_df[col] = np.nan

    score_components = pd.DataFrame(
        {
            "acc_rmse_score": rank_percentile(segment_df["acc_rmse_norm"], ascending=True),
            "gyr_rmse_score": rank_percentile(segment_df["gyr_rmse_norm"], ascending=True),
            "acc_high_score": rank_percentile(
                segment_df["acc_high_band_energy_ratio_norm"],
                ascending=True,
            ),
            "gyr_high_score": rank_percentile(
                segment_df["gyr_high_band_energy_ratio_norm"],
                ascending=True,
            ),
            "acc_corr_score": 1.0 - rank_percentile(segment_df["acc_pearson_r_norm"], ascending=True),
            "gyr_corr_score": 1.0 - rank_percentile(segment_df["gyr_pearson_r_norm"], ascending=True),
        }
    )
    segment_df["combined_difference_score"] = score_components.mean(axis=1, skipna=True)
    segment_df["combined_rmse"] = segment_df[
        ["acc_rmse_norm", "gyr_rmse_norm"]
    ].mean(axis=1, skipna=True)
    segment_df["combined_high_ratio"] = segment_df[
        ["acc_high_band_energy_ratio_norm", "gyr_high_band_energy_ratio_norm"]
    ].mean(axis=1, skipna=True)
    return segment_df.sort_values("combined_difference_score").reset_index(drop=True)


def accumulate_selection(
    selected: Dict[Tuple[str, str, str, str], Dict[str, object]],
    row: pd.Series,
    reason: str,
) -> None:
    key = (
        row["kc_type"],
        row["experiment_id"],
        row["motion_folder"],
        row["segment"],
    )
    record = selected.get(key)
    if record is None:
        record = row.to_dict()
        record["selection_reason"] = reason
        selected[key] = record
        return
    reasons = set(str(record["selection_reason"]).split("; "))
    reasons.add(reason)
    record["selection_reason"] = "; ".join(sorted(reasons))


def select_examples(segment_summary_df: pd.DataFrame) -> pd.DataFrame:
    if segment_summary_df.empty:
        return pd.DataFrame()

    selected: Dict[Tuple[str, str, str, str], Dict[str, object]] = {}
    sorted_df = segment_summary_df.sort_values("combined_difference_score").reset_index(drop=True)
    n_examples = min(5, len(sorted_df))

    for _, row in sorted_df.head(n_examples).iterrows():
        accumulate_selection(selected, row, "stable_case")

    median_score = sorted_df["combined_difference_score"].median()
    sorted_df["median_distance"] = (sorted_df["combined_difference_score"] - median_score).abs()
    for _, row in sorted_df.nsmallest(n_examples, "median_distance").iterrows():
        accumulate_selection(selected, row, "medium_case")

    extreme_candidates = sorted_df.sort_values(
        ["combined_high_ratio", "combined_difference_score"],
        ascending=[False, False],
    )
    for _, row in extreme_candidates.head(n_examples).iterrows():
        accumulate_selection(selected, row, "high_difference_case")

    for keyword in MOTION_KEYWORDS:
        mask = sorted_df["motion_name"].str.contains(keyword, case=False, na=False)
        if not mask.any():
            continue
        best_row = (
            sorted_df.loc[mask]
            .sort_values(["combined_difference_score", "combined_high_ratio"], ascending=[False, False])
            .iloc[0]
        )
        accumulate_selection(selected, best_row, f"motion_type_{keyword}")

    for segment, group in sorted_df.groupby("segment"):
        best_row = group.sort_values("combined_difference_score", ascending=False).iloc[0]
        accumulate_selection(selected, best_row, f"segment_cover_{segment}")

    exp01_seg3 = sorted_df[
        (sorted_df["experiment_id"].str.lower() == "exp01")
        & (sorted_df["segment"].str.lower() == "seg3")
    ]
    if not exp01_seg3.empty:
        anomaly_candidate = exp01_seg3.sort_values(
            ["combined_difference_score", "combined_high_ratio"],
            ascending=[False, False],
        ).iloc[0]
        q75 = sorted_df["combined_difference_score"].quantile(0.75)
        if float(anomaly_candidate["combined_difference_score"]) >= float(q75):
            accumulate_selection(selected, anomaly_candidate, "anomaly_case_exp01_seg3")

    selected_df = pd.DataFrame(selected.values())
    if not selected_df.empty:
        selected_df = selected_df.sort_values(
            ["combined_difference_score", "combined_high_ratio"],
            ascending=[False, False],
        ).reset_index(drop=True)
    return selected_df


def build_sample_stem(row: pd.Series) -> str:
    motion_short = row["motion_index"]
    return (
        f"kc_{row['kc_type']}_{row['experiment_id']}_{motion_short}_{row['segment']}"
    )


def save_dataframe(df: pd.DataFrame, output_name: str) -> None:
    top_level_path = OUTPUT_ROOT / output_name
    table_path = TABLES_DIR / output_name
    df.to_csv(top_level_path, index=False)
    df.to_csv(table_path, index=False)


def setup_plot_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.figsize": (10, 6),
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
            "savefig.dpi": 180,
        }
    )


def save_figure(fig: plt.Figure, file_name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / file_name, bbox_inches="tight")
    plt.close(fig)


def plot_dataset_overview(metadata_df: pd.DataFrame) -> None:
    motion_df = build_motion_level_summary(metadata_df)
    if motion_df.empty:
        return

    motion_stats = (
        motion_df.groupby("motion_name")
        .agg(sample_count=("motion_folder", "count"), total_duration_sec=("duration_sec", "sum"))
        .sort_values("sample_count", ascending=False)
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(motion_stats.index, motion_stats["sample_count"], color="#4C78A8")
    ax.set_title("Motion Type Sample Counts")
    ax.set_ylabel("Motion count")
    ax.tick_params(axis="x", rotation=45)
    save_figure(fig, "dataset_motion_counts.png")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(motion_stats.index, motion_stats["total_duration_sec"], color="#F58518")
    ax.set_title("Motion Type Total Duration")
    ax.set_ylabel("Duration (s)")
    ax.tick_params(axis="x", rotation=45)
    save_figure(fig, "dataset_motion_durations.png")

    kc_stats = (
        motion_df.groupby("kc_type")
        .agg(sample_count=("motion_folder", "count"), total_duration_sec=("duration_sec", "sum"))
        .sort_index()
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(kc_stats.index, kc_stats["sample_count"], color="#54A24B")
    axes[0].set_title("arm vs gait Motion Counts")
    axes[0].set_ylabel("Motion count")
    axes[1].bar(kc_stats.index, kc_stats["total_duration_sec"], color="#E45756")
    axes[1].set_title("arm vs gait Total Duration")
    axes[1].set_ylabel("Duration (s)")
    save_figure(fig, "dataset_kc_comparison.png")


def plot_metric_overview(pairwise_df: pd.DataFrame, segment_summary_df: pd.DataFrame) -> None:
    if pairwise_df.empty:
        return

    focus = pairwise_df[pairwise_df["modality"].isin(["acc", "gyr"])].copy()
    if focus.empty:
        return

    fig, ax = plt.subplots(figsize=(13, 6))
    motion_names = sorted(focus["motion_name"].dropna().unique())
    positions: List[float] = []
    box_data: List[np.ndarray] = []
    box_labels: List[str] = []
    colors = {"acc": "#4C78A8", "gyr": "#F58518"}
    color_sequence: List[str] = []
    index = 0
    for motion_name in motion_names:
        for modality in ("acc", "gyr"):
            values = focus.loc[
                (focus["motion_name"] == motion_name) & (focus["modality"] == modality),
                "rmse_norm",
            ].dropna()
            if values.empty:
                continue
            positions.append(index)
            box_data.append(values.to_numpy())
            box_labels.append(f"{motion_name}\n{modality}")
            color_sequence.append(colors[modality])
            index += 1
    if box_data:
        box = ax.boxplot(box_data, positions=positions, patch_artist=True, showfliers=False)
        for patch, color in zip(box["boxes"], color_sequence):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_xticks(positions)
        ax.set_xticklabels(box_labels, rotation=45, ha="right")
        ax.set_ylabel("RMSE (norm)")
        ax.set_title("Rigid vs Nonrigid RMSE by Motion Type")
        save_figure(fig, "metrics_rmse_boxplot_by_motion.png")
    else:
        plt.close(fig)

    if not segment_summary_df.empty:
        segment_motion_heatmap = (
            segment_summary_df.pivot_table(
                index="segment",
                columns="motion_name",
                values="combined_rmse",
                aggfunc="mean",
            )
            .sort_index(key=lambda idx: [segment_sort_key(v) for v in idx])
        )
        plot_heatmap(
            segment_motion_heatmap,
            "Mean RMSE Heatmap (Segment x Motion)",
            "metrics_segment_rmse_heatmap.png",
            cmap="viridis",
            value_label="RMSE",
        )

        exp_segment_heatmap = (
            segment_summary_df.pivot_table(
                index="experiment_id",
                columns="segment",
                values="combined_high_ratio",
                aggfunc="mean",
            )
            .sort_index()
        )
        plot_heatmap(
            exp_segment_heatmap,
            "High-Frequency Energy Ratio Heatmap (Experiment x Segment)",
            "metrics_experiment_segment_highfreq_heatmap.png",
            cmap="magma",
            value_label="HF ratio",
        )


def plot_heatmap(
    heatmap_df: pd.DataFrame,
    title: str,
    file_name: str,
    cmap: str,
    value_label: str,
) -> None:
    if heatmap_df.empty:
        return
    data = heatmap_df.to_numpy(dtype=float)
    fig, ax = plt.subplots(
        figsize=(
            max(7, 0.75 * max(1, heatmap_df.shape[1])),
            max(4, 0.6 * max(1, heatmap_df.shape[0])),
        )
    )
    masked = np.ma.masked_invalid(data)
    image = ax.imshow(masked, aspect="auto", cmap=cmap)
    ax.set_xticks(np.arange(heatmap_df.shape[1]))
    ax.set_yticks(np.arange(heatmap_df.shape[0]))
    ax.set_xticklabels(list(heatmap_df.columns), rotation=45, ha="right")
    ax.set_yticklabels(list(heatmap_df.index))
    ax.set_title(title)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(value_label)
    save_figure(fig, file_name)


def compute_psd(signal_values: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    signal_values = np.asarray(signal_values, dtype=float)
    finite_mask = np.isfinite(signal_values)
    valid_values = signal_values[finite_mask]
    if len(valid_values) < 4 or fs <= 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    nperseg = min(1024, len(valid_values))
    freqs, psd = welch(valid_values, fs=fs, nperseg=nperseg)
    return freqs, psd


def get_motion_files(metadata_df: pd.DataFrame, row: pd.Series) -> Dict[str, Path]:
    subset = metadata_df[
        (metadata_df["kc_type"] == row["kc_type"])
        & (metadata_df["experiment_id"] == row["experiment_id"])
        & (metadata_df["motion_folder"] == row["motion_folder"])
    ]
    return {r.file_type: Path(r.path) for r in subset.itertuples()}


def plot_selected_examples(
    selected_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    file_cache: Dict[Path, LoadedCsv],
    warnings_list: List[str],
) -> None:
    if selected_df.empty:
        return

    for row in selected_df.itertuples(index=False):
        row_series = pd.Series(row._asdict())
        file_map = get_motion_files(metadata_df, row_series)
        if "imu_rigid" not in file_map or "imu_nonrigid" not in file_map:
            continue

        rigid_loaded = file_cache[file_map["imu_rigid"]]
        nonrigid_loaded = file_cache[file_map["imu_nonrigid"]]
        rigid_groups = standardize_signal_groups(rigid_loaded.data)
        nonrigid_groups = standardize_signal_groups(nonrigid_loaded.data)
        segment = row_series["segment"]
        if segment not in rigid_groups or segment not in nonrigid_groups:
            add_warning(
                warnings_list,
                f"Segment {segment} missing in selected example {row_series['motion_folder']}.",
            )
            continue

        rigid_segment = rigid_groups[segment]
        nonrigid_segment = nonrigid_groups[segment]
        aligned_rigid, aligned_nonrigid, common_time, aligned_fs, _ = local_linear_alignment(
            rigid_segment,
            nonrigid_segment,
            rigid_loaded.time_vector,
            nonrigid_loaded.time_vector,
            rigid_loaded.sampling_frequency,
            nonrigid_loaded.sampling_frequency,
        )
        stem = build_sample_stem(row_series)

        plot_three_axis_compare(
            common_time,
            aligned_rigid,
            aligned_nonrigid,
            "acc",
            f"{stem}_acc_compare.png",
            title_prefix=row_series["selection_reason"],
        )
        plot_three_axis_compare(
            common_time,
            aligned_rigid,
            aligned_nonrigid,
            "gyr",
            f"{stem}_gyr_compare.png",
            title_prefix=row_series["selection_reason"],
        )
        plot_norm_compare(common_time, aligned_rigid, aligned_nonrigid, stem)
        plot_residuals(common_time, aligned_rigid, aligned_nonrigid, stem)
        plot_psd_compare(aligned_rigid, aligned_nonrigid, aligned_fs, "acc", stem)
        plot_psd_compare(aligned_rigid, aligned_nonrigid, aligned_fs, "gyr", stem)

        if float(row_series.get("combined_high_ratio", np.nan)) >= 1.2:
            plot_zoomed_window(common_time, aligned_rigid, aligned_nonrigid, stem)

        if "omc" in file_map:
            try:
                plot_omc_reference(
                    row_series,
                    file_map["omc"],
                    rigid_loaded,
                    aligned_rigid,
                    common_time,
                    file_cache,
                    warnings_list,
                )
            except Exception as exc:
                add_warning(
                    warnings_list,
                    f"OMC plot failed for {row_series['motion_folder']} {segment}: {exc}",
                )


def plot_three_axis_compare(
    common_time: np.ndarray,
    rigid_df: pd.DataFrame,
    nonrigid_df: pd.DataFrame,
    modality: str,
    file_name: str,
    title_prefix: str,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for axis_obj, axis_name in zip(axes, ("x", "y", "z")):
        rigid_values = rigid_df[f"{modality}_{axis_name}"].to_numpy(dtype=float)
        nonrigid_values = nonrigid_df[f"{modality}_{axis_name}"].to_numpy(dtype=float)
        axis_obj.plot(common_time, rigid_values, label="rigid", linewidth=1.2)
        axis_obj.plot(common_time, nonrigid_values, label="nonrigid", linewidth=1.0, alpha=0.85)
        axis_obj.set_ylabel(axis_name)
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("Time (s)")
    axes[0].set_title(f"{title_prefix} | {modality.upper()} axis comparison")
    save_figure(fig, file_name)


def plot_norm_compare(
    common_time: np.ndarray,
    rigid_df: pd.DataFrame,
    nonrigid_df: pd.DataFrame,
    stem: str,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for axis_obj, modality in zip(axes, ("acc", "gyr")):
        axis_obj.plot(common_time, rigid_df[f"{modality}_norm"], label="rigid", linewidth=1.2)
        axis_obj.plot(
            common_time,
            nonrigid_df[f"{modality}_norm"],
            label="nonrigid",
            linewidth=1.0,
            alpha=0.85,
        )
        axis_obj.set_ylabel(f"{modality}_norm")
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("Time (s)")
    axes[0].set_title("Norm comparison")
    save_figure(fig, f"{stem}_norm_compare.png")


def plot_residuals(
    common_time: np.ndarray,
    rigid_df: pd.DataFrame,
    nonrigid_df: pd.DataFrame,
    stem: str,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for axis_obj, modality in zip(axes, ("acc", "gyr")):
        for axis_name in ("x", "y", "z"):
            residual = (
                nonrigid_df[f"{modality}_{axis_name}"].to_numpy(dtype=float)
                - rigid_df[f"{modality}_{axis_name}"].to_numpy(dtype=float)
            )
            axis_obj.plot(common_time, residual, linewidth=0.9, label=axis_name)
        axis_obj.set_ylabel(f"{modality} residual")
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("Time (s)")
    axes[0].set_title("Residual (nonrigid - rigid)")
    save_figure(fig, f"{stem}_residual.png")


def plot_psd_compare(
    rigid_df: pd.DataFrame,
    nonrigid_df: pd.DataFrame,
    fs: float,
    modality: str,
    stem: str,
) -> None:
    freqs_rigid, psd_rigid = compute_psd(rigid_df[f"{modality}_norm"].to_numpy(dtype=float), fs)
    freqs_nonrigid, psd_nonrigid = compute_psd(
        nonrigid_df[f"{modality}_norm"].to_numpy(dtype=float),
        fs,
    )
    if len(freqs_rigid) == 0 or len(freqs_nonrigid) == 0:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(freqs_rigid, psd_rigid, label="rigid", linewidth=1.2)
    ax.semilogy(freqs_nonrigid, psd_nonrigid, label="nonrigid", linewidth=1.2)
    ax.axvspan(0.0, 2.0, alpha=0.08, color="#54A24B", label="0-2 Hz")
    ax.axvspan(2.0, 5.0, alpha=0.08, color="#F2CF5B", label="2-5 Hz")
    ax.axvspan(5.0, min(15.0, fs / 2.0), alpha=0.08, color="#E45756", label="5-15 Hz")
    ax.set_xlim(0.0, min(20.0, fs / 2.0))
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD")
    ax.set_title(f"{modality.upper()} Welch PSD comparison")
    ax.legend(loc="upper right", ncol=2)
    save_figure(fig, f"{stem}_{modality}_psd.png")


def plot_zoomed_window(
    common_time: np.ndarray,
    rigid_df: pd.DataFrame,
    nonrigid_df: pd.DataFrame,
    stem: str,
) -> None:
    residual = nonrigid_df["acc_norm"].to_numpy(dtype=float) - rigid_df["acc_norm"].to_numpy(dtype=float)
    if len(residual) < 10:
        return
    window_samples = min(max(80, len(residual) // 6), len(residual))
    energy = np.convolve(np.abs(residual), np.ones(window_samples), mode="same")
    center = int(np.argmax(energy))
    half_window = window_samples // 2
    start = max(center - half_window, 0)
    end = min(center + half_window, len(common_time))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(common_time[start:end], rigid_df["acc_norm"].iloc[start:end], label="rigid")
    ax.plot(common_time[start:end], nonrigid_df["acc_norm"].iloc[start:end], label="nonrigid")
    ax.set_title("Zoomed window around strongest residual activity")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("acc_norm")
    ax.legend(loc="upper right")
    save_figure(fig, f"{stem}_zoomed_window.png")


def quaternion_to_angular_speed(quat_df: pd.DataFrame, fs: float) -> np.ndarray:
    required_cols = ["quat_w", "quat_x", "quat_y", "quat_z"]
    if not all(col in quat_df.columns for col in required_cols) or fs <= 0:
        return np.array([], dtype=float)
    quat_values = quat_df[required_cols].to_numpy(dtype=float)
    finite_mask = np.all(np.isfinite(quat_values), axis=1)
    quat_values = quat_values[finite_mask]
    if len(quat_values) < 3:
        return np.array([], dtype=float)
    rotations = Rotation.from_quat(quat_values[:, [1, 2, 3, 0]])
    delta = rotations[1:] * rotations[:-1].inv()
    angular_speed = delta.magnitude() * fs
    return angular_speed


def plot_omc_reference(
    row_series: pd.Series,
    omc_path: Path,
    rigid_loaded: LoadedCsv,
    aligned_rigid: pd.DataFrame,
    common_time: np.ndarray,
    file_cache: Dict[Path, LoadedCsv],
    warnings_list: List[str],
) -> None:
    omc_loaded = file_cache.get(omc_path)
    if omc_loaded is None:
        omc_loaded = read_dataset_csv(omc_path, warnings_list)
        file_cache[omc_path] = omc_loaded

    omc_groups = standardize_omc_groups(omc_loaded.data)
    segment = row_series["segment"]
    if segment not in omc_groups:
        add_warning(warnings_list, f"OMC segment {segment} missing in {omc_path.name}.")
        return

    segment_df = omc_groups[segment]
    quat_columns = [f"quat_{axis}" for axis in ("w", "x", "y", "z")]
    if not all(col in segment_df.columns for col in quat_columns):
        add_warning(warnings_list, f"Quaternion columns missing for {segment} in {omc_path.name}.")
        return

    quat_time = omc_loaded.time_vector
    angular_speed = quaternion_to_angular_speed(segment_df, omc_loaded.sampling_frequency)
    if len(angular_speed) == 0:
        return
    angular_time = quat_time[1 : len(angular_speed) + 1]

    # This interpolation is only for trend-level visualization between sensors
    # sampled in different systems; it should not be interpreted as a strict
    # frame-consistent physical equality.
    imu_gyr_interp = np.interp(
        angular_time,
        common_time,
        aligned_rigid["gyr_norm"].to_numpy(dtype=float),
    )

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=False)
    for quat_axis in quat_columns:
        axes[0].plot(quat_time, segment_df[quat_axis], linewidth=1.0, label=quat_axis)
    axes[0].set_title(
        "OMC quaternion components and trend-only angular-speed comparison"
    )
    axes[0].set_ylabel("Quaternion")
    axes[0].legend(loc="upper right", ncol=2)

    axes[1].plot(angular_time, angular_speed, label="OMC derived angular speed", linewidth=1.1)
    axes[1].plot(angular_time, imu_gyr_interp, label="IMU rigid gyr_norm", linewidth=1.1)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Trend magnitude")
    axes[1].legend(loc="upper right")
    axes[1].text(
        0.01,
        0.02,
        "Trend-only view: OMC and IMU use different sensing/reference definitions.",
        transform=axes[1].transAxes,
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.7},
    )
    save_figure(fig, f"{build_sample_stem(row_series)}_omc_reference.png")


def summarize_top_differences(segment_summary_df: pd.DataFrame) -> Dict[str, str]:
    if segment_summary_df.empty:
        return {
            "motion": "insufficient data",
            "segment": "insufficient data",
            "high_frequency": "insufficient data",
        }

    motion_summary = (
        segment_summary_df.groupby("motion_name")
        .agg(
            mean_score=("combined_difference_score", "mean"),
            mean_high=("combined_high_ratio", "mean"),
            sample_count=("segment", "count"),
        )
        .sort_values(["mean_score", "mean_high"], ascending=False)
    )
    segment_summary = (
        segment_summary_df.groupby("segment")
        .agg(
            mean_score=("combined_difference_score", "mean"),
            mean_high=("combined_high_ratio", "mean"),
        )
        .sort_values(["mean_score", "mean_high"], ascending=False)
    )

    top_motion = motion_summary.index[0]
    top_segment = segment_summary.index[0]
    high_freq_statement = (
        f"Mean combined high-frequency ratio = {segment_summary_df['combined_high_ratio'].mean():.2f}; "
        f"max = {segment_summary_df['combined_high_ratio'].max():.2f}."
    )
    return {
        "motion": f"{top_motion} (score={motion_summary.iloc[0]['mean_score']:.3f})",
        "segment": f"{top_segment} (score={segment_summary.iloc[0]['mean_score']:.3f})",
        "high_frequency": high_freq_statement,
    }


def generate_report(
    metadata_df: pd.DataFrame,
    segment_summary_df: pd.DataFrame,
    selected_df: pd.DataFrame,
) -> None:
    motion_df = build_motion_level_summary(metadata_df)
    highlights = summarize_top_differences(segment_summary_df)

    training_candidates = selected_df[
        selected_df["selection_reason"].str.contains("medium_case|stable_case", na=False)
    ]
    anomaly_candidates = selected_df[
        selected_df["selection_reason"].str.contains("high_difference_case|anomaly", na=False)
    ]

    report_lines = [
        "DIODEM exploratory analysis summary",
        "=" * 40,
        "",
        f"Dataset root: {DATASET_ROOT}",
        f"Total CSV files indexed: {len(metadata_df)}",
        f"Total motions (motion-level): {len(motion_df)}",
        "",
        "Key findings",
        f"- Motion type with strongest rigid/nonrigid difference: {highlights['motion']}",
        f"- Most sensitive segment overall: {highlights['segment']}",
        f"- High-frequency difference summary: {highlights['high_frequency']}",
        "",
        "Recommended downstream usage",
        "- Training candidates: prioritize stable_case and medium_case samples to learn baseline behavior.",
        "- Validation candidates: include medium_case samples spanning different motion types and segments.",
        "- Anomaly candidates: use high_difference_case and anomaly_case_exp01_seg3 samples for stress testing.",
        "",
        "Selected sample suggestions",
        f"- Stable/medium samples available: {len(training_candidates)}",
        f"- High-difference/anomaly samples available: {len(anomaly_candidates)}",
    ]

    if not selected_df.empty:
        report_lines.extend(
            [
                "",
                "Top representative samples",
            ]
        )
        for row in selected_df.head(10).itertuples():
            report_lines.append(
                f"- {row.kc_type} {row.experiment_id} {row.motion_folder} {row.segment}: "
                f"{row.selection_reason} | score={row.combined_difference_score:.3f} | "
                f"high_ratio={row.combined_high_ratio:.3f}"
            )

    report_path = OUTPUT_ROOT / "report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")


def save_warning_log(warnings_list: List[str]) -> None:
    if not warnings_list:
        return
    warning_path = TABLES_DIR / "warnings.log"
    warning_path.write_text("\n".join(warnings_list), encoding="utf-8")


def main() -> None:
    warnings.filterwarnings("ignore", message="Glyph .* missing from current font.")
    ensure_output_dirs()
    setup_plot_style()

    print_stage("Stage 1/6 - scanning dataset and building metadata index")
    metadata_df, file_cache, warnings_list = scan_dataset(DATASET_ROOT)
    if metadata_df.empty:
        raise RuntimeError(f"No valid CSV files were found under {DATASET_ROOT}")
    save_dataframe(metadata_df, "metadata_summary.csv")
    print_dataset_summary(metadata_df)

    print_stage("Stage 2/6 - computing rigid/nonrigid pairwise metrics")
    pairwise_df = compute_pairwise_metrics(metadata_df, file_cache, warnings_list)
    save_dataframe(pairwise_df, "pairwise_metrics.csv")

    print_stage("Stage 3/6 - selecting representative examples")
    segment_summary_df = build_segment_summary(pairwise_df)
    selected_df = select_examples(segment_summary_df)
    save_dataframe(selected_df, "selected_examples.csv")

    print_stage("Stage 4/6 - generating dataset overview figures")
    plot_dataset_overview(metadata_df)

    print_stage("Stage 5/6 - generating metric overview and representative figures")
    plot_metric_overview(pairwise_df, segment_summary_df)
    plot_selected_examples(selected_df, metadata_df, file_cache, warnings_list)

    print_stage("Stage 6/6 - writing report and warning log")
    generate_report(metadata_df, segment_summary_df, selected_df)
    save_warning_log(warnings_list)

    print_stage(f"Done. Outputs saved to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
