from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import yaml


IMU_SIGNAL_PATTERN = re.compile(r"^(seg\d+)_(acc|gyr|mag)_(x|y|z)$", re.IGNORECASE)


def ensure_dir(path: Path) -> Path:
    """Create a directory if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    """Load a YAML config and attach useful path anchors."""
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config at {config_path} did not produce a mapping.")
    config["config_path"] = str(config_path)
    config["config_dir"] = str(config_path.parent)
    config["repo_root"] = str(config_path.parent.parent.parent)
    return config


def resolve_path(path_like: str | Path, base_dir: Path) -> Path:
    """Resolve a path relative to a base directory."""
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def save_json(data: Dict[str, Any], path: Path) -> None:
    """Save a JSON file with deterministic formatting."""
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON file."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_pickle(data: Any, path: Path) -> None:
    """Save a pickle file."""
    with path.open("wb") as handle:
        pickle.dump(data, handle)


def load_pickle(path: Path) -> Any:
    """Load a pickle file."""
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_csv_table(path: Path) -> pd.DataFrame:
    """Load a CSV table with a clear error if it is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Required CSV not found: {path}")
    return pd.read_csv(path)


def extract_sampling_frequency(csv_path: Path) -> float:
    """Parse the `# sampling frequency:` header from a DIODEM CSV."""
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


def read_imu_csv(csv_path: Path) -> Dict[str, Any]:
    """Read a DIODEM IMU CSV into a numeric DataFrame plus time vector."""
    sampling_frequency = extract_sampling_frequency(csv_path)
    frame = pd.read_csv(csv_path, comment="#")
    frame.columns = [str(col).strip() for col in frame.columns]
    frame = frame.loc[:, [col for col in frame.columns if col and not col.lower().startswith("unnamed")]]
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(float)
    time_vector = np.arange(len(frame), dtype=float) / sampling_frequency
    return {
        "data": frame,
        "sampling_frequency": sampling_frequency,
        "time_vector": time_vector,
    }


def split_imu_segments(frame: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Split raw IMU columns into per-segment channel frames."""
    grouped: Dict[str, Dict[str, pd.Series]] = {}
    for column in frame.columns:
        match = IMU_SIGNAL_PATTERN.match(column)
        if not match:
            continue
        segment_id, signal_type, axis = match.groups()
        grouped.setdefault(segment_id.lower(), {})
        grouped[segment_id.lower()][f"{signal_type.lower()}_{axis.lower()}"] = frame[column]

    segments: Dict[str, pd.DataFrame] = {}
    for segment_id, columns in grouped.items():
        segment_frame = pd.DataFrame(index=frame.index)
        for signal_type in ("acc", "gyr", "mag"):
            for axis in ("x", "y", "z"):
                name = f"{signal_type}_{axis}"
                if name in columns:
                    segment_frame[name] = columns[name].astype(float)
        segments[segment_id] = segment_frame
    return segments


def select_channels(segment_frame: pd.DataFrame, channels: Iterable[str]) -> np.ndarray:
    """Extract channels from a per-segment frame as a `[T, C]` array."""
    channel_list = list(channels)
    missing = [channel for channel in channel_list if channel not in segment_frame.columns]
    if missing:
        raise KeyError(f"Missing requested channels: {missing}")
    values = segment_frame[channel_list].to_numpy(dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"Expected a 2D segment array, got shape {values.shape}")
    return values


def inverse_zscore(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Undo z-score normalization."""
    return values * std + mean

