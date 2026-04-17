from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from project.utils.io import load_pickle
from project.utils.torch_compat import Dataset, TORCH_AVAILABLE, require_torch, torch


def generate_window_start_indices(length: int, window_size: int, stride: int) -> List[int]:
    """Return valid sliding-window start indices."""
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive integers.")
    if length < window_size:
        return []
    return list(range(0, length - window_size + 1, stride))


def fit_normalization_stats(samples: Dict[str, np.ndarray], normalization: str) -> Dict[str, Any]:
    """Fit normalization statistics from raw train samples."""
    if normalization == "none":
        return {"mode": "none"}

    inputs = np.asarray(samples["inputs"], dtype=np.float32)
    targets = np.asarray(samples["targets"], dtype=np.float32)
    if inputs.ndim != 3 or targets.ndim != 3:
        raise ValueError("Expected `[N, T, C]` arrays for inputs and targets.")

    if normalization == "global_zscore":
        return {
            "mode": normalization,
            "input_mean": float(inputs.mean()),
            "input_std": float(max(inputs.std(), 1e-6)),
            "target_mean": float(targets.mean()),
            "target_std": float(max(targets.std(), 1e-6)),
        }

    if normalization == "per_channel_zscore":
        return {
            "mode": normalization,
            "input_mean": inputs.mean(axis=(0, 1)).astype(np.float32),
            "input_std": np.maximum(inputs.std(axis=(0, 1)), 1e-6).astype(np.float32),
            "target_mean": targets.mean(axis=(0, 1)).astype(np.float32),
            "target_std": np.maximum(targets.std(axis=(0, 1)), 1e-6).astype(np.float32),
        }

    raise ValueError(f"Unsupported normalization mode: {normalization}")


def apply_normalization(
    samples: Dict[str, np.ndarray],
    stats: Dict[str, Any],
    normalization: str,
) -> Dict[str, np.ndarray]:
    """Apply pre-fit normalization to cached raw windows."""
    if normalization == "none" or stats.get("mode") == "none":
        return {
            "inputs": np.asarray(samples["inputs"], dtype=np.float32),
            "targets": np.asarray(samples["targets"], dtype=np.float32),
        }

    inputs = np.asarray(samples["inputs"], dtype=np.float32)
    targets = np.asarray(samples["targets"], dtype=np.float32)
    input_mean = np.asarray(stats["input_mean"], dtype=np.float32)
    input_std = np.asarray(stats["input_std"], dtype=np.float32)
    target_mean = np.asarray(stats["target_mean"], dtype=np.float32)
    target_std = np.asarray(stats["target_std"], dtype=np.float32)
    normalized = {
        "inputs": ((inputs - input_mean) / input_std).astype(np.float32),
        "targets": ((targets - target_mean) / target_std).astype(np.float32),
    }
    return normalized


class WindowedPairDataset(Dataset):
    """Torch dataset wrapper around cached raw `[N, T, C]` windows."""

    def __init__(
        self,
        bundle_path: Path,
        normalization: str = "none",
        normalization_stats: Dict[str, Any] | None = None,
    ) -> None:
        bundle = load_pickle(bundle_path)
        self.raw_inputs = np.asarray(bundle["inputs"], dtype=np.float32)
        self.raw_targets = np.asarray(bundle["targets"], dtype=np.float32)
        self.metadata = list(bundle["metadata"])
        self.channels = list(bundle["channels"])
        self.normalization = normalization
        self.normalization_stats = normalization_stats or {"mode": "none"}

        if self.raw_inputs.shape != self.raw_targets.shape:
            raise ValueError("Inputs and targets must share the same cached shape.")
        if self.raw_inputs.ndim != 3:
            raise ValueError(f"Expected `[N, T, C]` cached data, got {self.raw_inputs.shape}")

    def __len__(self) -> int:
        return int(self.raw_inputs.shape[0])

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = {
            "inputs": self.raw_inputs[index],
            "targets": self.raw_targets[index],
            "metadata": self.metadata[index],
        }
        normalized = apply_normalization(item, self.normalization_stats, self.normalization)
        if TORCH_AVAILABLE:  # pragma: no branch
            return {
                "inputs": torch.from_numpy(normalized["inputs"]),
                "targets": torch.from_numpy(normalized["targets"]),
                "metadata": item["metadata"],
            }
        return {
            "inputs": normalized["inputs"],
            "targets": normalized["targets"],
            "metadata": item["metadata"],
        }


def windowed_pair_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate windows while preserving metadata as a simple Python list."""
    require_torch()
    inputs = torch.stack([sample["inputs"] for sample in batch], dim=0)
    targets = torch.stack([sample["targets"] for sample in batch], dim=0)
    metadata = [sample["metadata"] for sample in batch]
    return {"inputs": inputs, "targets": targets, "metadata": metadata}
