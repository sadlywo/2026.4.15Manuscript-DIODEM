from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch


def _norm(values: np.ndarray, channels: List[str], prefix: str) -> np.ndarray:
    idx = [i for i, name in enumerate(channels) if name.startswith(prefix)]
    if not idx:
        return np.zeros(values.shape[0], dtype=np.float32)
    return np.linalg.norm(values[:, idx], axis=1)


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_prediction_bundle(
    sample: Dict[str, np.ndarray],
    channels: List[str],
    sampling_frequency: float,
    output_dir: Path,
    stem: str,
) -> None:
    """Create all requested diagnostic plots for one predicted window."""
    output_dir.mkdir(parents=True, exist_ok=True)
    time_vector = np.arange(sample["inputs"].shape[0], dtype=float) / sampling_frequency

    fig, axes = plt.subplots(len(channels), 1, figsize=(12, max(6, 2 * len(channels))), sharex=True)
    if len(channels) == 1:
        axes = [axes]
    for axis_obj, channel_idx in zip(axes, range(len(channels))):
        axis_obj.plot(time_vector, sample["inputs"][:, channel_idx], label="nonrigid", alpha=0.9)
        axis_obj.plot(time_vector, sample["targets"][:, channel_idx], label="rigid", alpha=0.9)
        axis_obj.plot(time_vector, sample["predictions"][:, channel_idx], label="pred", alpha=0.9)
        axis_obj.set_ylabel(channels[channel_idx])
    axes[0].legend(loc="upper right", ncol=3)
    axes[-1].set_xlabel("Time (s)")
    _save(fig, output_dir / f"{stem}_triple_compare.png")

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(time_vector, _norm(sample["inputs"], channels, "acc_") - _norm(sample["targets"], channels, "acc_"), label="nonrigid-rigid")
    axes[0].plot(time_vector, _norm(sample["predictions"], channels, "acc_") - _norm(sample["targets"], channels, "acc_"), label="pred-rigid")
    axes[0].set_ylabel("acc residual")
    axes[1].plot(time_vector, _norm(sample["inputs"], channels, "gyr_") - _norm(sample["targets"], channels, "gyr_"), label="nonrigid-rigid")
    axes[1].plot(time_vector, _norm(sample["predictions"], channels, "gyr_") - _norm(sample["targets"], channels, "gyr_"), label="pred-rigid")
    axes[1].set_ylabel("gyr residual")
    axes[1].set_xlabel("Time (s)")
    axes[0].legend(loc="upper right")
    _save(fig, output_dir / f"{stem}_residual.png")

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for axis_obj, prefix in zip(axes, ("acc_", "gyr_")):
        axis_obj.plot(time_vector, _norm(sample["inputs"], channels, prefix), label="nonrigid")
        axis_obj.plot(time_vector, _norm(sample["targets"], channels, prefix), label="rigid")
        axis_obj.plot(time_vector, _norm(sample["predictions"], channels, prefix), label="pred")
        axis_obj.set_ylabel(f"{prefix[:-1]}_norm")
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("Time (s)")
    _save(fig, output_dir / f"{stem}_norm_compare.png")

    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    for axis_obj, prefix in zip(axes, ("acc_", "gyr_")):
        for label, array in (
            ("nonrigid", sample["inputs"]),
            ("rigid", sample["targets"]),
            ("pred", sample["predictions"]),
        ):
            norm_signal = _norm(array, channels, prefix)
            freqs, psd = welch(norm_signal, fs=sampling_frequency, nperseg=min(256, len(norm_signal)))
            axis_obj.semilogy(freqs, psd, label=label)
        axis_obj.set_ylabel(f"{prefix[:-1]} PSD")
        axis_obj.set_xlim(0.0, min(20.0, sampling_frequency / 2.0))
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("Frequency (Hz)")
    _save(fig, output_dir / f"{stem}_psd_compare.png")

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(time_vector, np.mean(np.abs(sample["predictions"] - sample["targets"]), axis=1))
    axes[0].set_ylabel("mean abs error")
    axes[1].plot(time_vector, np.sqrt(np.mean((sample["predictions"] - sample["targets"]) ** 2, axis=1)))
    axes[1].set_ylabel("rmse over channels")
    axes[1].set_xlabel("Time (s)")
    _save(fig, output_dir / f"{stem}_error_over_time.png")

