from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Sequence

import numpy as np

from acquisition.sinct485 import INPUT_CHANNELS


PREDICTION_CHANNELS = ["pred_acc_x", "pred_acc_y", "pred_acc_z", "pred_gyr_x", "pred_gyr_y", "pred_gyr_z"]
CHANNEL_LABELS = ["Acc x", "Acc y", "Acc z", "Gyr x", "Gyr y", "Gyr z"]
CHANNEL_UNITS = ["m s$^{-2}$", "m s$^{-2}$", "m s$^{-2}$", "rad s$^{-1}$", "rad s$^{-1}$", "rad s$^{-1}$"]
RAW_COLOR = "#4C78A8"
PRED_COLOR = "#E45756"


@dataclass(frozen=True)
class SignalSnapshot:
    time_s: np.ndarray
    raw: np.ndarray
    pred: np.ndarray


class RollingSignalBuffer:
    """Fixed-length rolling store for raw and compensated six-channel samples."""

    def __init__(self, max_points: int) -> None:
        if int(max_points) <= 0:
            raise ValueError("max_points must be positive.")
        self.max_points = int(max_points)
        self._time_s: Deque[float] = deque(maxlen=self.max_points)
        self._raw: Deque[np.ndarray] = deque(maxlen=self.max_points)
        self._pred: Deque[np.ndarray] = deque(maxlen=self.max_points)

    def append(self, elapsed_s: float, raw_values: Sequence[float], pred_values: Sequence[float]) -> None:
        raw = np.asarray(raw_values, dtype=np.float32)
        pred = np.asarray(pred_values, dtype=np.float32)
        if raw.shape != (6,) or pred.shape != (6,):
            raise ValueError(f"Expected raw and pred vectors with shape (6,), got {raw.shape} and {pred.shape}.")
        self._time_s.append(float(elapsed_s))
        self._raw.append(raw)
        self._pred.append(pred)

    def append_row(self, row: Dict[str, object]) -> None:
        raw = [float(row[channel]) for channel in INPUT_CHANNELS]
        pred = [float(row[channel]) for channel in PREDICTION_CHANNELS]
        self.append(elapsed_s=float(row["elapsed_s"]), raw_values=raw, pred_values=pred)

    def snapshot(self) -> SignalSnapshot:
        if not self._time_s:
            return SignalSnapshot(
                time_s=np.asarray([], dtype=np.float32),
                raw=np.empty((0, 6), dtype=np.float32),
                pred=np.empty((0, 6), dtype=np.float32),
            )
        return SignalSnapshot(
            time_s=np.asarray(self._time_s, dtype=np.float32),
            raw=np.vstack(list(self._raw)).astype(np.float32),
            pred=np.vstack(list(self._pred)).astype(np.float32),
        )


def apply_nature_realtime_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


class LiveSignalPlotter:
    """Matplotlib real-time viewer for raw and compensated IMU streams."""

    def __init__(
        self,
        *,
        window_sec: float = 10.0,
        rate_hz: float = 40.0,
        update_interval_ms: float = 100.0,
        title: str = "SINCT-485 live compensation",
    ) -> None:
        if window_sec <= 0:
            raise ValueError("window_sec must be positive.")
        if rate_hz <= 0:
            raise ValueError("rate_hz must be positive.")
        self.buffer = RollingSignalBuffer(max_points=max(2, int(math.ceil(window_sec * rate_hz))))
        self.update_interval_sec = max(0.001, float(update_interval_ms) / 1000.0)
        self._last_update = 0.0

        apply_nature_realtime_style()
        import matplotlib.pyplot as plt

        plt.ion()
        self._plt = plt
        self.figure, axes = plt.subplots(2, 3, figsize=(11.0, 5.8), sharex=True)
        self.axes = list(axes.ravel())
        self.raw_lines = []
        self.pred_lines = []
        for index, axis in enumerate(self.axes):
            raw_line, = axis.plot([], [], color=RAW_COLOR, linewidth=1.0, label="Raw")
            pred_line, = axis.plot([], [], color=PRED_COLOR, linewidth=1.0, label="Compensated")
            axis.set_title(CHANNEL_LABELS[index], fontsize=8, fontweight="bold")
            axis.set_ylabel(CHANNEL_UNITS[index])
            axis.grid(True, color="#e6e6e6", linewidth=0.5)
            self.raw_lines.append(raw_line)
            self.pred_lines.append(pred_line)
        for axis in self.axes[3:]:
            axis.set_xlabel("Time (s)")
        self.axes[0].legend(loc="upper left", ncols=2)
        self.figure.suptitle(title, fontsize=10, fontweight="bold")
        self.figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
        self.figure.show()

    def push_row(self, row: Dict[str, object]) -> None:
        self.buffer.append_row(row)
        now = time.perf_counter()
        if now - self._last_update >= self.update_interval_sec:
            self.update(force=False)
            self._last_update = now

    def update(self, force: bool = True) -> None:
        snapshot = self.buffer.snapshot()
        if snapshot.time_s.size == 0:
            return
        for index in range(6):
            self.raw_lines[index].set_data(snapshot.time_s, snapshot.raw[:, index])
            self.pred_lines[index].set_data(snapshot.time_s, snapshot.pred[:, index])
            axis = self.axes[index]
            axis.set_xlim(float(snapshot.time_s[0]), float(snapshot.time_s[-1]) if snapshot.time_s.size > 1 else float(snapshot.time_s[0] + 1.0))
            y_values = np.concatenate([snapshot.raw[:, index], snapshot.pred[:, index]])
            y_min = float(np.nanmin(y_values))
            y_max = float(np.nanmax(y_values))
            if not np.isfinite(y_min) or not np.isfinite(y_max):
                y_min, y_max = -1.0, 1.0
            if abs(y_max - y_min) < 1e-6:
                y_min -= 1.0
                y_max += 1.0
            pad = 0.08 * (y_max - y_min)
            axis.set_ylim(y_min - pad, y_max + pad)
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()
        self._plt.pause(0.001 if force else 0.0001)

    def close(self) -> None:
        self.update(force=True)
        self._plt.ioff()
