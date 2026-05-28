from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from project.inference import StreamingCompensator


CHANNELS = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]
LEFT_COLORS = {
    "acc_x": "#E31A1C",
    "acc_y": "#33A02C",
    "acc_z": "#1F78B4",
    "gyr_x": "#6A3D9A",
    "gyr_y": "#FF7F00",
    "gyr_z": "#4DBBD5",
}
CHANNEL_LABEL_COLOR = "#111827"
RAW_COLORS = {
    "uncompensated": "#E76F00",
    "reference": "#111827",
}
RIGHT_COLORS = {
    "original": "#2F6DB3",
    "compensated": "#E76F00",
    "reference": "#555555",
}


def _setup_nature_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.0,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "legend.frameon": False,
        }
    )


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_origin_export(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = _read_csv_rows(csv_path)
    times = np.asarray([float(row["time_s"]) for row in rows], dtype=np.float32)
    input_matrix = np.asarray(
        [
            [float(row[f"nonrigid_{prefix}_{axis}"]) for prefix in ("acc",) for axis in ("x", "y", "z")]
            + [float(row[f"nonrigid_{prefix}_{axis}"]) for prefix in ("gyr",) for axis in ("x", "y", "z")]
            for row in rows
        ],
        dtype=np.float32,
    )
    ref_matrix = np.asarray(
        [
            [float(row[f"rigid_{prefix}_{axis}"]) for prefix in ("acc",) for axis in ("x", "y", "z")]
            + [float(row[f"rigid_{prefix}_{axis}"]) for prefix in ("gyr",) for axis in ("x", "y", "z")]
            for row in rows
        ],
        dtype=np.float32,
    )
    return times, input_matrix, ref_matrix


def _robust_scale(values: np.ndarray) -> np.ndarray:
    centered = values - np.median(values)
    scale = np.percentile(np.abs(centered), 95)
    if scale < 1e-8:
        scale = np.max(np.abs(centered)) + 1e-8
    return centered / scale


def _normalize_for_stack(values: np.ndarray, amplitude: float = 0.34) -> np.ndarray:
    return amplitude * np.clip(_robust_scale(values), -2.2, 2.2)


def _shared_group_scale(matrix: np.ndarray) -> Tuple[np.ndarray, float]:
    centers = np.median(matrix, axis=0)
    centered = matrix - centers
    scale = float(np.percentile(np.abs(centered), 95))
    if scale < 1e-8:
        scale = float(np.max(np.abs(centered)) + 1e-8)
    return centers, scale


def _stack_with_shared_scale(values: np.ndarray, center: float, scale: float, amplitude: float) -> np.ndarray:
    return amplitude * np.clip((values - center) / scale, -2.3, 2.3)


def _draw_time_arrow(ax: plt.Axes, x0: float, x1: float, y: float, font_size: int) -> None:
    ax.annotate(
        "",
        xy=(x1, y),
        xytext=(x0, y),
        arrowprops={"arrowstyle": "->", "lw": 1.6, "color": "#111827"},
        annotation_clip=False,
    )
    ax.text((x0 + x1) / 2.0, y - 0.22, "Time", ha="center", va="top", fontsize=font_size, fontweight="bold")


def _draw_unit_bracket(ax: plt.Axes, x: float, y0: float, y1: float, label: str, font_size: int) -> None:
    cap = 0.08 * (ax.get_xlim()[1] - ax.get_xlim()[0])
    ax.plot([x, x], [y0, y1], color="#111827", lw=1.2, clip_on=False)
    ax.plot([x - cap, x], [y0, y0], color="#111827", lw=1.2, clip_on=False)
    ax.plot([x - cap, x], [y1, y1], color="#111827", lw=1.2, clip_on=False)
    ax.text(x + cap * 0.18, (y0 + y1) / 2, label, ha="left", va="center", fontsize=font_size)


def _plot_raw_6axis(times: np.ndarray, input_matrix: np.ndarray, ref_matrix: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.55, 4.4), dpi=600)
    offsets = np.asarray([5.2, 4.25, 3.3, 2.0, 1.05, 0.1], dtype=np.float32)
    t = times - times[0]
    acc_centers, acc_scale = _shared_group_scale(np.vstack([input_matrix[:, :3], ref_matrix[:, :3]]))
    gyr_centers, gyr_scale = _shared_group_scale(np.vstack([input_matrix[:, 3:], ref_matrix[:, 3:]]))

    for idx, channel in enumerate(CHANNELS):
        if idx < 3:
            y = offsets[idx] + _stack_with_shared_scale(input_matrix[:, idx], acc_centers[idx], acc_scale, amplitude=0.34)
            ref_y = offsets[idx] + _stack_with_shared_scale(ref_matrix[:, idx], acc_centers[idx], acc_scale, amplitude=0.34)
        else:
            group_idx = idx - 3
            y = offsets[idx] + _stack_with_shared_scale(input_matrix[:, idx], gyr_centers[group_idx], gyr_scale, amplitude=0.34)
            ref_y = offsets[idx] + _stack_with_shared_scale(ref_matrix[:, idx], gyr_centers[group_idx], gyr_scale, amplitude=0.34)
        ax.plot(t, y, color=RAW_COLORS["uncompensated"], lw=1.0, alpha=0.94, zorder=2)
        ax.plot(t, ref_y, color=RAW_COLORS["reference"], lw=1.05, ls=(0, (4, 3)), alpha=0.82, zorder=3)
        axis_label = channel.replace("acc_", "a").replace("gyr_", "g")
        axis_label = axis_label[:-1] + f"$_{axis_label[-1]}$"
        ax.text(t[0] - 0.08 * t.ptp(), offsets[idx], axis_label, color=CHANNEL_LABEL_COLOR, fontsize=7.4, ha="right", va="center")

    ax.plot([], [], color=RAW_COLORS["uncompensated"], lw=1.35, label="Soft-attached IMU")
    ax.plot([], [], color=RAW_COLORS["reference"], lw=1.25, ls=(0, (4, 3)), label="Rigid reference")
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.18, 1.02),
        fontsize=6.2,
        handlelength=2.5,
        borderaxespad=0.0,
        columnspacing=1.0,
        handletextpad=0.45,
        ncol=2,
    )

    for xpos in np.linspace(t[0] + 0.18 * t.ptp(), t[-1] - 0.16 * t.ptp(), 4):
        ax.axvline(xpos, color="#B7BDC7", lw=0.8, ls="--", zorder=0)

    ax.text(t[0] - 0.24 * t.ptp(), np.mean(offsets[:3]), "Accel", color="#1D4ED8", fontsize=8.6, fontweight="bold", rotation=90, ha="center", va="center")
    ax.text(t[0] - 0.24 * t.ptp(), np.mean(offsets[3:]), "Gyro", color="#1D4ED8", fontsize=8.6, fontweight="bold", rotation=90, ha="center", va="center")
    ax.text(t[0] - 0.18 * t.ptp(), np.mean(offsets[:3]), "(3-axis)", color="#1D4ED8", fontsize=7.0, fontweight="bold", rotation=90, ha="center", va="center")
    ax.text(t[0] - 0.18 * t.ptp(), np.mean(offsets[3:]), "(3-axis)", color="#1D4ED8", fontsize=7.0, fontweight="bold", rotation=90, ha="center", va="center")

    ax.set_xlim(t[0] - 0.34 * t.ptp(), t[-1] + 0.26 * t.ptp())
    ax.set_ylim(-0.68, 6.23)
    x_right = t[-1] + 0.13 * t.ptp()
    _draw_unit_bracket(ax, x_right, 3.03, 5.45, "m/s$^2$", 7.0)
    _draw_unit_bracket(ax, x_right, -0.18, 2.25, "rad/s", 7.0)
    _draw_time_arrow(ax, t[0] + 0.54 * t.ptp(), t[0] + 0.86 * t.ptp(), -0.53, 8)

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    fig.savefig(output_path.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(output_path.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def _plot_compensation_comparison(
    times: np.ndarray,
    input_matrix: np.ndarray,
    ref_matrix: np.ndarray,
    pred_matrix: np.ndarray,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(3.95, 6.45), dpi=600)
    offsets = np.asarray([5.85, 4.85, 3.85, 2.45, 1.45, 0.45], dtype=np.float32)
    t = times - times[0]
    acc_combined = np.vstack([input_matrix[:, :3], ref_matrix[:, :3], pred_matrix[:, :3]])
    gyr_combined = np.vstack([input_matrix[:, 3:], ref_matrix[:, 3:], pred_matrix[:, 3:]])
    acc_centers, acc_scale = _shared_group_scale(acc_combined)
    gyr_centers, gyr_scale = _shared_group_scale(gyr_combined)

    for idx, channel in enumerate(CHANNELS):
        if idx < 3:
            center = acc_centers[idx]
            scale = acc_scale
        else:
            group_idx = idx - 3
            center = gyr_centers[group_idx]
            scale = gyr_scale
        raw = _stack_with_shared_scale(input_matrix[:, idx], center, scale, amplitude=0.20) + offsets[idx]
        ref = _stack_with_shared_scale(ref_matrix[:, idx], center, scale, amplitude=0.32) + offsets[idx]
        pred = _stack_with_shared_scale(pred_matrix[:, idx], center, scale, amplitude=0.32) + offsets[idx]

        ax.plot(t, raw, color=RIGHT_COLORS["original"], lw=0.75, alpha=0.42)
        ax.plot(t, ref, color=RIGHT_COLORS["reference"], lw=0.95, ls=(0, (4, 3)), alpha=0.86)
        ax.plot(t, pred, color=RIGHT_COLORS["compensated"], lw=1.35)
        axis_label = channel.replace("acc_", "a").replace("gyr_", "g")
        axis_label = axis_label[:-1] + f"$_{axis_label[-1]}$"
        ax.text(t[0] - 0.08 * t.ptp(), offsets[idx], axis_label, color=CHANNEL_LABEL_COLOR, fontsize=7.2, ha="right", va="center")

    ax.plot([], [], color=RIGHT_COLORS["original"], lw=1.1, alpha=0.55, label="Original")
    ax.plot([], [], color=RIGHT_COLORS["compensated"], lw=1.7, label="Compensated")
    ax.plot([], [], color=RIGHT_COLORS["reference"], lw=1.1, ls=(0, (4, 3)), label="Rigid reference")
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.15, 0.83),
        fontsize=6,
        handlelength=2.1,
        borderaxespad=0.0,
        columnspacing=0.85,
        handletextpad=0.42,
        ncol=3,
    )

    ax.text(t[0] - 0.22 * t.ptp(), np.mean(offsets[:3]), "Accel\n(3-axis)", color="#1D4ED8", fontsize=8.4, fontweight="bold", rotation=90, ha="center", va="center", linespacing=0.9)
    ax.text(t[0] - 0.22 * t.ptp(), np.mean(offsets[3:]), "Gyro\n(3-axis)", color="#1D4ED8", fontsize=8.4, fontweight="bold", rotation=90, ha="center", va="center", linespacing=0.9)

    ax.set_xlim(t[0] - 0.32 * t.ptp(), t[-1] + 0.26 * t.ptp())
    ax.set_ylim(-0.35, 7.9)
    x_right = t[-1] + 0.13 * t.ptp()
    _draw_unit_bracket(ax, x_right, 3.43, 6.12, "m/s$^2$", 7.0)
    _draw_unit_bracket(ax, x_right, 0.05, 2.76, "rad/s", 7.0)
    _draw_time_arrow(ax, t[0] + 0.35 * t.ptp(), t[0] + 0.72 * t.ptp(), -0.18, 8)

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    fig.savefig(output_path.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(output_path.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def _write_source_data(
    output_path: Path,
    times: np.ndarray,
    input_matrix: np.ndarray,
    ref_matrix: np.ndarray,
    pred_matrix: np.ndarray,
) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["time_s"]
        fieldnames.extend([f"original_{name}" for name in CHANNELS])
        fieldnames.extend([f"rigid_{name}" for name in CHANNELS])
        fieldnames.extend([f"compensated_{name}" for name in CHANNELS])
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row_idx, time_value in enumerate(times):
            row: Dict[str, str] = {"time_s": f"{float(time_value):.6f}"}
            row.update({f"original_{name}": f"{float(input_matrix[row_idx, idx]):.8f}" for idx, name in enumerate(CHANNELS)})
            row.update({f"rigid_{name}": f"{float(ref_matrix[row_idx, idx]):.8f}" for idx, name in enumerate(CHANNELS)})
            row.update({f"compensated_{name}": f"{float(pred_matrix[row_idx, idx]):.8f}" for idx, name in enumerate(CHANNELS)})
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw Nature-style IMU compensation illustration panels from real data.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/supervised_tcn_causal_by_experiment/seed_runs/seed_0042/training/checkpoints/best.pt"),
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("outputs/paper_tables/origin_motion_panel_data/canonical_origin_export.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("docs/illustration-compensated"))
    args = parser.parse_args()

    _setup_nature_style()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    times, input_matrix, ref_matrix = _read_origin_export(args.input_csv.resolve())
    compensator = StreamingCompensator.from_checkpoint(args.checkpoint.resolve(), device_name="cpu")
    pred_matrix = compensator.process_sequence(input_matrix, reset=True)["predictions"]

    _plot_raw_6axis(times, input_matrix, ref_matrix, output_dir / "raw_6axis_real_imu_signals")
    _plot_compensation_comparison(times, input_matrix, ref_matrix, pred_matrix, output_dir / "compensated_vs_original_real_signals")
    _write_source_data(output_dir / "illustration_source_data.csv", times, input_matrix, ref_matrix, pred_matrix)
    print(f"Wrote illustration panels to {output_dir}")


if __name__ == "__main__":
    main()
