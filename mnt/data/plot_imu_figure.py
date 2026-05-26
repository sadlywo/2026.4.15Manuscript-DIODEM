"""Create a publication-ready multi-panel IMU signal comparison figure.

The script reads the DIODEM representative IMU traces from an Excel workbook,
plots rigid-reference and non-rigid signals for six motion conditions, and
exports high-resolution figure files plus summary error metrics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import SubplotSpec
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


PANEL_CONFIG = [
    ("Canonical", "(a) Standard"),
    ("Shaking", "(b) Shaking"),
    ("Freeze", "(c) Freeze"),
    ("FastSlowFast", "(d) Cycle"),
    ("Dangle", "(e) Dangle"),
    ("Slow", "(f) Slow"),
]

COLORS = {
    "nonrigid_acc": "#2B6CB0",
    "rigid_acc": "#C53030",
    "nonrigid_gyro": "#2F855A",
    "rigid_gyro": "#DD6B20",
    "fill_acc": "#7A8794",
    "fill_gyro": "#7A8794",
}


def resolve_data_path(path_like: str | Path, must_exist: bool = False) -> Path:
    """Resolve /mnt/data paths in both Linux and Windows/Codex workspaces."""
    path = Path(path_like)
    tried = [path]

    if path.exists():
        return path

    normalized = str(path_like).replace("\\", "/")
    if normalized.startswith("/mnt/data"):
        relative = normalized.removeprefix("/mnt/data").lstrip("/")
        tried.append(Path.cwd() / "mnt" / "data" / relative)
        tried.append(Path(__file__).resolve().parent / relative)
    else:
        tried.append(Path.cwd() / path)
        tried.append(Path(__file__).resolve().parent / path.name)

    for candidate in tried[1:]:
        if candidate.exists():
            return candidate

    if must_exist:
        tried_text = "\n  - ".join(str(item) for item in tried)
        raise FileNotFoundError(f"Could not find input file. Tried:\n  - {tried_text}")

    if normalized == "/mnt/data" or normalized.startswith("/mnt/data/"):
        relative = normalized.removeprefix("/mnt/data").lstrip("/")
        return Path.cwd() / "mnt" / "data" / relative
    return path


def configure_matplotlib() -> None:
    """Set restrained, journal-oriented matplotlib defaults."""
    available_fonts = {font.name for font in fm.fontManager.ttflist}
    preferred_font = "Arial" if "Arial" in available_fonts else "DejaVu Sans"

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [preferred_font, "Arial", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 10,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.8,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def read_sheet_data(excel_path: str | Path, sheet_name: str):
    """Read one sheet using fixed column positions, not potentially duplicated names."""
    excel_path = Path(excel_path)
    raw = pd.read_excel(excel_path, sheet_name=sheet_name, usecols=[0, 1, 2, 3, 4, 5])
    numeric = raw.apply(pd.to_numeric, errors="coerce").dropna(how="all")

    return (
        numeric.iloc[:, 0].to_numpy(dtype=float),
        numeric.iloc[:, 1].to_numpy(dtype=float),
        numeric.iloc[:, 2].to_numpy(dtype=float),
        numeric.iloc[:, 3].to_numpy(dtype=float),
        numeric.iloc[:, 4].to_numpy(dtype=float),
        numeric.iloc[:, 5].to_numpy(dtype=float),
    )


def _error_stats(nonrigid: np.ndarray, rigid: np.ndarray) -> tuple[float, float, float, float]:
    """Return RMSE, NRMSE, MAE, and peak absolute error for aligned signal pairs."""
    mask = np.isfinite(nonrigid) & np.isfinite(rigid)
    if not np.any(mask):
        return np.nan, np.nan, np.nan, np.nan

    error = nonrigid[mask] - rigid[mask]
    rmse = float(np.sqrt(np.mean(np.square(error))))
    mae = float(np.mean(np.abs(error)))
    peak_error = float(np.max(np.abs(error)))
    rigid_range = float(np.nanmax(rigid[mask]) - np.nanmin(rigid[mask]))
    nrmse = rmse / rigid_range if rigid_range > 0 else np.nan
    return rmse, nrmse, mae, peak_error


def compute_metrics(nonrigid_acc, rigid_acc, nonrigid_gyro, rigid_gyro) -> dict[str, float]:
    """Compute signal discrepancy metrics for acceleration and gyroscope traces."""
    nonrigid_acc = np.asarray(nonrigid_acc, dtype=float)
    rigid_acc = np.asarray(rigid_acc, dtype=float)
    nonrigid_gyro = np.asarray(nonrigid_gyro, dtype=float)
    rigid_gyro = np.asarray(rigid_gyro, dtype=float)

    acc_rmse, acc_nrmse, acc_mae, acc_peak_error = _error_stats(nonrigid_acc, rigid_acc)
    gyro_rmse, gyro_nrmse, gyro_mae, gyro_peak_error = _error_stats(
        nonrigid_gyro, rigid_gyro
    )
    return {
        "acc_rmse": acc_rmse,
        "acc_nrmse": acc_nrmse,
        "acc_mae": acc_mae,
        "gyro_rmse": gyro_rmse,
        "gyro_nrmse": gyro_nrmse,
        "gyro_mae": gyro_mae,
        "acc_peak_error": acc_peak_error,
        "gyro_peak_error": gyro_peak_error,
    }


def nice_ylim(values, margin: float = 0.08) -> tuple[float, float]:
    """Create a stable y-axis range with a small margin around finite values."""
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -1.0, 1.0

    ymin = float(np.nanmin(finite))
    ymax = float(np.nanmax(finite))
    span = ymax - ymin
    if span == 0:
        pad = max(abs(ymin) * margin, 1.0)
    else:
        pad = margin * span
    return ymin - pad, ymax + pad


def _finite_for_plot(time, signal_a, signal_b) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time = np.asarray(time, dtype=float)
    signal_a = np.asarray(signal_a, dtype=float)
    signal_b = np.asarray(signal_b, dtype=float)
    mask = np.isfinite(time) & np.isfinite(signal_a) & np.isfinite(signal_b)
    return time[mask], signal_a[mask], signal_b[mask]


def _style_axis(ax, show_ylabel: bool, ylabel: str | None = None) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3, width=0.7, pad=2)
    ax.grid(axis="y", color="#D0D0D0", alpha=0.25, linewidth=0.4)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune=None))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, prune=None))
    if show_ylabel and ylabel:
        ax.set_ylabel(ylabel)
        ax.yaxis.set_label_coords(-0.105, 0.5)


def plot_panel(fig, outer_spec: SubplotSpec, data: dict, title: str, show_ylabel: bool = False):
    """Draw one motion-condition panel with stacked acceleration and gyroscope axes."""
    inner = outer_spec.subgridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.08)
    ax_acc = fig.add_subplot(inner[0])
    ax_gyro = fig.add_subplot(inner[1], sharex=ax_acc)

    time_acc, nonrigid_acc, rigid_acc = _finite_for_plot(
        data["time_acc"], data["nonrigid_acc"], data["rigid_acc"]
    )
    time_gyro, nonrigid_gyro, rigid_gyro = _finite_for_plot(
        data["time_gyro"], data["nonrigid_gyro"], data["rigid_gyro"]
    )

    ax_acc.fill_between(
        time_acc,
        nonrigid_acc,
        rigid_acc,
        color=COLORS["fill_acc"],
        alpha=0.14,
        linewidth=0,
        zorder=1,
    )
    ax_acc.plot(
        time_acc,
        nonrigid_acc,
        color=COLORS["nonrigid_acc"],
        linestyle="-",
        linewidth=1.2,
        zorder=3,
    )
    ax_acc.plot(
        time_acc,
        rigid_acc,
        color=COLORS["rigid_acc"],
        linestyle="--",
        linewidth=1.2,
        zorder=4,
    )

    ax_gyro.fill_between(
        time_gyro,
        nonrigid_gyro,
        rigid_gyro,
        color=COLORS["fill_gyro"],
        alpha=0.14,
        linewidth=0,
        zorder=1,
    )
    ax_gyro.plot(
        time_gyro,
        nonrigid_gyro,
        color=COLORS["nonrigid_gyro"],
        linestyle="-",
        linewidth=1.1,
        zorder=3,
    )
    ax_gyro.plot(
        time_gyro,
        rigid_gyro,
        color=COLORS["rigid_gyro"],
        linestyle="--",
        linewidth=1.1,
        zorder=4,
    )

    ax_acc.set_ylim(nice_ylim(np.r_[nonrigid_acc, rigid_acc]))
    ax_gyro.set_ylim(nice_ylim(np.r_[nonrigid_gyro, rigid_gyro]))
    ax_acc.set_title(title, loc="left", fontweight="bold", pad=4)
    ax_gyro.set_xlabel("Time (s)")
    plt.setp(ax_acc.get_xticklabels(), visible=False)

    _style_axis(ax_acc, show_ylabel, "Acceleration (m/s$^2$)")
    _style_axis(ax_gyro, show_ylabel, "Gyroscope (rad/s)")
    return ax_acc, ax_gyro


def create_figure(excel_path: str | Path, output_dir: str | Path):
    """Read all sheets, create the multi-panel figure, and save figures plus metrics."""
    configure_matplotlib()
    excel_path = resolve_data_path(excel_path, must_exist=True)
    output_dir = resolve_data_path(output_dir, must_exist=False)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(8.0, 5.6), constrained_layout=False)
    outer = fig.add_gridspec(
        2,
        3,
        left=0.075,
        right=0.99,
        top=0.96,
        bottom=0.135,
        wspace=0.34,
        hspace=0.38,
    )

    metrics_rows = []
    for index, (sheet_name, panel_title) in enumerate(PANEL_CONFIG):
        (
            time_acc,
            nonrigid_acc,
            rigid_acc,
            time_gyro,
            nonrigid_gyro,
            rigid_gyro,
        ) = read_sheet_data(excel_path, sheet_name)
        data = {
            "time_acc": time_acc,
            "nonrigid_acc": nonrigid_acc,
            "rigid_acc": rigid_acc,
            "time_gyro": time_gyro,
            "nonrigid_gyro": nonrigid_gyro,
            "rigid_gyro": rigid_gyro,
        }

        row, col = divmod(index, 3)
        plot_panel(fig, outer[row, col], data, panel_title, show_ylabel=(col == 0))

        motion_condition = panel_title.split(") ", maxsplit=1)[1]
        metrics = compute_metrics(nonrigid_acc, rigid_acc, nonrigid_gyro, rigid_gyro)
        metrics_rows.append(
            {
                "motion_condition": motion_condition,
                "sheet_name": sheet_name,
                **metrics,
            }
        )

    legend_handles = [
        Line2D([0], [0], color=COLORS["nonrigid_acc"], lw=1.2, linestyle="-"),
        Line2D([0], [0], color=COLORS["rigid_acc"], lw=1.2, linestyle="--"),
        Line2D([0], [0], color=COLORS["nonrigid_gyro"], lw=1.1, linestyle="-"),
        Line2D([0], [0], color=COLORS["rigid_gyro"], lw=1.1, linestyle="--"),
    ]
    legend_labels = [
        "Non-rigid acceleration",
        "Rigid-reference acceleration",
        "Non-rigid gyroscope",
        "Rigid-reference gyroscope",
    ]
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=4,
        frameon=False,
        handlelength=2.4,
        columnspacing=1.5,
    )

    base = output_dir / "figure1_representative_imu_signals"
    png_path = base.with_suffix(".png")
    pdf_path = base.with_suffix(".pdf")
    svg_path = base.with_suffix(".svg")
    csv_path = output_dir / "figure1_signal_error_metrics.csv"

    fig.savefig(png_path, dpi=600, bbox_inches="tight", transparent=False, facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", transparent=False, facecolor="white")
    fig.savefig(svg_path, bbox_inches="tight", transparent=False, facecolor="white")
    plt.close(fig)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    return {
        "png": png_path,
        "pdf": pdf_path,
        "svg": svg_path,
        "metrics_csv": csv_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a multi-panel rigid vs non-rigid IMU signal figure."
    )
    parser.add_argument(
        "--excel-path",
        default="/mnt/data/原始绘图数据.xlsx",
        help="Path to the Excel workbook containing the six IMU signal sheets.",
    )
    parser.add_argument(
        "--output-dir",
        default="/mnt/data",
        help="Directory for exported figure files and the metrics CSV.",
    )
    args = parser.parse_args()

    outputs = create_figure(args.excel_path, args.output_dir)
    print("Generated files:")
    for key, value in outputs.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
