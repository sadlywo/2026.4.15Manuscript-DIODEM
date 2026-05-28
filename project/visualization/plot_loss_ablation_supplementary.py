from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


VARIANT_ORDER = [
    "Full model",
    "MSE only",
    "w/o spectral loss",
    "w/o attachment latent",
]

VARIANT_LABELS = {
    "Full model": "Full model",
    "MSE only": "MSE only",
    "w/o spectral loss": "w/o spectral",
    "w/o attachment latent": "w/o attach. latent",
}

CONFIG_COLUMNS = ["Latent", "L1", "MSE", "Spectral"]

VARIANT_COLORS = {
    "Full model": "#0076B9",
    "MSE only": "#9B6AA8",
    "w/o spectral loss": "#9BCB78",
    "w/o attachment latent": "#D88B69",
}


def _style() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.linewidth": 0.72,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 2.4,
            "ytick.major.size": 2.4,
            "legend.frameon": False,
        }
    )


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.10,
        1.055,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        fontweight="bold",
    )


def _parse_percent(value: object) -> float:
    match = re.search(r"[-+]?\d*\.?\d+", str(value))
    if not match:
        return np.nan
    return float(match.group(0))


def _parse_metric_mean_std(value: object) -> tuple[float, float]:
    numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(value))
    if not numbers:
        return np.nan, 0.0
    mean = float(numbers[0])
    std = float(numbers[1]) if len(numbers) > 1 else 0.0
    return mean, std


def _padded_limits(values: pd.Series, errors: pd.Series | None = None, pad_fraction: float = 0.10) -> tuple[float, float]:
    values_array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if errors is None:
        errors_array = np.zeros_like(values_array)
    else:
        errors_array = pd.to_numeric(errors, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    lower_values = values_array - errors_array
    upper_values = values_array + errors_array
    lower = float(np.nanmin(lower_values))
    upper = float(np.nanmax(upper_values))
    span = upper - lower
    if not np.isfinite(span) or span <= 0:
        span = max(abs(upper), 1.0) * 0.10
    pad = span * pad_fraction
    return lower - pad, upper + pad


def _load_table(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["Variant"] = pd.Categorical(frame["Variant"], categories=VARIANT_ORDER, ordered=True)
    frame = frame.sort_values("Variant").reset_index(drop=True)
    frame["Variant label"] = frame["Variant"].map(VARIANT_LABELS).astype(str)
    frame["Delta RMSE %"] = frame["Delta RMSE vs Full"].map(_parse_percent)
    frame["Delta PSD %"] = frame["Delta PSD vs Full"].map(_parse_percent)
    frame["Delta HF %"] = frame["Delta HF vs Full"].map(_parse_percent)
    frame["HF loss %"] = -frame["Delta HF %"]
    for column in ["RMSE", "PSD Dist.", "HF Improve."]:
        parsed = frame[column].map(_parse_metric_mean_std)
        frame[f"{column} Mean"] = parsed.map(lambda item: item[0])
        frame[f"{column} Std"] = parsed.map(lambda item: item[1])
    return frame


def _plot_configuration(ax: plt.Axes, frame: pd.DataFrame) -> None:
    matrix = frame[CONFIG_COLUMNS].eq("Y").astype(float).to_numpy()
    cmap = LinearSegmentedColormap.from_list("config", ["#F1F3F5", "#0076B9"])
    ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_title("Model and loss configuration", loc="left", fontsize=8, fontweight="bold")
    ax.set_xticks(np.arange(len(CONFIG_COLUMNS)))
    ax.set_xticklabels(CONFIG_COLUMNS, rotation=45, ha="right", fontsize=6)
    ax.set_yticks(np.arange(len(frame)))
    ax.set_yticklabels(frame["Variant label"], fontsize=6.2)
    ax.tick_params(length=0)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            text = "Y" if matrix[i, j] > 0.5 else "N"
            color = "white" if matrix[i, j] > 0.5 else "#6B7280"
            ax.text(j, i, text, ha="center", va="center", fontsize=5.4, color=color, fontweight="bold")
    ax.set_xticks(np.arange(-0.5, len(CONFIG_COLUMNS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(frame), 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.8)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _plot_rmse(ax: plt.Axes, frame: pd.DataFrame) -> None:
    y = np.arange(len(frame))[::-1]
    values = frame["RMSE Mean"].to_numpy(dtype=float)
    errors = frame["RMSE Std"].to_numpy(dtype=float)
    colors = [VARIANT_COLORS[str(v)] for v in frame["Variant"]]
    bars = ax.barh(y, values, color=colors, edgecolor="#263238", linewidth=0.35, height=0.58, zorder=3)
    if np.any(errors > 0):
        ax.errorbar(values, y, xerr=errors, fmt="none", ecolor="#263238", elinewidth=0.65, capsize=1.8, zorder=4)
    bars[0].set_edgecolor("black")
    bars[0].set_linewidth(0.85)
    full_rmse = float(frame.loc[frame["Variant"].astype(str) == "Full model", "RMSE Mean"].iloc[0])
    ax.axvline(full_rmse, color="#263238", lw=0.85, ls=(0, (2, 2)), zorder=2)
    for yi, (_, row) in zip(y, frame.iterrows()):
        delta = float(row["Delta RMSE %"])
        label = f"{row['RMSE Mean']:.4f}"
        if str(row["Variant"]) != "Full model":
            label += f" ({delta:+.2f}%)"
        ax.text(row["RMSE Mean"] + row["RMSE Std"] + 0.0010, yi, label, va="center", ha="left", fontsize=5.5, color="#263238")
    ax.set_yticks(y)
    ax.set_yticklabels(frame["Variant label"], fontsize=6.2)
    ax.set_xlabel("RMSE (lower is better)")
    ax.set_title("Primary reconstruction error", loc="left", fontsize=8, fontweight="bold")
    ax.grid(axis="x", color="#DDE3EA", lw=0.55, ls=(0, (2.0, 2.6)), zorder=0)
    left, right = _padded_limits(frame["RMSE Mean"], frame["RMSE Std"], pad_fraction=0.18)
    ax.set_xlim(max(0.0, left), right * 1.06)


def _plot_spectral_scatter(ax: plt.Axes, frame: pd.DataFrame) -> None:
    for _, row in frame.iterrows():
        variant = str(row["Variant"])
        marker = "*" if variant == "Full model" else "o"
        size = 92 if variant == "Full model" else 42
        ax.scatter(
            row["PSD Dist. Mean"],
            row["HF Improve. Mean"],
            s=size,
            marker=marker,
            color=VARIANT_COLORS[variant],
            edgecolor="#263238",
            linewidth=0.45,
            zorder=4,
        )
        if float(row["PSD Dist. Std"]) > 0 or float(row["HF Improve. Std"]) > 0:
            ax.errorbar(
                row["PSD Dist. Mean"],
                row["HF Improve. Mean"],
                xerr=row["PSD Dist. Std"],
                yerr=row["HF Improve. Std"],
                fmt="none",
                ecolor="#263238",
                elinewidth=0.55,
                capsize=1.6,
                zorder=3,
            )
    offsets = {
        "Full model": (0.00005, 0.035),
        "MSE only": (0.00006, -0.025),
        "w/o spectral loss": (0.00004, -0.055),
        "w/o attachment latent": (0.00006, 0.030),
    }
    for _, row in frame.iterrows():
        dx, dy = offsets[str(row["Variant"])]
        ax.text(
            row["PSD Dist. Mean"] + dx,
            row["HF Improve. Mean"] + dy,
            row["Variant label"],
            fontsize=5.5,
            color="#263238",
            ha="left",
            va="center",
        )
    ax.set_xlabel("PSD distance (lower)")
    ax.set_ylabel("HF improvement (higher)")
    ax.set_title("Spectral fidelity and high-frequency suppression", loc="left", fontsize=8, fontweight="bold")
    ax.grid(color="#DDE3EA", lw=0.55, ls=(0, (2.0, 2.6)), zorder=0)
    x_left, x_right = _padded_limits(frame["PSD Dist. Mean"], frame["PSD Dist. Std"], pad_fraction=0.16)
    y_bottom, y_top = _padded_limits(frame["HF Improve. Mean"], frame["HF Improve. Std"], pad_fraction=0.20)
    ax.set_xlim(x_left, x_right)
    ax.set_ylim(y_bottom, y_top)
    ax.annotate(
        "better",
        xy=(x_left + 0.18 * (x_right - x_left), y_top - 0.18 * (y_top - y_bottom)),
        xytext=(x_left + 0.42 * (x_right - x_left), y_top - 0.06 * (y_top - y_bottom)),
        arrowprops={"arrowstyle": "->", "lw": 0.7, "color": "#4B5563"},
        fontsize=5.8,
        color="#4B5563",
    )


def _plot_spectral_delta_bars(ax: plt.Axes, frame: pd.DataFrame) -> None:
    plot_frame = frame.loc[frame["Variant"].astype(str) != "Full model"].copy().reset_index(drop=True)
    plot_frame["PSD degradation %"] = plot_frame["Delta PSD %"]
    plot_frame["HF degradation %"] = -plot_frame["Delta HF %"]
    y = np.arange(len(plot_frame))[::-1]
    offset = 0.18
    psd_color = "#A94F4F"
    hf_color = "#D8A24A"
    improve_color = "#2F7F5F"

    for yi, (_, row) in zip(y, plot_frame.iterrows()):
        psd_value = float(row["PSD degradation %"])
        hf_value = float(row["HF degradation %"])
        psd_bar_color = improve_color if psd_value < 0 else psd_color
        hf_bar_color = improve_color if hf_value < 0 else hf_color
        ax.barh(yi + offset, psd_value, height=0.28, color=psd_bar_color, edgecolor="#263238", linewidth=0.3, zorder=3)
        ax.barh(yi - offset, hf_value, height=0.28, color=hf_bar_color, edgecolor="#263238", linewidth=0.3, zorder=3)

        psd_align = "left" if psd_value >= 0 else "right"
        hf_align = "left" if hf_value >= 0 else "right"
        psd_pad = 0.35 if psd_value >= 0 else -0.35
        hf_pad = 0.35 if hf_value >= 0 else -0.35
        ax.text(psd_value + psd_pad, yi + offset, f"{psd_value:+.1f}%", va="center", ha=psd_align, fontsize=5.4, color="#263238")
        ax.text(hf_value + hf_pad, yi - offset, f"{hf_value:+.1f}%", va="center", ha=hf_align, fontsize=5.4, color="#263238")

    max_abs = float(np.nanmax(np.abs(plot_frame[["PSD degradation %", "HF degradation %"]].to_numpy(dtype=float))))
    x_limit = max(5.0, max_abs * 1.28)
    ax.axvline(0, color="#263238", lw=0.75, zorder=2)
    ax.set_xlim(-x_limit, x_limit)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_frame["Variant label"], fontsize=6.2)
    ax.set_xlabel("Change versus full model (%)")
    ax.set_title("Spectral and high-frequency degradation", loc="left", fontsize=8, fontweight="bold")
    ax.grid(axis="x", color="#DDE3EA", lw=0.55, ls=(0, (2.0, 2.6)), zorder=0)
    ax.text(-x_limit * 0.98, y[0] + 0.65, "improved", fontsize=5.6, color=improve_color, ha="left", va="center")
    ax.text(x_limit * 0.98, y[0] + 0.65, "worse", fontsize=5.6, color=psd_color, ha="right", va="center")
    handles = [
        Patch(facecolor=psd_color, edgecolor="#263238", label="PSD distance"),
        Patch(facecolor=hf_color, edgecolor="#263238", label="HF loss"),
        Patch(facecolor=improve_color, edgecolor="#263238", label="Improvement"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=5.5, handlelength=1.0, borderaxespad=0.2)


def _plot_delta_heatmap(ax: plt.Axes, frame: pd.DataFrame) -> None:
    heat_frame = frame.copy()
    heat = heat_frame[["Delta RMSE %", "Delta PSD %", "HF loss %"]].to_numpy(dtype=float)
    cmap = LinearSegmentedColormap.from_list(
        "degradation",
        ["#2F7F5F", "#F7F8F4", "#F4B35D", "#A94F4F"],
    )
    image = ax.imshow(heat, aspect="auto", cmap=cmap, vmin=-3.0, vmax=21.0)
    ax.set_title("Relative degradation versus full model", loc="left", fontsize=8, fontweight="bold")
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(["RMSE", "PSD dist.", "HF loss"], fontsize=6.2)
    ax.set_yticks(np.arange(len(frame)))
    ax.set_yticklabels(frame["Variant label"], fontsize=6.2)
    ax.tick_params(length=0)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            value = heat[i, j]
            color = "white" if value > 12 else "#263238"
            suffix = "%" if j < 3 else ""
            ax.text(j, i, f"{value:+.1f}{suffix}", ha="center", va="center", fontsize=5.5, color=color)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.018)
    cbar.ax.tick_params(labelsize=5.4, width=0.55, length=2)
    cbar.set_label("Change vs full model (%)", fontsize=5.6)


def make_figure(table_path: Path, output_dir: Path) -> dict[str, Path]:
    _style()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = _load_table(table_path)
    source_path = output_dir / "loss_ablation_supplementary_figure_source.csv"
    frame.to_csv(source_path, index=False)

    fig = plt.figure(figsize=(7.35, 4.55), constrained_layout=False)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], width_ratios=[1.0, 1.12], hspace=0.43, wspace=0.34)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    _plot_configuration(ax_a, frame)
    _plot_rmse(ax_b, frame)
    _plot_spectral_scatter(ax_c, frame)
    _plot_delta_heatmap(ax_d, frame)

    for label, axis in zip(["a", "b", "c", "d"], [ax_a, ax_b, ax_c, ax_d]):
        _panel_label(axis, label)

    handles = [
        Patch(facecolor=VARIANT_COLORS["Full model"], edgecolor="#263238", label="Full model"),
        Patch(facecolor="#74B9A2", edgecolor="#263238", label="Ablated variants"),
        Patch(facecolor=VARIANT_COLORS["MSE only"], edgecolor="#263238", label="MSE-only baseline"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.992),
        ncol=3,
        fontsize=6.2,
        handlelength=1.1,
        columnspacing=1.1,
    )
    fig.subplots_adjust(left=0.145, right=0.980, top=0.885, bottom=0.115)

    stem = output_dir / "loss_ablation_supplementary_nature"
    outputs = {
        "svg": stem.with_suffix(".svg"),
        "pdf": stem.with_suffix(".pdf"),
        "png": stem.with_suffix(".png"),
        "tiff": stem.with_suffix(".tiff"),
        "source": source_path,
    }
    fig.savefig(outputs["svg"], bbox_inches="tight")
    fig.savefig(outputs["pdf"], bbox_inches="tight")
    fig.savefig(outputs["png"], dpi=600, bbox_inches="tight")
    fig.savefig(outputs["tiff"], dpi=600, bbox_inches="tight")
    plt.close(fig)
    return outputs


def make_two_panel_figure(table_path: Path, output_dir: Path) -> dict[str, Path]:
    _style()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = _load_table(table_path)
    source_path = output_dir / "loss_ablation_two_panel_figure_source.csv"
    frame.to_csv(source_path, index=False)

    fig = plt.figure(figsize=(7.20, 2.85), constrained_layout=False)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.08, 1.0], wspace=0.34)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])

    _plot_rmse(ax_a, frame)
    _plot_spectral_delta_bars(ax_b, frame)
    _panel_label(ax_a, "a")
    _panel_label(ax_b, "b")

    handles = [
        Patch(facecolor=VARIANT_COLORS["Full model"], edgecolor="#263238", label="Full model"),
        Patch(facecolor="#74B9A2", edgecolor="#263238", label="Ablated variants"),
        Patch(facecolor=VARIANT_COLORS["MSE only"], edgecolor="#263238", label="MSE-only baseline"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.995),
        ncol=3,
        fontsize=6.3,
        handlelength=1.1,
        columnspacing=1.1,
    )
    fig.subplots_adjust(left=0.135, right=0.985, top=0.775, bottom=0.215)

    stem = output_dir / "loss_ablation_two_panel_nature"
    outputs = {
        "svg": stem.with_suffix(".svg"),
        "pdf": stem.with_suffix(".pdf"),
        "png": stem.with_suffix(".png"),
        "tiff": stem.with_suffix(".tiff"),
        "source": source_path,
    }
    fig.savefig(outputs["svg"], bbox_inches="tight")
    fig.savefig(outputs["pdf"], bbox_inches="tight")
    fig.savefig(outputs["png"], dpi=600, bbox_inches="tight")
    fig.savefig(outputs["tiff"], dpi=600, bbox_inches="tight")
    plt.close(fig)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a Nature-style supplementary loss ablation figure.")
    parser.add_argument(
        "--table",
        type=Path,
        default=Path("outputs/loss_ablation/supplementary_loss_ablation_table.csv"),
        help="Input supplementary loss ablation table.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/loss_ablation/figures"),
        help="Directory for exported figure files.",
    )
    parser.add_argument(
        "--layout",
        choices=["full", "two-panel"],
        default="full",
        help="Export the original four-panel supplementary figure or the compact two-panel version.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.layout == "two-panel":
        outputs = make_two_panel_figure(args.table, args.output_dir)
    else:
        outputs = make_figure(args.table, args.output_dir)
    print("Saved loss ablation supplementary figure:")
    for kind, path in outputs.items():
        print(f"  {kind}: {path}")


if __name__ == "__main__":
    main()
