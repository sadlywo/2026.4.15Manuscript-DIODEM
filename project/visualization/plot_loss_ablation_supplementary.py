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
    "w/o L1 loss",
    "w/o MSE loss",
    "w/o spectral loss",
    "MSE only",
    "w/o attachment latent",
]

CORE_TWO_PANEL_ORDER = [
    "Full model",
    "w/o spectral loss",
    "w/o attachment latent",
    "MSE only",
]

SUMMARY_VARIANT_LABELS = {
    "full_model": "Full model",
    "no_l1_loss": "w/o L1 loss",
    "no_mse_loss": "w/o MSE loss",
    "no_spectral_loss": "w/o spectral loss",
    "mse_only": "MSE only",
    "no_attachment_latent": "w/o attachment latent",
}

VARIANT_LABELS = {
    "Full model": "Full model",
    "w/o L1 loss": "w/o L1",
    "w/o MSE loss": "w/o MSE",
    "MSE only": "MSE only",
    "w/o spectral loss": "w/o spectral",
    "w/o attachment latent": "w/o attach. latent",
}

CONFIG_COLUMNS = ["Latent", "L1", "MSE", "Spectral"]

VARIANT_COLORS = {
    "Full model": "#82C61E",
    "w/o L1 loss": "#F49568",
    "w/o MSE loss": "#ED746A",
    "w/o spectral loss": "#DEAE8F",
    "MSE only": "#F2C879",
    "w/o attachment latent": "#DBAA77",
}

VARIANT_EDGE = "#1F2A2E"

VARIANT_MARKERS = {
    "Full model": "*",
    "w/o L1 loss": "o",
    "w/o MSE loss": "s",
    "w/o spectral loss": "o",
    "MSE only": "o",
    "w/o attachment latent": "o",
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
    if "variant_name" in frame.columns:
        return _load_summary_table(frame)
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


def _load_summary_table(frame: pd.DataFrame) -> pd.DataFrame:
    summary = frame.copy()
    summary["Variant"] = summary["variant_name"].map(SUMMARY_VARIANT_LABELS)
    summary = summary.dropna(subset=["Variant"]).copy()
    summary["Variant"] = pd.Categorical(summary["Variant"], categories=VARIANT_ORDER, ordered=True)
    summary = summary.sort_values("Variant").reset_index(drop=True)
    summary["Variant label"] = summary["Variant"].map(VARIANT_LABELS).astype(str)
    summary["Latent"] = np.where(pd.to_numeric(summary["attach_latent_dim"], errors="coerce") > 0, "Y", "N")
    summary["L1"] = np.where(pd.to_numeric(summary["time_l1"], errors="coerce") > 0, "Y", "N")
    summary["MSE"] = np.where(pd.to_numeric(summary["mse"], errors="coerce") > 0, "Y", "N")
    summary["Spectral"] = np.where(pd.to_numeric(summary["spectral"], errors="coerce") > 0, "Y", "N")
    rename_map = {
        "rmse_mean": "RMSE Mean",
        "rmse_std": "RMSE Std",
        "psd_distance_mean": "PSD Dist. Mean",
        "psd_distance_std": "PSD Dist. Std",
        "hf_ratio_improvement_mean": "HF Improve. Mean",
        "hf_ratio_improvement_std": "HF Improve. Std",
        "num_seeds": "Seeds",
    }
    summary = summary.rename(columns=rename_map)
    for column in [
        "RMSE Mean",
        "RMSE Std",
        "PSD Dist. Mean",
        "PSD Dist. Std",
        "HF Improve. Mean",
        "HF Improve. Std",
        "Seeds",
    ]:
        summary[column] = pd.to_numeric(summary[column], errors="coerce")

    full = summary.loc[summary["Variant"].astype(str) == "Full model"].iloc[0]
    summary["Delta RMSE %"] = (summary["RMSE Mean"] - full["RMSE Mean"]) / full["RMSE Mean"] * 100.0
    summary["Delta PSD %"] = (summary["PSD Dist. Mean"] - full["PSD Dist. Mean"]) / full["PSD Dist. Mean"] * 100.0
    summary["Delta HF %"] = (summary["HF Improve. Mean"] - full["HF Improve. Mean"]) / full["HF Improve. Mean"] * 100.0
    summary["HF loss %"] = -summary["Delta HF %"]
    summary["RMSE"] = summary.apply(lambda row: f"{row['RMSE Mean']:.4f} +/- {row['RMSE Std']:.4f}", axis=1)
    summary["PSD Dist."] = summary.apply(
        lambda row: f"{row['PSD Dist. Mean']:.5f} +/- {row['PSD Dist. Std']:.5f}",
        axis=1,
    )
    summary["HF Improve."] = summary.apply(
        lambda row: f"{row['HF Improve. Mean']:.3f} +/- {row['HF Improve. Std']:.3f}",
        axis=1,
    )
    return summary


def _filter_focus(frame: pd.DataFrame, focus: str) -> pd.DataFrame:
    if focus == "all":
        order = VARIANT_ORDER
    elif focus == "core":
        order = CORE_TWO_PANEL_ORDER
    else:
        raise ValueError(f"Unsupported focus mode: {focus}")
    focused = frame.loc[frame["Variant"].astype(str).isin(order)].copy()
    focused["Variant"] = pd.Categorical(focused["Variant"].astype(str), categories=order, ordered=True)
    return focused.sort_values("Variant").reset_index(drop=True)


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
    full_rmse = float(frame.loc[frame["Variant"].astype(str) == "Full model", "RMSE Mean"].iloc[0])
    ax.axvline(full_rmse, color="#263238", lw=0.85, ls=(0, (2, 2)), zorder=1)
    for yi, (_, row) in zip(y, frame.iterrows()):
        variant = str(row["Variant"])
        color = VARIANT_COLORS[variant]
        delta = float(row["Delta RMSE %"])
        ax.errorbar(
            row["RMSE Mean"],
            yi,
            xerr=row["RMSE Std"],
            fmt="o",
            ms=5.1 if variant != "Full model" else 6.2,
            color=color,
            ecolor=color,
            elinewidth=1.0,
            capsize=2.4,
            mec="white",
            mew=0.55,
            zorder=4,
        )
        label = f"{row['RMSE Mean']:.4f}"
        if variant != "Full model":
            label += f" ({delta:+.2f}%)"
        ax.text(row["RMSE Mean"] + row["RMSE Std"] + 0.0010, yi, label, va="center", ha="left", fontsize=5.6, color="#263238")
    ax.set_yticks(y)
    if "Seeds" in frame.columns:
        ytick_labels = [f"{row['Variant label']} (n={int(row['Seeds'])})" for _, row in frame.iterrows()]
    else:
        ytick_labels = frame["Variant label"].tolist()
    ax.set_yticklabels(ytick_labels, fontsize=6.2)
    ax.set_xlabel("RMSE (lower is better)")
    ax.set_title("Primary reconstruction error", loc="left", fontsize=8.2, fontweight="bold")
    ax.grid(axis="x", color="#DDE3EA", lw=0.55, ls=(0, (2.0, 2.6)), zorder=0)
    left, right = _padded_limits(frame["RMSE Mean"], frame["RMSE Std"], pad_fraction=0.18)
    span = right - left
    ax.set_xlim(max(0.0, left), right + span * 0.18)


def _variant_family(variant: str) -> str:
    if variant == "Full model":
        return "full"
    if variant == "MSE only":
        return "mse_only"
    return "ablated"


def _plot_rmse_bars(ax: plt.Axes, frame: pd.DataFrame) -> None:
    plot_frame = frame.reset_index(drop=True)
    values = plot_frame["RMSE Mean"].to_numpy(dtype=float)
    errors = plot_frame["RMSE Std"].to_numpy(dtype=float)
    lower, upper = _padded_limits(plot_frame["RMSE Mean"], plot_frame["RMSE Std"], pad_fraction=0.24)
    lower = max(0.0, np.floor(lower * 200.0) / 200.0)
    upper = np.ceil((upper + (upper - lower) * 0.10) * 200.0) / 200.0
    y = np.arange(len(plot_frame))
    full_rmse = float(plot_frame.loc[plot_frame["Variant"].astype(str) == "Full model", "RMSE Mean"].iloc[0])

    ax.axvline(full_rmse, color=VARIANT_EDGE, lw=0.9, ls=(0, (2.0, 2.2)), zorder=2)
    for yi, (_, row) in zip(y, plot_frame.iterrows()):
        variant = str(row["Variant"])
        color = VARIANT_COLORS[variant]
        value = float(row["RMSE Mean"])
        err = float(row["RMSE Std"])
        ax.barh(
            yi,
            value - lower,
            left=lower,
            height=0.58,
            color=color,
            edgecolor=VARIANT_EDGE,
            linewidth=0.72,
            zorder=3,
        )
        ax.errorbar(
            value,
            yi,
            xerr=err,
            fmt="none",
            ecolor=VARIANT_EDGE,
            elinewidth=0.72,
            capsize=2.1,
            capthick=0.72,
            zorder=4,
        )
        label = f"{value:.4f}"
        if variant != "Full model":
            label += f" ({float(row['Delta RMSE %']):+.2f}%)"
        ax.text(
            value + err + (upper - lower) * 0.030,
            yi,
            label,
            ha="left",
            va="center",
            fontsize=5.6,
            color="#263238",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(plot_frame["Variant label"], fontsize=6.4)
    ax.invert_yaxis()
    ax.set_xlim(lower, upper)
    ax.set_xlabel("RMSE (lower is better)")
    ax.set_title("Primary reconstruction error", loc="left", fontsize=8.2, fontweight="bold")
    ax.grid(axis="x", color="#DCE5ED", lw=0.55, ls=(0, (2.0, 2.6)), zorder=0)
    ax.tick_params(axis="y", length=0)


def _plot_metric_bars(
    ax: plt.Axes,
    frame: pd.DataFrame,
    mean_column: str,
    std_column: str,
    delta_column: str,
    reference_title: str,
    xlabel: str,
    *,
    digits: int,
    show_ylabels: bool,
) -> None:
    plot_frame = frame.reset_index(drop=True)
    lower, upper = _padded_limits(plot_frame[mean_column], plot_frame[std_column], pad_fraction=0.28)
    span = upper - lower
    lower = lower - span * 0.02
    upper = upper + span * 0.18
    y = np.arange(len(plot_frame))
    full_value = float(plot_frame.loc[plot_frame["Variant"].astype(str) == "Full model", mean_column].iloc[0])

    ax.axvline(full_value, color=VARIANT_EDGE, lw=0.85, ls=(0, (2.0, 2.2)), zorder=2)
    for yi, (_, row) in zip(y, plot_frame.iterrows()):
        variant = str(row["Variant"])
        value = float(row[mean_column])
        err = float(row[std_column])
        ax.barh(
            yi,
            value - lower,
            left=lower,
            height=0.56,
            color=VARIANT_COLORS[variant],
            edgecolor=VARIANT_EDGE,
            linewidth=0.68,
            zorder=3,
        )
        ax.errorbar(
            value,
            yi,
            xerr=err,
            fmt="none",
            ecolor=VARIANT_EDGE,
            elinewidth=0.70,
            capsize=2.0,
            capthick=0.70,
            zorder=4,
        )
        label = f"{value:.{digits}f}"
        if variant != "Full model":
            label += f" ({float(row[delta_column]):+.2f}%)"
        ax.text(
            value + err + (upper - lower) * 0.035,
            yi,
            label,
            ha="left",
            va="center",
            fontsize=5.1,
            color="#263238",
        )

    ax.set_yticks(y)
    if show_ylabels:
        ax.set_yticklabels(plot_frame["Variant label"], fontsize=6.0)
    else:
        ax.set_yticklabels([])
    ax.invert_yaxis()
    ax.set_xlim(lower, upper)
    ax.set_xlabel(xlabel, fontsize=6.2)
    ax.set_title(reference_title, loc="left", fontsize=7.2, fontweight="bold")
    ax.grid(axis="x", color="#DCE5ED", lw=0.52, ls=(0, (2.0, 2.6)), zorder=0)
    ax.tick_params(axis="y", length=0)


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
    x_left, x_right = _padded_limits(frame["PSD Dist. Mean"], frame["PSD Dist. Std"], pad_fraction=0.18)
    y_bottom, y_top = _padded_limits(frame["HF Improve. Mean"], frame["HF Improve. Std"], pad_fraction=0.22)
    x_span = x_right - x_left
    y_span = y_top - y_bottom
    offsets = {
        "Full model": (0.020 * x_span, 0.110 * y_span, "right"),
        "w/o L1 loss": (0.035 * x_span, 0.075 * y_span, "left"),
        "w/o MSE loss": (0.035 * x_span, -0.075 * y_span, "left"),
        "MSE only": (0.030 * x_span, -0.075 * y_span, "left"),
        "w/o spectral loss": (0.030 * x_span, -0.110 * y_span, "left"),
        "w/o attachment latent": (0.035 * x_span, -0.045 * y_span, "left"),
    }
    for _, row in frame.iterrows():
        dx, dy, ha = offsets.get(str(row["Variant"]), (0.030 * x_span, 0.030 * y_span, "left"))
        ax.text(
            row["PSD Dist. Mean"] + dx,
            row["HF Improve. Mean"] + dy,
            row["Variant label"],
            fontsize=5.6,
            color="#263238",
            ha=ha,
            va="center",
            fontweight="bold" if str(row["Variant"]) == "Full model" else "normal",
        )
    ax.set_xlabel("PSD distance (lower)")
    ax.set_ylabel("HF improvement (higher)")
    ax.set_title("Spectral fidelity and high-frequency suppression", loc="left", fontsize=8.2, fontweight="bold")
    ax.grid(color="#DDE3EA", lw=0.55, ls=(0, (2.0, 2.6)), zorder=0)
    ax.set_xlim(x_left, x_right)
    ax.set_ylim(y_bottom, y_top)


def _plot_spectral_hf_scatter(ax: plt.Axes, frame: pd.DataFrame) -> None:
    plot_frame = frame.reset_index(drop=True)
    draw_frame = pd.concat(
        [
            plot_frame.loc[plot_frame["Variant"].astype(str) != "Full model"],
            plot_frame.loc[plot_frame["Variant"].astype(str) == "Full model"],
        ],
        ignore_index=True,
    )
    for _, row in draw_frame.iterrows():
        variant = str(row["Variant"])
        marker = VARIANT_MARKERS.get(variant, "o")
        size = 136 if variant == "Full model" else 54
        ax.errorbar(
            row["PSD Dist. Mean"],
            row["HF Improve. Mean"],
            xerr=row["PSD Dist. Std"],
            yerr=row["HF Improve. Std"],
            fmt="none",
            ecolor=VARIANT_EDGE,
            elinewidth=0.62,
            capsize=1.9,
            capthick=0.62,
            zorder=3,
        )
        ax.scatter(
            row["PSD Dist. Mean"],
            row["HF Improve. Mean"],
            s=size,
            marker=marker,
            color=VARIANT_COLORS[variant],
            edgecolor=VARIANT_EDGE,
            linewidth=0.65,
            alpha=0.96,
            zorder=5 if variant == "Full model" else 4,
        )

    x_left, x_right = _padded_limits(plot_frame["PSD Dist. Mean"], plot_frame["PSD Dist. Std"], pad_fraction=0.26)
    y_bottom, y_top = _padded_limits(plot_frame["HF Improve. Mean"], plot_frame["HF Improve. Std"], pad_fraction=0.30)
    x_span = x_right - x_left
    y_span = y_top - y_bottom
    label_offsets = {
        "Full model": (-30, 38, "right"),
        "w/o L1 loss": (28, 36, "left"),
        "w/o MSE loss": (-34, -36, "right"),
        "w/o spectral loss": (40, -42, "left"),
        "w/o attachment latent": (34, 32, "left"),
        "MSE only": (28, -14, "left"),
    }
    for _, row in plot_frame.iterrows():
        variant = str(row["Variant"])
        dx, dy, ha = label_offsets.get(variant, (18, 12, "left"))
        ax.annotate(
            row["Variant label"],
            xy=(row["PSD Dist. Mean"], row["HF Improve. Mean"]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va="center",
            fontsize=5.6,
            color="#263238",
            fontweight="bold" if variant == "Full model" else "normal",
            bbox={"boxstyle": "square,pad=0.08", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
            arrowprops={
                "arrowstyle": "-",
                "color": "#54636A",
                "lw": 0.45,
                "shrinkA": 1,
                "shrinkB": 3,
            },
            zorder=6,
        )
    ax.set_xlim(x_left - 0.030 * x_span, x_right + 0.095 * x_span)
    ax.set_ylim(y_bottom - 0.020 * y_span, y_top + 0.055 * y_span)
    ax.set_xlabel("PSD distance (lower)")
    ax.set_ylabel("HF improvement (higher)")
    ax.set_title("Spectral fidelity and high-frequency suppression", loc="left", fontsize=8.2, fontweight="bold")
    ax.grid(color="#DCE5ED", lw=0.55, ls=(0, (2.0, 2.6)), zorder=0)

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
    ax.set_title("Spectral and high-frequency effects", loc="left", fontsize=8.2, fontweight="bold")
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

    fig = plt.figure(figsize=(7.35, 5.20), constrained_layout=False)
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
        bbox_to_anchor=(0.52, 0.95),
        ncol=3,
        fontsize=10,
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


def make_two_panel_figure(table_path: Path, output_dir: Path, focus: str = "core") -> dict[str, Path]:
    _style()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = _filter_focus(_load_table(table_path), focus=focus)
    source_path = output_dir / "loss_ablation_two_panel_figure_source.csv"
    frame.to_csv(source_path, index=False)

    fig = plt.figure(figsize=(8.45, 3.00), constrained_layout=False)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.02, 1.18], wspace=0.40)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])

    _plot_rmse_bars(ax_a, frame)
    _plot_spectral_hf_scatter(ax_b, frame)
    _panel_label(ax_a, "a")
    _panel_label(ax_b, "b")

    handles = [
        Patch(facecolor=VARIANT_COLORS["Full model"], edgecolor=VARIANT_EDGE, label="Full model"),
        Patch(facecolor=VARIANT_COLORS["w/o spectral loss"], edgecolor=VARIANT_EDGE, label="Ablated variants"),
        Patch(facecolor=VARIANT_COLORS["MSE only"], edgecolor=VARIANT_EDGE, label="MSE-only baseline"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.535, 0.80),
        ncol=3,
        fontsize=9,
        handlelength=1.05,
        columnspacing=1.0,
    )
    fig.subplots_adjust(left=0.100, right=0.985, top=0.800, bottom=0.200)

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


def make_three_panel_figure(table_path: Path, output_dir: Path, focus: str = "all") -> dict[str, Path]:
    _style()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = _filter_focus(_load_table(table_path), focus=focus)
    source_path = output_dir / "loss_ablation_three_panel_figure_source.csv"
    frame.to_csv(source_path, index=False)

    fig = plt.figure(figsize=(8.95, 3.15), constrained_layout=False)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.28, 1.0, 1.0], wspace=0.20)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[0, 2])

    _plot_metric_bars(
        ax_a,
        frame,
        "RMSE Mean",
        "RMSE Std",
        "Delta RMSE %",
        "Reconstruction error",
        "RMSE (lower)",
        digits=4,
        show_ylabels=True,
    )
    _plot_metric_bars(
        ax_b,
        frame,
        "PSD Dist. Mean",
        "PSD Dist. Std",
        "Delta PSD %",
        "Spectral fidelity",
        "PSD distance (lower)",
        digits=5,
        show_ylabels=False,
    )
    _plot_metric_bars(
        ax_c,
        frame,
        "HF Improve. Mean",
        "HF Improve. Std",
        "Delta HF %",
        "High-frequency suppression",
        "HF improvement (higher)",
        digits=3,
        show_ylabels=False,
    )
    for label, axis in zip(["a", "b", "c"], [ax_a, ax_b, ax_c]):
        _panel_label(axis, label)

    present_variants = set(frame["Variant"].astype(str))
    legend_variants = [variant for variant in VARIANT_ORDER if variant in present_variants]
    handles = [
        Patch(
            facecolor=VARIANT_COLORS[variant],
            edgecolor=VARIANT_EDGE,
            label=VARIANT_LABELS.get(variant, variant),
        )
        for variant in legend_variants
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.545, 1.005),
        ncol=min(6, max(1, len(handles))),
        fontsize=6.8,
        handlelength=1.05,
        columnspacing=0.72,
    )
    fig.subplots_adjust(left=0.090, right=0.988, top=0.860, bottom=0.205)

    stem = output_dir / "loss_ablation_three_panel_nature"
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
        default=Path("outputs/paper_tables/supplementary_loss_ablation_table.csv"),
        help="Input supplementary loss ablation table or raw ablation_summary.csv.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Raw ablation_summary.csv path. When supplied, it overrides --table.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/loss_ablation/figures"),
        help="Directory for exported figure files.",
    )
    parser.add_argument(
        "--layout",
        choices=["full", "two-panel", "three-panel"],
        default="three-panel",
        help="Export the original four-panel, compact two-panel, or separated three-panel figure.",
    )
    parser.add_argument(
        "--focus",
        choices=["core", "all"],
        default="all",
        help="Variant subset for the two-panel figure. 'all' shows the complete six-variant ablation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.summary if args.summary is not None else args.table
    if args.layout == "two-panel":
        outputs = make_two_panel_figure(input_path, args.output_dir, focus=args.focus)
    elif args.layout == "three-panel":
        outputs = make_three_panel_figure(input_path, args.output_dir, focus=args.focus)
    else:
        outputs = make_figure(input_path, args.output_dir)
    print("Saved loss ablation supplementary figure:")
    for kind, path in outputs.items():
        print(f"  {kind}: {path}")


if __name__ == "__main__":
    main()
