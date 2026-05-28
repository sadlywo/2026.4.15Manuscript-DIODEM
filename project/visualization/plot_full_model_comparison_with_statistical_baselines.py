from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


METHOD_ORDER = [
    "TCN-causal",
    "Transformer-causal",
    "GRU-causal",
    "LSTM-causal",
    "MLP-causal",
    "Identity/raw",
    "Moving-average low-pass",
    "Butterworth low-pass",
    "Savitzky-Golay",
    "Wiener filter",
]

SETTING_ORDER = ["by_experiment", "by_motion_type", "anomaly_test_only"]
SETTING_LABELS = {
    "by_experiment": "Experiment split",
    "by_motion_type": "Motion-type split",
    "anomaly_test_only": "Anomaly-only test",
}

SHORT_METHOD_LABELS = {
    "TCN-causal": "TCN",
    "Transformer-causal": "Transformer",
    "GRU-causal": "GRU",
    "LSTM-causal": "LSTM",
    "MLP-causal": "MLP",
    "Identity/raw": "Raw",
    "Moving-average low-pass": "Moving avg.",
    "Butterworth low-pass": "Butterworth",
    "Savitzky-Golay": "Savitzky",
    "Wiener filter": "Wiener",
}

METHOD_COLORS = {
    "TCN-causal": "#0B4F6C",
    "Transformer-causal": "#7566A0",
    "GRU-causal": "#3F7F5F",
    "LSTM-causal": "#B66A55",
    "MLP-causal": "#6F7885",
    "Identity/raw": "#2B2B2B",
    "Moving-average low-pass": "#A9ADB2",
    "Butterworth low-pass": "#BDAE82",
    "Savitzky-Golay": "#8EA8B8",
    "Wiener filter": "#C7959B",
}

METHOD_MARKERS = {
    "TCN-causal": "D",
    "Transformer-causal": "o",
    "GRU-causal": "o",
    "LSTM-causal": "o",
    "MLP-causal": "o",
    "Identity/raw": "s",
    "Moving-average low-pass": "s",
    "Butterworth low-pass": "s",
    "Savitzky-Golay": "s",
    "Wiener filter": "s",
}


def _style() -> None:
    # Nature/high-impact compatible editable-vector defaults.
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
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "legend.frameon": False,
        }
    )


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.135,
        1.075,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )


def _load_table(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["setting"] = pd.Categorical(frame["setting"], categories=SETTING_ORDER, ordered=True)
    frame["method"] = pd.Categorical(frame["method"], categories=METHOD_ORDER, ordered=True)
    return frame.sort_values(["setting", "method"]).reset_index(drop=True)


def _safe_error(value: float) -> float:
    if pd.isna(value) or not np.isfinite(value):
        return 0.0
    return max(float(value), 0.0)


def _plot_rmse_forest(ax: plt.Axes, frame: pd.DataFrame) -> None:
    y_positions: list[float] = []
    y_labels: list[str] = []
    y = 0.0
    group_centers: list[tuple[str, float]] = []

    for setting in SETTING_ORDER:
        start_y = y
        subset = frame[frame["setting"] == setting]
        for method in METHOD_ORDER:
            row = subset[subset["method"] == method].iloc[0]
            color = METHOD_COLORS[method]
            marker = METHOD_MARKERS[method]
            ax.errorbar(
                row["rmse_mean"],
                y,
                xerr=_safe_error(row["rmse_std"]),
                fmt=marker,
                mfc=color,
                mec="black",
                mew=0.35,
                color=color,
                ecolor=color,
                markersize=4.2 if method == "TCN-causal" else 3.4,
                elinewidth=1.0,
                capsize=2.4,
                capthick=0.8,
                zorder=4,
            )
            if method in {"TCN-causal", "Identity/raw", "Moving-average low-pass"}:
                ax.text(
                    row["rmse_mean"] + 0.018,
                    y,
                    f"{row['rmse_mean']:.3f} +/- {row['rmse_std']:.3f}",
                    va="center",
                    ha="left",
                    fontsize=5.1,
                    color="#2E3440",
                )
            y_positions.append(y)
            y_labels.append(SHORT_METHOD_LABELS[method])
            y += 1.0
        center = (start_y + y - 1.0) / 2.0
        group_centers.append((setting, center))
        if setting != SETTING_ORDER[-1]:
            ax.axhline(y - 0.45, color="#D7DDE5", lw=0.72, zorder=1)
            y += 1.1

    for idx, (setting, center) in enumerate(group_centers):
        if idx % 2 == 0:
            ax.axhspan(center - 5.0, center + 5.0, color="#F7F9FB", zorder=0)
        ax.text(
            -0.115,
            center,
            SETTING_LABELS[setting],
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            rotation=90,
            fontsize=6.0,
            fontweight="bold",
            color="#3B4252",
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=5.7)
    ax.invert_yaxis()
    ax.set_xlabel("RMSE (m/s^2 and rad/s combined, lower is better)")
    ax.set_title("Compensation error across evaluation settings", loc="left", fontsize=8.2, fontweight="bold")
    ax.grid(axis="x", color="#E1E6EC", lw=0.62)
    ax.set_xlim(0.34, 1.27)
    ax.text(
        0.99,
        0.025,
        "error bars: seed s.d.",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.5,
        color="#5B6470",
    )


def _plot_frequency_tradeoff(ax: plt.Axes, frame: pd.DataFrame) -> None:
    subset = frame[frame["setting"] == "by_experiment"]
    for method in METHOD_ORDER:
        row = subset[subset["method"] == method].iloc[0]
        ax.errorbar(
            row["psd_distance_mean"],
            row["hf_improvement_mean"],
            xerr=_safe_error(row["psd_distance_std"]),
            yerr=_safe_error(row["hf_improvement_std"]),
            fmt=METHOD_MARKERS[method],
            color=METHOD_COLORS[method],
            ecolor=METHOD_COLORS[method],
            mfc=METHOD_COLORS[method],
            mec="black",
            mew=0.35,
            markersize=4.1 if method == "TCN-causal" else 3.5,
            elinewidth=0.9,
            capsize=2.2,
            capthick=0.75,
            zorder=4 if method == "TCN-causal" else 3,
        )

    label_offsets = {
        "TCN-causal": (0.012, 0.10),
        "MLP-causal": (0.012, -0.20),
        "Identity/raw": (0.010, -0.18),
        "Moving-average low-pass": (0.010, 0.18),
        "Wiener filter": (0.010, -0.18),
    }
    for method in label_offsets:
        row = subset[subset["method"] == method].iloc[0]
        dx, dy = label_offsets[method]
        ax.text(
            row["psd_distance_mean"] + dx,
            row["hf_improvement_mean"] + dy,
            SHORT_METHOD_LABELS[method],
            fontsize=5.2,
            color="#2E3440",
            ha="left",
            va="center",
        )

    ax.annotate(
        "causal sequence\nmodels",
        xy=(0.058, 8.22),
        xytext=(0.083, 8.55),
        arrowprops={"arrowstyle": "-", "lw": 0.65, "color": "#667085"},
        fontsize=5.3,
        color="#5B6470",
        ha="left",
        va="center",
    )
    ax.annotate(
        "classical filters",
        xy=(0.125, 8.02),
        xytext=(0.151, 8.43),
        arrowprops={"arrowstyle": "-", "lw": 0.65, "color": "#667085"},
        fontsize=5.3,
        color="#5B6470",
        ha="left",
        va="center",
    )
    ax.annotate(
        "better",
        xy=(0.057, 8.30),
        xytext=(0.115, 7.60),
        arrowprops={"arrowstyle": "->", "lw": 0.7, "color": "#4C566A"},
        fontsize=5.8,
        color="#4C566A",
    )
    ax.set_xlabel("PSD distance (lower)")
    ax.set_ylabel("HF improvement (higher)")
    ax.set_title("Frequency-domain behavior", loc="left", fontsize=8.2, fontweight="bold")
    ax.grid(color="#E1E6EC", lw=0.58)
    ax.set_xlim(0.035, 0.245)
    ax.set_ylim(1.0, 8.75)
    ax.text(
        0.02,
        0.03,
        "Experiment split",
        transform=ax.transAxes,
        fontsize=5.6,
        color="#5B6470",
    )


def _plot_deployment_tradeoff(ax: plt.Axes, frame: pd.DataFrame) -> None:
    subset = frame[frame["setting"] == "by_experiment"].copy()
    subset["plot_size"] = np.sqrt(subset["params_k"].clip(lower=0.0) + 12.0) * 4.0
    for method in METHOD_ORDER:
        row = subset[subset["method"] == method].iloc[0]
        ax.errorbar(
            row["cpu_window_ms_mean"],
            row["rmse_mean"],
            xerr=_safe_error(row["cpu_window_ms_std"]),
            yerr=_safe_error(row["rmse_std"]),
            fmt=METHOD_MARKERS[method],
            color=METHOD_COLORS[method],
            ecolor=METHOD_COLORS[method],
            mfc=METHOD_COLORS[method],
            mec="black",
            mew=0.35,
            markersize=max(3.2, min(8.8, float(row["plot_size"]) / 2.6)),
            elinewidth=0.9,
            capsize=2.2,
            capthick=0.75,
            alpha=0.94,
            zorder=4 if method == "TCN-causal" else 3,
        )

    label_offsets = {
        "TCN-causal": (0.58, -0.018),
        "MLP-causal": (1.30, 0.018),
        "Identity/raw": (1.45, -0.020),
        "Moving-average low-pass": (1.35, -0.008),
        "Wiener filter": (1.23, -0.004),
    }
    for method in label_offsets:
        row = subset[subset["method"] == method].iloc[0]
        xmul, dy = label_offsets[method]
        ax.text(
            row["cpu_window_ms_mean"] * xmul,
            row["rmse_mean"] + dy,
            SHORT_METHOD_LABELS[method],
            fontsize=5.1,
            color="#2E3440",
            ha="left",
            va="center",
        )

    ax.annotate(
        "GRU/LSTM/\nTransformer",
        xy=(23.0, 0.415),
        xytext=(4.8, 0.455),
        arrowprops={"arrowstyle": "-", "lw": 0.65, "color": "#667085"},
        fontsize=5.2,
        color="#5B6470",
        ha="left",
        va="center",
    )
    ax.annotate(
        "filter cluster",
        xy=(1.2, 0.585),
        xytext=(0.18, 0.640),
        arrowprops={"arrowstyle": "-", "lw": 0.65, "color": "#667085"},
        fontsize=5.2,
        color="#5B6470",
        ha="left",
        va="center",
    )
    ax.set_xscale("log")
    ax.set_xlabel("CPU forward time (ms/window, log)")
    ax.set_ylabel("RMSE (lower)")
    ax.set_title("Accuracy-latency trade-off", loc="left", fontsize=8.2, fontweight="bold")
    ax.grid(color="#E1E6EC", lw=0.58, which="both")
    ax.set_xlim(8e-4, 60)
    ax.set_ylim(0.34, 0.88)
    ax.text(
        0.02,
        0.96,
        "Size: parameters",
        transform=ax.transAxes,
        fontsize=5.5,
        color="#5B6470",
        va="top",
    )


def _p_to_star(value: object) -> str:
    try:
        p = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _plot_statistical_gap(ax: plt.Axes, frame: pd.DataFrame) -> None:
    methods = [method for method in METHOD_ORDER if method != "TCN-causal"]
    heat = (
        frame[frame["method"].isin(methods)]
        .pivot(index="method", columns="setting", values="rmse_gap_vs_tcn_causal_percent")
        .reindex(index=methods, columns=SETTING_ORDER)
    )
    p_values = (
        frame[frame["method"].isin(methods)]
        .pivot(index="method", columns="setting", values="rmse_wilcoxon_p_vs_tcn_causal")
        .reindex(index=methods, columns=SETTING_ORDER)
    )
    cmap = LinearSegmentedColormap.from_list(
        "rmse_gap",
        ["#F7FBFF", "#DDEBF7", "#9BBBD5", "#D2A679", "#9E4F3F"],
    )
    image = ax.imshow(heat.values, aspect="auto", cmap=cmap, vmin=0, vmax=115)
    for i, method in enumerate(methods):
        for j, setting in enumerate(SETTING_ORDER):
            value = heat.loc[method, setting]
            star = _p_to_star(p_values.loc[method, setting])
            text_color = "white" if value > 72 else "#1F2933"
            ax.text(
                j,
                i,
                f"+{value:.0f}%\n{star}",
                ha="center",
                va="center",
                fontsize=5.3,
                color=text_color,
            )

    ax.set_xticks(np.arange(len(SETTING_ORDER)))
    ax.set_xticklabels(["Exp.", "Motion", "Anomaly"], fontsize=5.7)
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels([SHORT_METHOD_LABELS[m] for m in methods], fontsize=5.6)
    ax.set_title("Relative error gap and paired test", loc="left", fontsize=8.2, fontweight="bold")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = ax.figure.colorbar(image, ax=ax, fraction=0.045, pad=0.018)
    cbar.ax.tick_params(labelsize=5.4, width=0.55, length=2)
    cbar.set_label("RMSE increase vs TCN-causal (%)", fontsize=5.6)
    ax.text(
        0.0,
        -0.18,
        "Wilcoxon paired test over motion groups: * p<0.05, ** p<0.01, *** p<0.001; ns, not significant.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.2,
        color="#5B6470",
    )


def make_figure(table_path: Path, output_dir: Path) -> dict[str, Path]:
    _style()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = _load_table(table_path)

    source_path = output_dir / "full_model_comparison_with_statistical_baselines_figure_source.csv"
    frame.to_csv(source_path, index=False)

    fig = plt.figure(figsize=(7.40, 6.95), constrained_layout=False)
    grid = fig.add_gridspec(
        nrows=2,
        ncols=4,
        height_ratios=[1.56, 1.0],
        width_ratios=[1.05, 1.05, 1.15, 1.05],
        hspace=0.50,
        wspace=0.62,
    )

    ax_a = fig.add_subplot(grid[0, :])
    ax_b = fig.add_subplot(grid[1, 0])
    ax_c = fig.add_subplot(grid[1, 1])
    ax_d = fig.add_subplot(grid[1, 2:])

    _plot_rmse_forest(ax_a, frame)
    _plot_frequency_tradeoff(ax_b, frame)
    _plot_deployment_tradeoff(ax_c, frame)
    _plot_statistical_gap(ax_d, frame)

    for label, axis in zip(["a", "b", "c", "d"], [ax_a, ax_b, ax_c, ax_d]):
        _panel_label(axis, label)

    fig.text(
        0.008,
        0.005,
        "All quantitative error bars denote mean +/- s.d. across random seeds. "
        "Classical deterministic filters have zero seed s.d.; statistical comparison uses paired Wilcoxon tests versus TCN-causal.",
        ha="left",
        va="bottom",
        fontsize=5.4,
        color="#5B6470",
    )
    fig.subplots_adjust(left=0.135, right=0.985, top=0.975, bottom=0.100)

    stem = output_dir / "full_model_comparison_with_statistical_baselines_nature"
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
    parser = argparse.ArgumentParser(
        description="Create a Nature-style full model comparison figure with statistical baselines."
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=Path("outputs/causal_model_comparison/tables/full_model_comparison_with_statistical_baselines_numeric.csv"),
        help="Numeric comparison table produced by build_full_model_comparison_table.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/causal_model_comparison/figures"),
        help="Directory for exported figure files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = make_figure(args.table, args.output_dir)
    print("Saved Nature-style full comparison figure:")
    for kind, path in outputs.items():
        print(f"  {kind}: {path}")


if __name__ == "__main__":
    main()
