from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "matplotlib is required for plotting. Install it in the active environment "
        "with `pip install matplotlib` and rerun this script."
    ) from exc


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TABLE_DIR = ROOT / "outputs" / "paper_tables"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "paper_figures"

SETTING_ORDER = ["by_experiment", "anomaly_test_only", "by_motion_type"]
SETTING_LABELS = {
    "by_experiment": "By-experiment",
    "anomaly_test_only": "Anomaly\nTest-only",
    "by_motion_type": "By-motion-type",
}
MODEL_ORDER = ["transformer", "tcn_causal", "tcn", "gru", "lowpass"]
MODEL_LABELS = {
    "transformer": "Transformer",
    "tcn_causal": "TCN-causal",
    "tcn": "TCN",
    "gru": "GRU",
    "lowpass": "Lowpass",
    "butterworth": "Butterworth",
    "savgol": "Savitzky-Golay",
    "wiener": "Wiener",
    "identity": "Identity",
}
MODEL_COLORS = {
    "transformer": "#1f6f8b",
    "tcn_causal": "#e67e22",
    "tcn": "#4c78a8",
    "gru": "#59a14f",
    "lowpass": "#6c757d",
    "butterworth": "#8d99ae",
    "savgol": "#b07aa1",
    "wiener": "#9c755f",
    "identity": "#bab0ab",
}
MODEL_MARKERS = {
    "transformer": "o",
    "tcn_causal": "D",
    "tcn": "s",
    "gru": "^",
    "lowpass": "X",
}
PANEL_LABELS = ["(A)", "(B)", "(C)", "(D)"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create publication-style figures from paper result tables."
    )
    parser.add_argument(
        "--table-dir",
        type=Path,
        default=DEFAULT_TABLE_DIR,
        help="Directory containing the paper CSV tables.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the generated figures will be saved.",
    )
    args = parser.parse_args()

    table_dir = args.table_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    _apply_publication_style()

    comparison_rows = _load_csv_rows(table_dir / "table1_by_experiment_full_comparison.csv")
    generalization_rows = _load_csv_rows(table_dir / "table2_generalization_core_models.csv")
    deployment_rows = _load_csv_rows(table_dir / "table3_deployment_streaming_summary.csv")

    plot_by_experiment_comparison(comparison_rows, output_dir)
    plot_generalization_summary(generalization_rows, output_dir)
    plot_deployment_tradeoff(generalization_rows, deployment_rows, output_dir)
    plot_generalization_gap(generalization_rows, output_dir)

    print(f"Saved figures to {output_dir}")


def _apply_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 10,
            "axes.titlesize": 11.5,
            "axes.labelsize": 11,
            "axes.linewidth": 1.1,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9,
            "figure.titlesize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": "#d0d0d0",
            "grid.linewidth": 0.75,
            "grid.alpha": 0.7,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _load_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _parse_value_with_std(text: str) -> Tuple[float, float]:
    cleaned = (text or "").strip()
    if not cleaned:
        return math.nan, math.nan
    parts = re.split(r"\s*\+\-\s*", cleaned)
    if len(parts) == 1:
        return _to_float(parts[0]), 0.0
    return _to_float(parts[0]), _to_float(parts[1])


def _to_float(text: str) -> float:
    cleaned = (text or "").strip().replace(",", "")
    return float(cleaned)


def _save_figure(fig: plt.Figure, stem: str, output_dir: Path) -> None:
    fig.tight_layout(pad=0.8)
    for suffix in (".png", ".svg", ".pdf"):
        fig.savefig(output_dir / f"{stem}{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_by_experiment_comparison(rows: List[Dict[str, str]], output_dir: Path) -> None:
    rmse_rows = []
    psd_rows = []
    for row in rows:
        model_name = row["Model"]
        rmse_mean, rmse_std = _parse_value_with_std(row["RMSE"])
        psd_mean, psd_std = _parse_value_with_std(row["PSD Dist."])
        rmse_rows.append((model_name, rmse_mean, rmse_std))
        psd_rows.append((model_name, psd_mean, psd_std))

    rmse_rows.sort(key=lambda item: item[1])
    psd_rows.sort(key=lambda item: item[1])

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.6))
    for axis, metric_rows, x_label, panel_label in (
        (axes[0], rmse_rows, "RMSE (lower is better)", PANEL_LABELS[0]),
        (axes[1], psd_rows, "PSD distance (lower is better)", PANEL_LABELS[1]),
    ):
        y_labels = [MODEL_LABELS.get(name, name) for name, _, _ in metric_rows]
        y_positions = list(range(len(metric_rows)))
        for y_pos, (model_name, mean_value, std_value) in enumerate(metric_rows):
            color = MODEL_COLORS.get(model_name, "#808080")
            is_baseline = model_name in {"lowpass", "butterworth", "savgol", "wiener", "identity"}
            axis.barh(
                y_pos,
                mean_value,
                xerr=std_value,
                height=0.72,
                color=color if not is_baseline else "#d9dde3",
                edgecolor=color,
                linewidth=1.0,
                hatch="//" if is_baseline else None,
                error_kw={"elinewidth": 1.1, "capsize": 3.2, "capthick": 1.1, "ecolor": "#333333"},
                zorder=3,
            )
        axis.set_yticks(y_positions, y_labels)
        axis.invert_yaxis()
        axis.set_xlabel(x_label)
        _style_axis(axis, panel_label)
        axis.grid(True, axis="x")
        axis.xaxis.set_major_locator(MaxNLocator(nbins=6))

    axes[0].set_title("By-experiment model ranking")
    axes[1].set_title("Spectral consistency under by-experiment split")
    axes[0].text(
        0.98,
        0.08,
        "Transformer: best offline accuracy\nTCN-causal: best real-time candidate",
        transform=axes[0].transAxes,
        ha="right",
        va="bottom",
        fontsize=8.7,
        color="#333333",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#f7f7f7", "edgecolor": "#c7c7c7"},
    )
    _save_figure(fig, "figure1_by_experiment_comparison_ieee", output_dir)


def plot_generalization_summary(rows: List[Dict[str, str]], output_dir: Path) -> None:
    metrics = [
        ("RMSE", "RMSE", False),
        ("Pearson", "Pearson", True),
        ("PSD Dist.", "PSD distance", False),
        ("HF Improve.", "HF improvement", True),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 8.8), sharex=True)
    axes = axes.ravel()
    x_positions = list(range(len(SETTING_ORDER)))

    for panel_idx, (axis, (column_name, y_label, higher_is_better)) in enumerate(zip(axes, metrics)):
        for model_name in ["transformer", "tcn_causal", "tcn", "lowpass"]:
            series_mean: List[float] = []
            series_std: List[float] = []
            for setting_name in SETTING_ORDER:
                row = _find_row(rows, setting_name, model_name)
                mean_value, std_value = _parse_value_with_std(row[column_name])
                series_mean.append(mean_value)
                series_std.append(std_value)

            axis.errorbar(
                x_positions,
                series_mean,
                yerr=series_std,
                color=MODEL_COLORS[model_name],
                marker=MODEL_MARKERS[model_name],
                markerfacecolor="white",
                markeredgewidth=1.4,
                markersize=6.6,
                linewidth=1.9,
                elinewidth=1.2,
                capsize=3.6,
                label=MODEL_LABELS[model_name],
                zorder=4,
            )

        axis.set_title(y_label, pad=8)
        axis.set_ylabel(y_label)
        _style_axis(axis, PANEL_LABELS[panel_idx])
        axis.grid(True, axis="y")
        axis.set_xticks(x_positions, [SETTING_LABELS[item] for item in SETTING_ORDER])
        axis.tick_params(axis="x", pad=7)
        if higher_is_better:
            axis.annotate(
                "Higher is better",
                xy=(0.98, 0.06),
                xycoords="axes fraction",
                ha="right",
                va="bottom",
                fontsize=8.5,
                color="#666666",
            )
        else:
            axis.annotate(
                "Lower is better",
                xy=(0.98, 0.06),
                xycoords="axes fraction",
                ha="right",
                va="bottom",
                fontsize=8.5,
                color="#666666",
            )
        if column_name == "RMSE":
            axis.axvspan(1.5, 2.5, color="#f8f4ed", alpha=0.8, zorder=0)
            axis.annotate(
                "Largest gap under\nunseen motions",
                xy=(2.0, min(series_mean) + 0.03),
                xycoords="data",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color="#7a4f2b",
            )

    handles = [
        Line2D(
            [0],
            [0],
            color=MODEL_COLORS[name],
            marker=MODEL_MARKERS[name],
            linewidth=2.0,
            markersize=6.5,
            label=MODEL_LABELS[name],
        )
        for name in ["transformer", "tcn_causal", "tcn", "lowpass"]
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.01))
    _save_figure(fig, "figure2_generalization_summary_ieee", output_dir)


def plot_deployment_tradeoff(
    generalization_rows: List[Dict[str, str]],
    deployment_rows: List[Dict[str, str]],
    output_dir: Path,
) -> None:
    by_experiment_rmse = {
        row["Model"]: _parse_value_with_std(row["RMSE"])
        for row in generalization_rows
        if row["Setting"] == "by_experiment"
    }

    fig, ax = plt.subplots(figsize=(9.8, 7.0))
    _style_axis(ax, "(A)")
    ax.grid(True, axis="both")

    for row in deployment_rows:
        model_name = row["Model"]
        if model_name not in MODEL_ORDER:
            continue
        if model_name not in by_experiment_rmse:
            continue

        cpu_ms, cpu_std = _parse_value_with_std(row["CPU ms/window"])
        rmse, rmse_std = by_experiment_rmse[model_name]
        model_size_mb = _to_float(row["FP32 MB"])
        bubble_size = 220 + 520 * max(model_size_mb, 0.02)

        ax.errorbar(
            cpu_ms,
            rmse,
            xerr=cpu_std,
            yerr=rmse_std,
            fmt="none",
            ecolor="#555555",
            elinewidth=1.0,
            capsize=3.2,
            zorder=2,
        )
        ax.scatter(
            cpu_ms,
            rmse,
            s=bubble_size,
            color=MODEL_COLORS[model_name],
            alpha=0.92,
            edgecolors="white",
            linewidths=1.1,
            zorder=3,
        )
        dx, dy = _label_offset(model_name)
        ax.annotate(
            MODEL_LABELS[model_name],
            xy=(cpu_ms, rmse),
            xytext=(cpu_ms + dx, rmse + dy),
            textcoords="data",
            fontsize=9.5,
            color="#333333",
            arrowprops={
                "arrowstyle": "-",
                "color": "#666666",
                "linewidth": 0.8,
                "shrinkA": 4,
                "shrinkB": 4,
            },
        )

    tcn_causal_row = next((row for row in deployment_rows if row["Model"] == "tcn_causal"), None)
    if tcn_causal_row is not None:
        streaming_mean = _to_float(tcn_causal_row["Streaming mean ms/step"])
        streaming_p95 = _to_float(tcn_causal_row["Streaming p95 ms/step"])
        ax.text(
            0.04,
            0.05,
            f"TCN-causal streaming verified\nmean {streaming_mean:.3f} ms/step | p95 {streaming_p95:.3f} ms/step",
            transform=ax.transAxes,
            fontsize=9.5,
            ha="left",
            va="bottom",
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "#fff5eb",
                "edgecolor": "#f0c490",
            },
        )

    ax.set_xlabel("CPU inference time per window (ms)")
    ax.set_ylabel("RMSE under by-experiment setting")
    ax.set_title("Accuracy-Efficiency Trade-off")
    ax.annotate(
        "Better",
        xy=(0.07, 0.93),
        xycoords="axes fraction",
        fontsize=10,
        color="#555555",
        ha="left",
        va="center",
    )
    ax.annotate(
        "",
        xy=(0.14, 0.88),
        xytext=(0.07, 0.93),
        xycoords="axes fraction",
        arrowprops={"arrowstyle": "->", "linewidth": 1.0, "color": "#555555"},
    )
    ax.set_xlim(left=-0.05)
    ax.set_ylim(
        bottom=min(value[0] for value in by_experiment_rmse.values()) - 0.03,
        top=max(value[0] for value in by_experiment_rmse.values()) + 0.07,
    )
    _save_figure(fig, "figure3_deployment_tradeoff_ieee", output_dir)


def plot_generalization_gap(rows: List[Dict[str, str]], output_dir: Path) -> None:
    models = ["transformer", "tcn_causal", "tcn", "lowpass"]
    anomaly_gap = []
    anomaly_gap_std = []
    motion_gap = []
    motion_gap_std = []
    for model_name in models:
        base_rmse, base_std = _parse_value_with_std(_find_row(rows, "by_experiment", model_name)["RMSE"])
        anomaly_rmse, anomaly_std = _parse_value_with_std(_find_row(rows, "anomaly_test_only", model_name)["RMSE"])
        motion_rmse, motion_std = _parse_value_with_std(_find_row(rows, "by_motion_type", model_name)["RMSE"])
        anomaly_gap.append(100.0 * (anomaly_rmse - base_rmse) / base_rmse)
        motion_gap.append(100.0 * (motion_rmse - base_rmse) / base_rmse)
        anomaly_gap_std.append(_gap_std(base_rmse, base_std, anomaly_rmse, anomaly_std))
        motion_gap_std.append(_gap_std(base_rmse, base_std, motion_rmse, motion_std))

    fig, ax = plt.subplots(figsize=(9.8, 6.4))
    _style_axis(ax, "(A)")
    x_positions = list(range(len(models)))
    width = 0.34
    bars1 = ax.bar(
        [x - width / 2 for x in x_positions],
        anomaly_gap,
        yerr=anomaly_gap_std,
        width=width,
        color="#aec7e8",
        edgecolor="white",
        linewidth=0.9,
        error_kw={"elinewidth": 1.05, "capsize": 3.2, "capthick": 1.0, "ecolor": "#333333"},
        label="Anomaly test-only gap",
    )
    bars2 = ax.bar(
        [x + width / 2 for x in x_positions],
        motion_gap,
        yerr=motion_gap_std,
        width=width,
        color="#f4a261",
        edgecolor="white",
        linewidth=0.9,
        error_kw={"elinewidth": 1.05, "capsize": 3.2, "capthick": 1.0, "ecolor": "#333333"},
        label="By-motion-type gap",
    )

    for bar_group in (bars1, bars2):
        for patch in bar_group:
            height = patch.get_height()
            ax.text(
                patch.get_x() + patch.get_width() / 2,
                height + 1.2,
                f"{height:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#333333",
            )

    ax.set_xticks(x_positions, [MODEL_LABELS[name] for name in models])
    ax.set_ylabel("Relative RMSE increase from by-experiment")
    ax.set_title("Cross-condition Generalization Gap")
    ax.grid(True, axis="y")
    ax.legend(frameon=False, loc="upper left")
    ax.text(
        0.98,
        0.94,
        "Motion shift dominates\nall model families",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9.5,
        color="#7a4f2b",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#fff5eb", "edgecolor": "#f0c490"},
    )
    _save_figure(fig, "figure4_generalization_gap_ieee", output_dir)


def _find_row(rows: List[Dict[str, str]], setting_name: str, model_name: str) -> Dict[str, str]:
    for row in rows:
        if row["Setting"] == setting_name and row["Model"] == model_name:
            return row
    raise KeyError(f"Missing row for setting={setting_name}, model={model_name}")


def _label_offset(model_name: str) -> Tuple[float, float]:
    offsets = {
        "transformer": (0.06, -0.004),
        "tcn_causal": (0.06, 0.015),
        "tcn": (0.06, -0.018),
        "gru": (0.05, 0.012),
        "lowpass": (0.10, 0.012),
    }
    return offsets.get(model_name, (0.05, 0.01))


def _style_axis(axis, panel_label: str) -> None:
    axis.text(
        -0.16,
        1.05,
        panel_label,
        transform=axis.transAxes,
        fontsize=11.5,
        fontweight="bold",
        va="bottom",
        ha="left",
        color="#222222",
    )
    axis.spines["left"].set_linewidth(1.05)
    axis.spines["bottom"].set_linewidth(1.05)


def _gap_std(base_mean: float, base_std: float, test_mean: float, test_std: float) -> float:
    if base_mean == 0:
        return 0.0
    d_gap_d_test = 100.0 / base_mean
    d_gap_d_base = -100.0 * test_mean / (base_mean * base_mean)
    variance = (d_gap_d_test * test_std) ** 2 + (d_gap_d_base * base_std) ** 2
    return math.sqrt(max(variance, 0.0))


if __name__ == "__main__":
    main()
