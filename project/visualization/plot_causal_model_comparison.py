from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_ORDER = ["TCN-causal", "Transformer-causal", "GRU-causal", "LSTM-causal", "MLP-causal"]
SETTING_ORDER = ["by_experiment", "by_motion_type", "anomaly_test_only"]
SETTING_LABELS = {
    "by_experiment": "By experiment",
    "by_motion_type": "By motion type",
    "anomaly_test_only": "Anomaly test-only",
}
MODEL_COLORS = {
    "TCN-causal": "#2E5E8C",
    "Transformer-causal": "#8A6F46",
    "GRU-causal": "#4C8C6B",
    "LSTM-causal": "#9A5A5A",
    "MLP-causal": "#777777",
}


def _parse_mean_std(value: str) -> tuple[float, float]:
    text = str(value)
    matches = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if not matches:
        return np.nan, np.nan
    mean = float(matches[0])
    std = float(matches[1]) if len(matches) > 1 else 0.0
    return mean, std


def _prepare_table(table_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(table_path)
    for column in ["RMSE", "Pearson", "PSD Dist.", "HF Improve.", "CPU forward ms/window", "Streaming ms/step"]:
        means = []
        stds = []
        for value in frame[column]:
            mean, std = _parse_mean_std(value)
            means.append(mean)
            stds.append(std)
        stem = (
            column.lower()
            .replace(" ", "_")
            .replace(".", "")
            .replace("/", "_per_")
            .replace("-", "_")
        )
        frame[f"{stem}_mean"] = means
        frame[f"{stem}_std"] = stds
    frame["Parameters"] = pd.to_numeric(frame["Parameters"], errors="coerce")
    frame["FP32 size (MB)"] = pd.to_numeric(frame["FP32 size (MB)"], errors="coerce")
    frame["Model"] = pd.Categorical(frame["Model"], categories=MODEL_ORDER, ordered=True)
    frame["Setting"] = pd.Categorical(frame["Setting"], categories=SETTING_ORDER, ordered=True)
    return frame.sort_values(["Setting", "Model"]).reset_index(drop=True)


def _plot_metric_bars(ax, frame: pd.DataFrame, metric: str, ylabel: str, higher_is_better: bool = False) -> None:
    width = 0.14
    x = np.arange(len(SETTING_ORDER))
    for index, model in enumerate(MODEL_ORDER):
        subset = frame[frame["Model"] == model].set_index("Setting").reindex(SETTING_ORDER)
        offset = (index - (len(MODEL_ORDER) - 1) / 2) * width
        ax.bar(
            x + offset,
            subset[f"{metric}_mean"],
            yerr=subset[f"{metric}_std"],
            width=width,
            color=MODEL_COLORS[model],
            edgecolor="black",
            linewidth=0.35,
            capsize=2.0,
            label=model,
            alpha=0.92,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([SETTING_LABELS[item] for item in SETTING_ORDER], rotation=18, ha="right")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#D7DCE2", linewidth=0.6, alpha=0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ymax = np.nanmax([patch.get_height() for patch in ax.patches])
    ax.set_ylim(0, ymax * 1.18 if ymax > 0 else 1.0)
    note = "higher is better" if higher_is_better else "lower is better"
    ax.text(
        0.02,
        0.95,
        note,
        transform=ax.transAxes,
        va="top",
        fontsize=7,
        color="#4B5563",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.5},
    )


def _plot_tradeoff(ax, frame: pd.DataFrame) -> None:
    markers = {"by_experiment": "o", "by_motion_type": "s", "anomaly_test_only": "^"}
    for setting in SETTING_ORDER:
        subset = frame[frame["Setting"] == setting]
        for _, row in subset.iterrows():
            model = str(row["Model"])
            ax.scatter(
                row["streaming_ms_per_step_mean"],
                row["rmse_mean"],
                s=max(row["Parameters"] / 1800.0, 28),
                marker=markers[setting],
                color=MODEL_COLORS[model],
                edgecolor="black",
                linewidth=0.45,
                alpha=0.85,
            )
            if setting == "by_experiment":
                x_factor = 1.12
                ha = "left"
                if model == "Transformer-causal":
                    x_factor = 0.74
                    ha = "right"
                ax.text(
                    row["streaming_ms_per_step_mean"] * x_factor,
                    row["rmse_mean"],
                    model.replace("-causal", ""),
                    fontsize=6.5,
                    va="center",
                    ha=ha,
                    color="#1F2937",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.68, "pad": 0.6},
                )
    ax.set_xscale("log")
    ax.set_xlim(0.035, 55.0)
    ax.set_xlabel("Streaming latency (ms/step, log scale)")
    ax.set_ylabel("RMSE")
    ax.grid(True, which="both", color="#D7DCE2", linewidth=0.55, alpha=0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(
        0.02,
        0.95,
        "lower-left is better",
        transform=ax.transAxes,
        va="top",
        fontsize=7,
        color="#4B5563",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.5},
    )


def plot_overview(table_path: Path, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = _prepare_table(table_path)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
            "axes.linewidth": 0.7,
            "figure.dpi": 140,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), constrained_layout=True)
    _plot_metric_bars(axes[0, 0], frame, "rmse", "RMSE")
    _plot_metric_bars(axes[0, 1], frame, "psd_dist", "PSD distance")
    _plot_metric_bars(axes[1, 0], frame, "hf_improve", "HF improvement", higher_is_better=True)
    _plot_tradeoff(axes[1, 1], frame)

    for label, ax in zip(["a", "b", "c", "d"], axes.ravel()):
        ax.text(-0.12, 1.08, label, transform=ax.transAxes, fontsize=10, fontweight="bold")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.52, 1.04))

    base = output_dir / "causal_model_comparison_overview"
    paths = {
        "png": base.with_suffix(".png"),
        "svg": base.with_suffix(".svg"),
        "pdf": base.with_suffix(".pdf"),
    }
    fig.savefig(paths["png"], dpi=600, bbox_inches="tight")
    fig.savefig(paths["svg"], bbox_inches="tight")
    fig.savefig(paths["pdf"], bbox_inches="tight")
    plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot causal model comparison overview.")
    parser.add_argument(
        "--table",
        type=Path,
        default=Path("outputs/causal_model_comparison/tables/causal_model_comparison_table.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/causal_model_comparison/figures"),
    )
    args = parser.parse_args()
    paths = plot_overview(table_path=args.table, output_dir=args.output_dir)
    for kind, path in paths.items():
        print(f"{kind}: {path}")


if __name__ == "__main__":
    main()
