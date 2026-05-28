from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


MODEL_ORDER = ["TCN-causal", "GRU-causal", "LSTM-causal", "Transformer-causal", "MLP-causal"]
SETTING_ORDER = ["By experiment", "By motion type", "Anomaly test-only"]
MODEL_COLORS = {
    "TCN-causal": "#315F86",
    "GRU-causal": "#4E8A68",
    "LSTM-causal": "#A06666",
    "Transformer-causal": "#9A8058",
    "MLP-causal": "#7C7C7C",
}
SETTING_MARKERS = {
    "By experiment": "o",
    "By motion type": "s",
    "Anomaly test-only": "^",
}


def _parse_mean_std(value: str) -> tuple[float, float]:
    numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(value))
    if not numbers:
        return np.nan, np.nan
    mean = float(numbers[0])
    std = float(numbers[1]) if len(numbers) > 1 else 0.0
    return mean, std


def _load_results(table_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(table_path)
    parse_columns = {
        "RMSE down": "rmse",
        "Pearson r up": "pearson",
        "PSD dist. down": "psd",
        "HF imp. up": "hf_improvement",
        "CPU window (ms) down": "cpu_window_ms",
        "Stream (ms/step) down": "stream_ms",
    }
    for source, target in parse_columns.items():
        means, stds = zip(*frame[source].map(_parse_mean_std))
        frame[f"{target}_mean"] = means
        frame[f"{target}_std"] = stds
    frame["params_k"] = pd.to_numeric(frame["Params (K)"], errors="coerce")
    frame["Model"] = pd.Categorical(frame["Model"], categories=MODEL_ORDER, ordered=True)
    frame["Evaluation setting"] = pd.Categorical(
        frame["Evaluation setting"],
        categories=SETTING_ORDER,
        ordered=True,
    )
    return frame.sort_values(["Evaluation setting", "Model"]).reset_index(drop=True)


def _normalise(values: pd.Series, higher_is_better: bool) -> pd.Series:
    min_value = float(values.min())
    max_value = float(values.max())
    if math.isclose(min_value, max_value):
        return pd.Series(np.ones(len(values)), index=values.index)
    scaled = (values - min_value) / (max_value - min_value)
    return scaled if higher_is_better else 1.0 - scaled


def _add_scores(frame: pd.DataFrame) -> pd.DataFrame:
    scored = frame.copy()
    scored["score_rmse"] = np.nan
    scored["score_pearson"] = np.nan
    scored["score_psd"] = np.nan
    scored["score_hf"] = np.nan
    scored["score_stream"] = np.nan
    for _, indices in scored.groupby("Evaluation setting", observed=True).groups.items():
        subset = scored.loc[indices]
        scored.loc[indices, "score_rmse"] = _normalise(subset["rmse_mean"], higher_is_better=False)
        scored.loc[indices, "score_pearson"] = _normalise(subset["pearson_mean"], higher_is_better=True)
        scored.loc[indices, "score_psd"] = _normalise(subset["psd_mean"], higher_is_better=False)
        scored.loc[indices, "score_hf"] = _normalise(subset["hf_improvement_mean"], higher_is_better=True)
        scored.loc[indices, "score_stream"] = _normalise(subset["stream_ms_mean"], higher_is_better=False)
    scored["compensation_score"] = scored[["score_rmse", "score_pearson", "score_psd", "score_hf"]].mean(axis=1)
    scored["deployment_score"] = scored[["score_stream"]].mean(axis=1)
    scored["balanced_score"] = scored[["compensation_score", "deployment_score"]].mean(axis=1)
    return scored


def _set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.linewidth": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
        }
    )


def _panel_label(ax, label: str) -> None:
    ax.text(-0.14, 1.06, label, transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")


def _plot_rmse_forest(ax, frame: pd.DataFrame) -> None:
    y_positions = []
    y_labels = []
    y = 0
    for setting in SETTING_ORDER:
        subset = frame[frame["Evaluation setting"] == setting]
        for model in MODEL_ORDER:
            row = subset[subset["Model"] == model].iloc[0]
            y_positions.append(y)
            y_labels.append(f"{setting}\n{model}" if model == MODEL_ORDER[0] else model.replace("-causal", ""))
            ax.errorbar(
                row["rmse_mean"],
                y,
                xerr=row["rmse_std"],
                fmt="o",
                color=MODEL_COLORS[model],
                ecolor=MODEL_COLORS[model],
                markersize=4.2 if model == "TCN-causal" else 3.6,
                elinewidth=1.35,
                capsize=3.0,
                capthick=1.05,
                markeredgecolor="black",
                markeredgewidth=0.35,
                zorder=3,
            )
            if model in {"TCN-causal", "MLP-causal"}:
                label_x_offset = 0.016
                if model == "TCN-causal" and setting == "By experiment":
                    label_x_offset = 0.070
                ax.text(
                    row["rmse_mean"] + label_x_offset,
                    y,
                    f'{row["rmse_mean"]:.3f}\u00b1{row["rmse_std"]:.3f}',
                    va="center",
                    fontsize=5.7,
                    color="#364152",
                )
            y += 1
        if setting != SETTING_ORDER[-1]:
            ax.axhline(y - 0.5, color="#D9DEE6", lw=0.7, zorder=1)
            y += 0.8
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels)
    ax.invert_yaxis()
    ax.set_xlabel("RMSE (lower is better)")
    ax.set_title("Error across evaluation settings", loc="left", fontsize=8, fontweight="bold")
    ax.grid(axis="x", color="#E3E7ED", lw=0.65)
    ax.set_xlim(0.32, 1.08)
    ax.annotate(
        "TCN-causal is consistently lowest",
        xy=(0.3912, 0),
        xytext=(0.56, 0.65),
        arrowprops={"arrowstyle": "->", "lw": 0.7, "color": "#364152"},
        fontsize=6.4,
        color="#364152",
    )


def _plot_frequency_panel(ax, frame: pd.DataFrame) -> None:
    for setting in SETTING_ORDER:
        subset = frame[frame["Evaluation setting"] == setting]
        for model in MODEL_ORDER:
            row = subset[subset["Model"] == model].iloc[0]
            ax.errorbar(
                row["psd_mean"],
                row["hf_improvement_mean"],
                xerr=row["psd_std"],
                yerr=row["hf_improvement_std"],
                fmt=SETTING_MARKERS[setting],
                color=MODEL_COLORS[model],
                ecolor=MODEL_COLORS[model],
                markersize=4.6,
                markeredgecolor="black",
                markeredgewidth=0.35,
                elinewidth=1.05,
                capsize=2.6,
                capthick=0.9,
                alpha=0.9,
            )
    ax.set_xlabel("PSD distance (lower)")
    ax.set_ylabel("HF improvement (higher)")
    ax.set_title("Spectral fidelity and high-frequency suppression", loc="left", fontsize=8, fontweight="bold")
    ax.grid(color="#E3E7ED", lw=0.65)
    ax.annotate(
        "better",
        xy=(0.058, 8.30),
        xytext=(0.16, 7.25),
        arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "#364152"},
        fontsize=6.5,
        color="#364152",
    )


def _plot_pareto_panel(ax, scored: pd.DataFrame) -> None:
    mean_frame = (
        scored.groupby("Model", observed=True)
        .agg(
            rmse_mean=("rmse_mean", "mean"),
            rmse_std=("rmse_mean", "std"),
            stream_ms_mean=("stream_ms_mean", "mean"),
            stream_ms_std=("stream_ms_mean", "std"),
            params_k=("params_k", "mean"),
            compensation_score=("compensation_score", "mean"),
        )
        .reset_index()
    )
    for _, row in scored.iterrows():
        model = str(row["Model"])
        ax.scatter(
            row["stream_ms_mean"],
            row["rmse_mean"],
            s=18,
            marker=SETTING_MARKERS[str(row["Evaluation setting"])],
            color=MODEL_COLORS[model],
            edgecolor="white",
            linewidth=0.25,
            alpha=0.38,
            zorder=2,
        )
    for _, row in mean_frame.iterrows():
        model = str(row["Model"])
        ax.errorbar(
            row["stream_ms_mean"],
            row["rmse_mean"],
            xerr=row["stream_ms_std"],
            yerr=row["rmse_std"],
            fmt="o",
            markersize=np.sqrt(28 + row["params_k"] * 0.72),
            mfc=MODEL_COLORS[model],
            mec="black",
            mew=0.5,
            ecolor=MODEL_COLORS[model],
            elinewidth=1.1,
            capsize=2.8,
            capthick=0.9,
            alpha=0.92,
            zorder=4,
        )
        label = model.replace("-causal", "")
        offsets = {
            "MLP-causal": (4, 2),
            "GRU-causal": (6, -1),
            "LSTM-causal": (8, 8),
            "Transformer-causal": (-58, -9),
            "TCN-causal": (-24, -13),
        }
        ax.annotate(
            label,
            xy=(row["stream_ms_mean"], row["rmse_mean"]),
            xytext=offsets.get(model, (5, 0)),
            textcoords="offset points",
            fontsize=6.6,
            color="#172033",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 0.7},
            zorder=5,
        )
    ax.set_xscale("log")
    ax.set_xlim(0.04, 45)
    ax.set_ylim(0.35, 0.86)
    ax.set_xlabel("Streaming latency (ms/step, log scale)")
    ax.set_ylabel("Mean RMSE")
    ax.set_title("Accuracy-deployment Pareto frontier", loc="left", fontsize=8, fontweight="bold")
    ax.grid(color="#E3E7ED", lw=0.65, which="both")
    ax.annotate(
        "lower-left is better",
        xy=(0.32, 0.43),
        xytext=(0.08, 0.74),
        arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "#364152"},
        fontsize=6.5,
        color="#364152",
    )


def _plot_score_panel(ax, scored: pd.DataFrame) -> None:
    mean_scores = (
        scored.groupby("Model", observed=True)
        .agg(
            compensation_score=("compensation_score", "mean"),
            compensation_score_std=("compensation_score", "std"),
            deployment_score=("deployment_score", "mean"),
            deployment_score_std=("deployment_score", "std"),
            balanced_score=("balanced_score", "mean"),
            balanced_score_std=("balanced_score", "std"),
        )
        .reset_index()
    )
    mean_scores["Model"] = pd.Categorical(mean_scores["Model"], categories=MODEL_ORDER, ordered=True)
    mean_scores = mean_scores.sort_values("balanced_score", ascending=False)
    y = np.arange(len(mean_scores))
    ax.barh(
        y,
        mean_scores["balanced_score"],
        xerr=mean_scores["balanced_score_std"],
        color=[MODEL_COLORS[str(model)] for model in mean_scores["Model"]],
        edgecolor="black",
        linewidth=0.35,
        height=0.55,
        error_kw={"elinewidth": 1.1, "capsize": 3.0, "capthick": 0.9, "ecolor": "#27303F"},
    )
    ax.errorbar(
        mean_scores["compensation_score"],
        y + 0.17,
        xerr=mean_scores["compensation_score_std"],
        fmt="D",
        markersize=4.2,
        mfc="white",
        mec="black",
        mew=0.45,
        ecolor="black",
        elinewidth=0.85,
        capsize=2.0,
        capthick=0.75,
        linestyle="",
        label="Compensation score",
        zorder=3,
    )
    ax.errorbar(
        mean_scores["deployment_score"],
        y - 0.17,
        xerr=mean_scores["deployment_score_std"],
        fmt="o",
        markersize=4.3,
        mfc="black",
        mec="black",
        mew=0.45,
        ecolor="black",
        elinewidth=0.85,
        capsize=2.0,
        capthick=0.75,
        linestyle="",
        label="Streaming score",
        zorder=3,
    )
    ax.text(
        0.01,
        1.04,
        "error bars: std across three evaluation settings",
        transform=ax.transAxes,
        fontsize=5.8,
        color="#4B5563",
        va="top",
    )
    ax.set_yticks(y)
    ax.set_yticklabels([str(model).replace("-causal", "") for model in mean_scores["Model"]])
    ax.set_xlim(0, 1.05)
    ax.invert_yaxis()
    ax.set_xlabel("Normalised score (0-1)")
    ax.set_title("Integrated performance summary", loc="left", fontsize=8, fontweight="bold")
    ax.grid(axis="x", color="#E3E7ED", lw=0.65)
    ax.legend(loc="lower right", fontsize=6, handletextpad=0.4, ncol=2)


def _add_legends(fig) -> None:
    model_handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=MODEL_COLORS[model], markeredgecolor="black", markeredgewidth=0.35, label=model)
        for model in MODEL_ORDER
    ]
    setting_handles = [
        Line2D([0], [0], marker=marker, linestyle="", markerfacecolor="white", markeredgecolor="#27303F", label=setting)
        for setting, marker in SETTING_MARKERS.items()
    ]
    first_legend = fig.legend(
        handles=model_handles,
        loc="upper center",
        bbox_to_anchor=(0.49, 1.02),
        ncol=5,
        fontsize=7,
        columnspacing=1.3,
        handletextpad=0.4,
        frameon=False,
    )
    fig.add_artist(first_legend)
    fig.legend(
        handles=setting_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.975),
        ncol=3,
        fontsize=6.5,
        columnspacing=1.2,
        handletextpad=0.4,
        frameon=False,
    )


def make_figure(table_path: Path, output_dir: Path) -> dict[str, Path]:
    _set_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = _load_results(table_path)
    scored = _add_scores(frame)
    scored.to_csv(output_dir / "paper_causal_model_benchmark_source.csv", index=False)

    fig = plt.figure(figsize=(7.25, 6.65))
    grid = fig.add_gridspec(
        3,
        2,
        width_ratios=[0.92, 1.08],
        height_ratios=[1.05, 0.95, 0.72],
        left=0.085,
        right=0.985,
        bottom=0.07,
        top=0.875,
        wspace=0.34,
        hspace=0.50,
    )
    ax_a = fig.add_subplot(grid[0:2, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 1])
    ax_d = fig.add_subplot(grid[2, :])

    _plot_rmse_forest(ax_a, frame)
    _plot_frequency_panel(ax_b, frame)
    _plot_pareto_panel(ax_c, scored)
    _plot_score_panel(ax_d, scored)

    _panel_label(ax_a, "a")
    _panel_label(ax_b, "b")
    _panel_label(ax_c, "c")
    _panel_label(ax_d, "d")
    _add_legends(fig)

    base = output_dir / "paper_causal_model_benchmark"
    paths = {
        "png": base.with_suffix(".png"),
        "svg": base.with_suffix(".svg"),
        "pdf": base.with_suffix(".pdf"),
        "tiff": base.with_suffix(".tiff"),
    }
    fig.savefig(paths["png"], dpi=600, bbox_inches="tight")
    fig.savefig(paths["svg"], bbox_inches="tight")
    fig.savefig(paths["pdf"], bbox_inches="tight")
    fig.savefig(paths["tiff"], dpi=600, bbox_inches="tight")
    plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a publication-grade causal model benchmark figure.")
    parser.add_argument(
        "--table",
        type=Path,
        default=Path("outputs/causal_model_comparison/tables/paper_main_causal_model_comparison.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/causal_model_comparison/figures"),
    )
    args = parser.parse_args()
    paths = make_figure(table_path=args.table, output_dir=args.output_dir)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
