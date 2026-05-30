from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


METHOD_ORDER = [
    "TCN-causal",
    "Transformer-causal",
    "GRU-causal",
    "LSTM-causal",
    "MLP-causal",
    "Moving-average low-pass",
    "Butterworth low-pass",
    "Savitzky-Golay",
    "Wiener filter",
    "Identity/raw",
]

LATENCY_ORDER = [
    "Identity/raw",
    "TCN-causal",
    "Transformer-causal",
    "GRU-causal",
    "LSTM-causal",
    "MLP-causal",
    "Moving-average low-pass",
    "Butterworth low-pass",
    "Savitzky-Golay",
    "Wiener filter",
]

NEURAL_ORDER = [
    "MLP-causal",
    "TCN-causal",
    "GRU-causal",
    "LSTM-causal",
    "Transformer-causal",
]

SHORT_LABELS = {
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
    "TCN-causal": "#0076B9",
    "Transformer-causal": "#4DAF4A",
    "GRU-causal": "#4DAF4A",
    "LSTM-causal": "#4DAF4A",
    "MLP-causal": "#4DAF4A",
    "Moving-average low-pass": "#F28E1C",
    "Butterworth low-pass": "#F28E1C",
    "Savitzky-Golay": "#F28E1C",
    "Wiener filter": "#F28E1C",
    "Identity/raw": "#2F3F48",
}

FAMILY_COLORS = {
    "proposed": "#0076B9",
    "causal_nn": "#4DAF4A",
    "classical": "#F28E1C",
    "raw": "#2F3F48",
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


def _method_family(method: str) -> str:
    if method == "Identity/raw":
        return "raw"
    if method == "TCN-causal":
        return "proposed"
    if method.endswith("-causal"):
        return "causal_nn"
    return "classical"


def _aggregate_mean_std(
    group: pd.DataFrame,
    mean_col: str,
    std_col: str,
    error_mode: str,
) -> tuple[float, float]:
    means = pd.to_numeric(group[mean_col], errors="coerce").to_numpy(dtype=float)
    stds = pd.to_numeric(group[std_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    mask = np.isfinite(means)
    if not mask.any():
        return np.nan, np.nan
    means = means[mask]
    stds = stds[mask]
    mean = float(np.mean(means))
    if error_mode == "pooled":
        # Combines within-setting seed variance and between-setting mean shift.
        spread = float(np.sqrt(np.mean(stds**2) + np.var(means, ddof=0)))
    else:
        # Shows only the seed-level s.d. reported in the result table.
        spread = float(np.sqrt(np.mean(stds**2)))
    return mean, spread


def _first_existing_column(frame: pd.DataFrame, candidates: list[str]) -> str:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise KeyError(f"None of these columns were found: {candidates}")


def _parameters_to_k(parameters: pd.Series) -> float:
    values = pd.to_numeric(parameters, errors="coerce").dropna()
    if values.empty:
        return 0.0
    value = float(values.iloc[0])
    return value / 1000.0 if value > 1000 else value


def _load_and_aggregate(table_path: Path, error_mode: str) -> pd.DataFrame:
    raw = pd.read_csv(table_path)
    columns = {
        "rmse_mean": _first_existing_column(raw, ["rmse_mean"]),
        "rmse_std": _first_existing_column(raw, ["rmse_std"]),
        "pearson_mean": _first_existing_column(raw, ["pearson_mean"]),
        "pearson_std": _first_existing_column(raw, ["pearson_std"]),
        "psd_mean": _first_existing_column(raw, ["psd_distance_mean"]),
        "psd_std": _first_existing_column(raw, ["psd_distance_std"]),
        "hf_mean": _first_existing_column(raw, ["hf_improvement_mean"]),
        "hf_std": _first_existing_column(raw, ["hf_improvement_std"]),
        "cpu_mean": _first_existing_column(raw, ["cpu_forward_ms_per_window_mean", "cpu_window_ms_mean"]),
        "cpu_std": _first_existing_column(raw, ["cpu_forward_ms_per_window_std", "cpu_window_ms_std"]),
        "stream_mean": _first_existing_column(raw, ["streaming_ms_per_step_mean", "stream_ms_step_mean"]),
        "stream_std": _first_existing_column(raw, ["streaming_ms_per_step_std", "stream_ms_step_std"]),
        "fp32_size_mb": _first_existing_column(raw, ["fp32_size_mb"]),
    }
    params_column = "params_k" if "params_k" in raw.columns else _first_existing_column(raw, ["parameters"])
    rows: list[dict[str, float | str]] = []
    metric_pairs = {
        "rmse": (columns["rmse_mean"], columns["rmse_std"]),
        "pearson": (columns["pearson_mean"], columns["pearson_std"]),
        "psd": (columns["psd_mean"], columns["psd_std"]),
        "hf": (columns["hf_mean"], columns["hf_std"]),
        "cpu": (columns["cpu_mean"], columns["cpu_std"]),
        "stream": (columns["stream_mean"], columns["stream_std"]),
    }
    for method in METHOD_ORDER:
        group = raw[raw["method"] == method]
        if group.empty:
            continue
        row: dict[str, float | str] = {
            "method": method,
            "label": SHORT_LABELS[method],
            "family": _method_family(method),
            "params_k": _parameters_to_k(group[params_column]),
            "fp32_size_mb": float(pd.to_numeric(group[columns["fp32_size_mb"]], errors="coerce").dropna().iloc[0]),
        }
        for prefix, (mean_col, std_col) in metric_pairs.items():
            mean, std = _aggregate_mean_std(group, mean_col, std_col, error_mode=error_mode)
            row[f"{prefix}_mean"] = mean
            row[f"{prefix}_std"] = std
        rows.append(row)
    return pd.DataFrame(rows)


def _ordered(frame: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    ordered = frame.copy()
    ordered["method"] = pd.Categorical(ordered["method"], categories=order, ordered=True)
    return ordered.sort_values("method").reset_index(drop=True)


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.10,
        1.045,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        fontweight="bold",
    )


def _metric_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    mean_col: str,
    std_col: str,
    title: str,
    xlabel: str,
    xlim: tuple[float, float] | None = None,
    show_y: bool = False,
) -> None:
    panel_frame = _ordered(frame, METHOD_ORDER)
    y = np.arange(len(panel_frame))[::-1]
    colors = [METHOD_COLORS[str(method)] for method in panel_frame["method"]]
    means = panel_frame[mean_col].to_numpy(dtype=float)
    stds = panel_frame[std_col].fillna(0.0).to_numpy(dtype=float)
    bars = ax.barh(
        y,
        means,
        xerr=stds,
        height=0.52,
        color=colors,
        edgecolor="#263238",
        linewidth=0.34,
        error_kw={"elinewidth": 0.62, "capsize": 1.6, "capthick": 0.58, "ecolor": "#2B2B2B"},
        zorder=3,
    )
    for bar, method in zip(bars, panel_frame["method"]):
        if method == "TCN-causal":
            bar.set_linewidth(0.86)
            bar.set_edgecolor("#000000")
    ax.set_title(title, loc="left", fontsize=8, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_yticks(y)
    if show_y:
        ax.set_yticklabels(panel_frame["label"], fontsize=6.2)
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
    ax.axhline(4.5, color="#808080", lw=0.55, ls=(0, (1.6, 2.2)), zorder=1)
    ax.axhline(0.5, color="#808080", lw=0.55, ls=(0, (1.6, 2.2)), zorder=1)
    ax.grid(axis="x", color="#DDE3EA", lw=0.55, ls=(0, (2.0, 2.6)), zorder=0)
    if xlim is not None:
        ax.set_xlim(*xlim)

def _footprint_panel(ax: plt.Axes, frame: pd.DataFrame) -> None:
    footprint_frame = _ordered(frame[frame["method"].isin(NEURAL_ORDER)], NEURAL_ORDER)
    x = footprint_frame["params_k"].to_numpy(dtype=float)
    y = footprint_frame["fp32_size_mb"].to_numpy(dtype=float)
    color = FAMILY_COLORS["proposed"]
    ax.vlines(x, ymin=0.01, ymax=y, color="#7DB8DF", lw=0.72, ls=(0, (1.4, 1.8)), zorder=2)
    ax.scatter(x, y, s=18, color=color, edgecolor="#263238", linewidth=0.35, zorder=4)
    label_offsets = {
        "MLP-causal": (-6, 1.18),
        "TCN-causal": (-12, 1.20),
        "GRU-causal": (-8, 1.20),
        "LSTM-causal": (-14, 1.18),
        "Transformer-causal": (-30, 1.20),
    }
    for _, row in footprint_frame.iterrows():
        dx, ymul = label_offsets[str(row["method"])]
        ax.text(
            row["params_k"] + dx,
            row["fp32_size_mb"] * ymul,
            str(row["label"]),
            fontsize=5.5,
            color="#263238",
            ha="left",
            va="bottom",
        )
    ax.set_title("Model footprint (neural models only)", loc="left", fontsize=8, fontweight="bold")
    ax.set_xlabel("Parameters (K)")
    ax.set_ylabel("FP32 size (MB)", color=FAMILY_COLORS["proposed"])
    ax.tick_params(axis="y", colors=FAMILY_COLORS["proposed"])
    ax.set_xlim(0, 500)
    ax.set_ylim(0.01, 10)
    ax.set_yscale("log")
    ax.set_yticks([0.01, 0.1, 1, 10])
    ax.set_yticklabels(["0.01", "0.1", "1", "10"])
    ax.grid(color="#DDE3EA", lw=0.55, ls=(0, (2.0, 2.6)), zorder=0)


def _latency_panel(ax: plt.Axes, frame: pd.DataFrame) -> None:
    panel_frame = _ordered(frame, LATENCY_ORDER)
    y = np.arange(len(panel_frame))[::-1]
    cpu = panel_frame["cpu_mean"].to_numpy(dtype=float)
    stream = panel_frame["stream_mean"].to_numpy(dtype=float)

    ax.scatter(
        cpu,
        y + 0.10,
        marker="o",
        s=16,
        color="#496A80",
        edgecolor="#263238",
        linewidth=0.35,
        label="CPU window",
        zorder=4,
    )
    mask = np.isfinite(stream)
    ax.scatter(
        stream[mask],
        y[mask] - 0.10,
        marker="D",
        s=17,
        color="#C47C52",
        edgecolor="#263238",
        linewidth=0.35,
        label="Streaming step",
        zorder=4,
    )
    ax.set_xscale("log")
    ax.set_title("Inference latency", loc="left", fontsize=8, fontweight="bold")
    ax.set_xlabel("Latency (ms, log scale)")
    ax.set_yticks(y)
    ax.set_yticklabels(panel_frame["label"], fontsize=6.2)
    ax.axhline(3.5, color="#808080", lw=0.55, ls=(0, (1.6, 2.2)), zorder=1)
    ax.grid(axis="x", color="#DDE3EA", lw=0.55, ls=(0, (2.0, 2.6)), which="both", zorder=0)
    ax.set_xlim(6e-4, 120)
    ax.legend(loc="lower right", fontsize=7, handlelength=1.2, borderpad=0.35, frameon=True)

def make_figure(table_path: Path, output_dir: Path, error_mode: str) -> dict[str, Path]:
    _style()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = _load_and_aggregate(table_path, error_mode=error_mode)
    source_path = output_dir / "complete_model_comparison_with_static_methods_numeric.csv"
    frame.to_csv(source_path, index=False)

    fig = plt.figure(figsize=(7.45, 5.70), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        4,
        height_ratios=[1.05, 1.0],
        hspace=0.44,
        wspace=0.42,
    )
    axes_top = [fig.add_subplot(grid[0, idx]) for idx in range(4)]
    ax_footprint = fig.add_subplot(grid[1, 0:2])
    ax_latency = fig.add_subplot(grid[1, 2:4])

    _metric_panel(
        axes_top[0],
        frame,
        "rmse_mean",
        "rmse_std",
        r"RMSE $\downarrow$",
        "RMSE",
        xlim=(0.20, 1.05),
        show_y=True,
    )
    _metric_panel(
        axes_top[1],
        frame,
        "pearson_mean",
        "pearson_std",
        r"Pearson r $\uparrow$",
        "Pearson r",
        xlim=(0.70, 0.91),
    )
    _metric_panel(
        axes_top[2],
        frame,
        "psd_mean",
        "psd_std",
        r"PSD distance $\downarrow$",
        "PSD distance",
        xlim=(0.00, 0.40),
    )
    _metric_panel(
        axes_top[3],
        frame,
        "hf_mean",
        "hf_std",
        r"HF improvement $\uparrow$",
        "HF improvement",
        xlim=(0.0, 8.2),
    )

    _footprint_panel(ax_footprint, frame)
    _latency_panel(ax_latency, frame)

    panel_axes = axes_top + [ax_footprint, ax_latency]
    for label, ax in zip(["a", "b", "c", "d", "e", "f"], panel_axes):
        _panel_label(ax, label)

    fig.legend(
        handles=[
            Patch(facecolor=FAMILY_COLORS["proposed"], edgecolor="#263238", label="Proposed TCN-causal"),
            Patch(facecolor=FAMILY_COLORS["causal_nn"], edgecolor="#263238", label="Other neural comparators"),
            Patch(facecolor=FAMILY_COLORS["classical"], edgecolor="#263238", label="Classical / statistical baselines"),
            Patch(facecolor=FAMILY_COLORS["raw"], edgecolor="#263238", label="Raw signal baseline"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.50, 0.993),
        ncol=4,
        fontsize=7.5,
        handlelength=1.1,
        columnspacing=1.0,
    )
    fig.subplots_adjust(left=0.098, right=0.992, top=0.915, bottom=0.085)

    stem = output_dir / "complete_model_comparison_with_static_methods_nature"
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
    parser = argparse.ArgumentParser(description="Create Nature-style metric-panel comparison figure.")
    parser.add_argument(
        "--table",
        type=Path,
        default=Path("outputs/causal_model_comparison/tables/complete_model_comparison_with_static_methods_numeric.csv"),
        help="Input numeric comparison table.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/causal_model_comparison/figures"),
        help="Directory for exported figures.",
    )
    parser.add_argument(
        "--error-mode",
        choices=["seed", "pooled"],
        default="seed",
        help="Use seed-level s.d. from result tables, or pooled seed plus between-setting spread.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = make_figure(args.table, args.output_dir, error_mode=args.error_mode)
    print("Saved Nature-style metric-panel figure:")
    for kind, path in outputs.items():
        print(f"  {kind}: {path}")


if __name__ == "__main__":
    main()
