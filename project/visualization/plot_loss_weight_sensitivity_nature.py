from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"


PALETTE = {
    "baseline": "#484878",
    "l1": "#7884B4",
    "mse": "#C96F53",
    "spectral": "#42949E",
    "neutral": "#767676",
    "light": "#E7E7EC",
    "grid": "#D8D8D8",
    "positive": "#2E9E44",
    "negative": "#B64342",
    "black": "#272727",
}

TERM_COLORS = {
    "none": PALETTE["baseline"],
    "time_l1": PALETTE["l1"],
    "mse": PALETTE["mse"],
    "spectral": PALETTE["spectral"],
}


def _apply_style() -> None:
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 7,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.06,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        color=PALETTE["black"],
    )


def _display_label(label: str) -> str:
    return (
        str(label)
        .replace("Baseline", "Final")
        .replace("Spectral", "Spec.")
        .replace(" = ", "=")
    )


def _load_tables(summary_path: Path, per_seed_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(summary_path)
    per_seed = pd.read_csv(per_seed_path)
    required_summary = {
        "variant",
        "variant_label",
        "changed_term",
        "rmse_mean",
        "rmse_std",
        "psd_distance_mean",
        "psd_distance_std",
        "hf_ratio_improvement_mean",
        "hf_ratio_improvement_std",
    }
    required_seed = {"variant", "variant_label", "seed", "rmse"}
    missing_summary = sorted(required_summary - set(summary.columns))
    missing_seed = sorted(required_seed - set(per_seed.columns))
    if missing_summary:
        raise ValueError(f"Missing required summary columns: {missing_summary}")
    if missing_seed:
        raise ValueError(f"Missing required per-seed columns: {missing_seed}")
    return summary, per_seed


def _make_directional_delta_frame(summary: pd.DataFrame) -> pd.DataFrame:
    baseline = summary.loc[summary["variant"] == "baseline"].iloc[0]
    rows = []
    metrics = [
        ("RMSE", "rmse_mean", "lower"),
        ("PSD distance", "psd_distance_mean", "lower"),
        ("HF improvement", "hf_ratio_improvement_mean", "higher"),
    ]
    for _, row in summary.iterrows():
        if row["variant"] == "baseline":
            continue
        for metric_label, column, direction in metrics:
            raw_delta_pct = (float(row[column]) - float(baseline[column])) / float(baseline[column]) * 100.0
            directional_delta = -raw_delta_pct if direction == "lower" else raw_delta_pct
            rows.append(
                {
                    "variant": row["variant"],
                    "variant_label": row["variant_label"],
                    "changed_term": row["changed_term"],
                    "metric": metric_label,
                    "directional_delta_percent": directional_delta,
                }
            )
    return pd.DataFrame(rows)


def _plot_tradeoff(ax: plt.Axes, summary: pd.DataFrame) -> None:
    for _, row in summary.iterrows():
        color = TERM_COLORS.get(str(row["changed_term"]), PALETTE["neutral"])
        marker = "*" if row["variant"] == "baseline" else "o"
        size = 115 if row["variant"] == "baseline" else 58
        ax.errorbar(
            row["rmse_mean"],
            row["psd_distance_mean"],
            xerr=row["rmse_std"],
            yerr=row["psd_distance_std"],
            fmt=marker,
            ms=np.sqrt(size),
            mfc=color,
            mec="white",
            mew=0.7,
            ecolor=color,
            elinewidth=0.8,
            capsize=2,
            alpha=0.95,
            zorder=5 if row["variant"] == "baseline" else 4,
        )

    labels = {
        "baseline": (-0.00045, 0.00038, "right"),
        "mse_1p0": (0.00010, -0.00040, "left"),
        "spec_0p4": (0.00008, -0.00022, "left"),
        "l1_2p0": (-0.00018, 0.00030, "right"),
        "spec_0p1": (-0.00016, 0.00028, "right"),
    }
    for _, row in summary.iterrows():
        if row["variant"] not in labels:
            continue
        dx, dy, ha = labels[row["variant"]]
        ax.text(
            row["rmse_mean"] + dx,
            row["psd_distance_mean"] + dy,
            _display_label(row["variant_label"]),
            fontsize=6.5,
            ha=ha,
            va="center",
            color=PALETTE["black"],
        )

    x_pad = (summary["rmse_mean"].max() - summary["rmse_mean"].min()) * 0.22
    y_pad = (summary["psd_distance_mean"].max() - summary["psd_distance_mean"].min()) * 0.23
    ax.set_xlim(summary["rmse_mean"].min() - x_pad, summary["rmse_mean"].max() + x_pad)
    ax.set_ylim(summary["psd_distance_mean"].min() - y_pad, summary["psd_distance_mean"].max() + y_pad)
    ax.set_xlabel("RMSE (mean +/- s.d.)")
    ax.set_ylabel("PSD distance (mean +/- s.d.)")
    ax.grid(axis="both", color=PALETTE["grid"], lw=0.4, alpha=0.55)
    ax.annotate(
        "better",
        xy=(0.09, 0.12),
        xytext=(0.25, 0.30),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "lw": 0.8, "color": PALETTE["neutral"]},
        ha="center",
        va="center",
        fontsize=6.5,
        color=PALETTE["neutral"],
    )
    _add_panel_label(ax, "a")


def _plot_directional_deltas(ax: plt.Axes, summary: pd.DataFrame) -> None:
    delta = _make_directional_delta_frame(summary)
    variant_order = [v for v in summary["variant"].tolist() if v != "baseline"]
    labels = {
        row["variant"]: _display_label(row["variant_label"])
        for _, row in summary.iterrows()
        if row["variant"] != "baseline"
    }
    metric_order = ["RMSE", "PSD distance", "HF improvement"]
    y_base = np.arange(len(variant_order))
    bar_height = 0.21
    offsets = [-bar_height, 0, bar_height]
    metric_colors = {
        "RMSE": "#B4C0E4",
        "PSD distance": "#AADCA9",
        "HF improvement": "#F0C0CC",
    }

    for offset, metric in zip(offsets, metric_order):
        values = []
        colors = []
        for variant in variant_order:
            value = float(delta.loc[(delta["variant"] == variant) & (delta["metric"] == metric), "directional_delta_percent"].iloc[0])
            values.append(value)
            colors.append(metric_colors[metric])
        ax.barh(
            y_base + offset,
            values,
            height=bar_height * 0.92,
            color=colors,
            edgecolor="white",
            linewidth=0.5,
            label=metric,
        )

    ax.axvline(0, color=PALETTE["black"], lw=0.75)
    ax.set_yticks(y_base)
    ax.set_yticklabels([labels[v] for v in variant_order])
    ax.invert_yaxis()
    ax.set_xlabel("Directional change vs final coefficients (%)")
    all_delta_values = delta["directional_delta_percent"].to_numpy(dtype=float)
    max_abs_delta = float(np.nanmax(np.abs(all_delta_values))) if len(all_delta_values) else 1.0
    ax.set_xlim(-max_abs_delta * 1.15, max_abs_delta * 1.15)
    ax.grid(axis="x", color=PALETTE["grid"], lw=0.4, alpha=0.55)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=3, handlelength=1.2, columnspacing=0.9)
    ax.text(
        0.985,
        0.03,
        "positive = better",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.4,
        color=PALETTE["neutral"],
    )
    _add_panel_label(ax, "b")


def _plot_seed_stability(ax: plt.Axes, summary: pd.DataFrame, per_seed: pd.DataFrame) -> None:
    order = summary.sort_values("rmse_mean")["variant"].tolist()
    labels = {
        row["variant"]: _display_label(row["variant_label"])
        for _, row in summary.iterrows()
    }
    y = np.arange(len(order))

    for idx, variant in enumerate(order):
        row = summary.loc[summary["variant"] == variant].iloc[0]
        color = TERM_COLORS.get(str(row["changed_term"]), PALETTE["neutral"])
        seed_values = per_seed.loc[per_seed["variant"] == variant, "rmse"].to_numpy()
        jitter = np.linspace(-0.12, 0.12, len(seed_values)) if len(seed_values) else np.array([])
        ax.scatter(
            seed_values,
            np.full_like(seed_values, idx, dtype=float) + jitter,
            s=12,
            color=color,
            alpha=0.42,
            linewidth=0,
            zorder=2,
        )
        ax.errorbar(
            row["rmse_mean"],
            idx,
            xerr=row["rmse_std"],
            fmt="o",
            ms=4.8,
            color=color,
            ecolor=color,
            elinewidth=1.0,
            capsize=2.2,
            mec="white",
            mew=0.5,
            zorder=4,
        )

    ax.set_yticks(y)
    ax.set_yticklabels([labels[v] for v in order])
    ax.invert_yaxis()
    ax.set_xlabel("RMSE across five seeds")
    ax.grid(axis="x", color=PALETTE["grid"], lw=0.4, alpha=0.55)
    _add_panel_label(ax, "c")


def make_figure(summary_path: Path, per_seed_path: Path, output_dir: Path, prefix: str) -> None:
    _apply_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary, per_seed = _load_tables(summary_path, per_seed_path)

    fig_height = max(5.2, 4.1 + 0.22 * len(summary))
    fig = plt.figure(figsize=(7.2, fig_height), constrained_layout=False)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.45], width_ratios=[1.25, 1.0], hspace=0.42, wspace=0.42)
    ax_tradeoff = fig.add_subplot(grid[0, :])
    ax_delta = fig.add_subplot(grid[1, 0])
    ax_stability = fig.add_subplot(grid[1, 1])

    _plot_tradeoff(ax_tradeoff, summary)
    _plot_directional_deltas(ax_delta, summary)
    _plot_seed_stability(ax_stability, summary, per_seed)

    handles = []
    labels = []
    for key, label in [("none", "Final coefficients"), ("time_l1", "L1 sweep"), ("mse", "MSE sweep"), ("spectral", "Spectral sweep")]:
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o" if key != "none" else "*",
                color="none",
                markerfacecolor=TERM_COLORS[key],
                markeredgecolor="white",
                markeredgewidth=0.7,
                markersize=6.5 if key != "none" else 8.5,
            )
        )
        labels.append(label)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.52, 0.985), ncol=4, frameon=False, columnspacing=1.2)
    fig.subplots_adjust(top=0.88, bottom=0.12, left=0.12, right=0.98)

    source_path = output_dir / f"{prefix}_source.csv"
    summary.to_csv(source_path, index=False)
    for ext, kwargs in {
        "svg": {},
        "pdf": {},
        "png": {"dpi": 600},
        "tiff": {"dpi": 600},
    }.items():
        fig.savefig(output_dir / f"{prefix}.{ext}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a Nature-style loss-weight sensitivity figure.")
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("outputs/supervised_tcn_causal_weight_sensitivity_local/weight_sensitivity_summary_numeric.csv"),
    )
    parser.add_argument(
        "--per-seed",
        type=Path,
        default=Path("outputs/supervised_tcn_causal_weight_sensitivity_local/weight_sensitivity_per_seed.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/supervised_tcn_causal_weight_sensitivity_local/figures"),
    )
    parser.add_argument("--prefix", type=str, default="loss_weight_sensitivity_nature")
    args = parser.parse_args()
    make_figure(
        summary_path=args.summary,
        per_seed_path=args.per_seed,
        output_dir=args.output_dir,
        prefix=args.prefix,
    )
    print(f"Wrote figure outputs to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
