from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


TERM_META = {
    "time_l1": {"label": "L1 weight", "color": "#66BC98", "marker": "o"},
    "mse": {"label": "MSE weight", "color": "#E3EA96", "marker": "s"},
    "spectral": {"label": "Spectral weight", "color": "#FCD089", "marker": "^"},
}

METRICS = [
    {
        "mean": "rmse_mean",
        "std": "rmse_std",
        "label": "RMSE (lower)",
        "title": "Reconstruction error",
        "panel": "a",
    },
    {
        "mean": "psd_distance_mean",
        "std": "psd_distance_std",
        "label": "PSD distance (lower)",
        "title": "Spectral fidelity",
        "panel": "b",
    },
    {
        "mean": "hf_ratio_improvement_mean",
        "std": "hf_ratio_improvement_std",
        "label": "HF improvement (higher)",
        "title": "High-frequency suppression",
        "panel": "c",
    },
]


def _style() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 7.0,
            "axes.linewidth": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 2.6,
            "ytick.major.size": 2.6,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _load_summary(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "variant",
        "changed_term",
        "time_l1",
        "mse",
        "spectral",
        "rmse_mean",
        "rmse_std",
        "psd_distance_mean",
        "psd_distance_std",
        "hf_ratio_improvement_mean",
        "hf_ratio_improvement_std",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    return frame


def _curve_source(summary: pd.DataFrame) -> pd.DataFrame:
    baseline = summary.loc[summary["variant"].astype(str) == "baseline"].iloc[0]
    rows = []
    for term in TERM_META:
        term_rows = summary.loc[summary["changed_term"].astype(str) == term].copy()
        baseline_row = baseline.copy()
        baseline_row["changed_term"] = term
        baseline_row["variant"] = f"{term}_baseline"
        baseline_row["variant_label"] = "Final coefficients"
        term_rows = pd.concat([term_rows, baseline_row.to_frame().T], ignore_index=True)
        term_rows["swept_weight"] = pd.to_numeric(term_rows[term], errors="coerce")
        term_rows = term_rows.sort_values("swept_weight")
        for _, row in term_rows.iterrows():
            rows.append(
                {
                    "term": term,
                    "term_label": TERM_META[term]["label"],
                    "variant": row["variant"],
                    "variant_label": row.get("variant_label", ""),
                    "swept_weight": float(row["swept_weight"]),
                    "is_final": str(row["variant_label"]) == "Final coefficients",
                    "rmse_mean": float(row["rmse_mean"]),
                    "rmse_std": float(row["rmse_std"]),
                    "psd_distance_mean": float(row["psd_distance_mean"]),
                    "psd_distance_std": float(row["psd_distance_std"]),
                    "hf_ratio_improvement_mean": float(row["hf_ratio_improvement_mean"]),
                    "hf_ratio_improvement_std": float(row["hf_ratio_improvement_std"]),
                }
            )
    return pd.DataFrame(rows)


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.14,
        1.055,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.0,
        fontweight="bold",
        color="#182026",
    )


def _format_weight_tick(value: float) -> str:
    return f"{value:g}"


def _plot_metric(ax: plt.Axes, source: pd.DataFrame, metric: dict[str, str]) -> None:
    baseline_rows = source.loc[source["is_final"]]
    baseline_value = float(baseline_rows[metric["mean"]].iloc[0])
    baseline_std = float(baseline_rows[metric["std"]].iloc[0])
    ax.axhspan(
        baseline_value - baseline_std,
        baseline_value + baseline_std,
        color="#9AA0A6",
        alpha=0.10,
        lw=0,
        zorder=0,
    )
    ax.axhline(baseline_value, color="#3E4C59", lw=0.8, ls=(0, (2.5, 2.2)), zorder=1)

    for term, meta in TERM_META.items():
        group = source.loc[source["term"] == term].sort_values("swept_weight")
        x = group["swept_weight"].to_numpy(dtype=float)
        y = group[metric["mean"]].to_numpy(dtype=float)
        yerr = group[metric["std"]].to_numpy(dtype=float)
        ax.plot(
            x,
            y,
            color=meta["color"],
            lw=1.45,
            marker=meta["marker"],
            ms=4.4,
            mec="#182026",
            mew=0.45,
            mfc=meta["color"],
            zorder=3,
        )
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="none",
            ecolor=meta["color"],
            elinewidth=0.8,
            capsize=2.0,
            capthick=0.8,
            alpha=0.9,
            zorder=2,
        )

        final = group.loc[group["is_final"]]
        ax.scatter(
            final["swept_weight"].to_numpy(dtype=float),
            final[metric["mean"]].to_numpy(dtype=float),
            s=46,
            marker="D",
            color=meta["color"],
            edgecolor="#182026",
            linewidth=0.7,
            zorder=5,
        )

    ax.set_xscale("log")
    ticks = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    ax.set_xticks(ticks)
    ax.set_xticklabels([_format_weight_tick(t) for t in ticks])
    ax.set_xlim(0.043, 2.35)
    ax.set_xlabel("Swept loss weight (log scale)")
    ax.set_ylabel(metric["label"])
    ax.set_title(metric["title"], loc="left", fontsize=8.4, fontweight="bold", color="#182026")
    ax.grid(axis="both", which="major", color="#D9E2EC", lw=0.55, ls=(0, (2.2, 2.8)), zorder=0)
    ax.tick_params(axis="both", labelsize=6.8)
    _panel_label(ax, metric["panel"])


def make_figure(summary_path: Path, output_dir: Path, prefix: str) -> dict[str, Path]:
    _style()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = _load_summary(summary_path)
    source = _curve_source(summary)

    fig, axes = plt.subplots(1, 3, figsize=(8.9, 2.72), constrained_layout=False)
    for ax, metric in zip(axes, METRICS):
        _plot_metric(ax, source, metric)

    handles = [
        Line2D(
            [0],
            [0],
            color=meta["color"],
            marker=meta["marker"],
            lw=1.55,
            ms=4.8,
            mec="#182026",
            mew=0.45,
            label=meta["label"],
        )
        for meta in TERM_META.values()
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            marker="D",
            color="none",
            markerfacecolor="#F7F9FB",
            markeredgecolor="#182026",
            markeredgewidth=0.8,
            markersize=5.0,
            label="Final coefficient",
        )
    )
    handles.append(
        Line2D(
            [0],
            [0],
            color="#3E4C59",
            lw=0.8,
            ls=(0, (2.5, 2.2)),
            label="Final metric",
        )
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.52, 1.025),
        ncol=5,
        fontsize=7.1,
        handlelength=1.45,
        columnspacing=0.9,
    )
    fig.subplots_adjust(left=0.074, right=0.992, top=0.785, bottom=0.245, wspace=0.31)

    stem = output_dir / prefix
    outputs = {
        "source": stem.with_name(f"{prefix}_source.csv"),
        "svg": stem.with_suffix(".svg"),
        "pdf": stem.with_suffix(".pdf"),
        "png": stem.with_suffix(".png"),
        "tiff": stem.with_suffix(".tiff"),
    }
    source.to_csv(outputs["source"], index=False)
    fig.savefig(outputs["svg"], bbox_inches="tight")
    fig.savefig(outputs["pdf"], bbox_inches="tight")
    fig.savefig(outputs["png"], dpi=600, bbox_inches="tight")
    fig.savefig(outputs["tiff"], dpi=600, bbox_inches="tight")
    plt.close(fig)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a three-panel loss-weight sensitivity curve figure.")
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("outputs/supervised_tcn_causal_weight_sensitivity_local/weight_sensitivity_summary_numeric.csv"),
        help="Input multi-seed loss-weight sensitivity summary CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/supervised_tcn_causal_weight_sensitivity_local/figures"),
        help="Directory for exported figure files.",
    )
    parser.add_argument("--prefix", type=str, default="loss_weight_sensitivity_three_panel_curves")
    args = parser.parse_args()
    outputs = make_figure(args.summary, args.output_dir, args.prefix)
    print("Saved three-panel loss-weight sensitivity figure:")
    for kind, path in outputs.items():
        print(f"  {kind}: {path}")


if __name__ == "__main__":
    main()
