from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle


VARIANT_LABELS = {
    "no_l1_loss": "w/o L1",
    "no_mse_loss": "w/o MSE",
    "no_spectral_loss": "w/o spectral",
    "mse_only": "MSE only",
    "no_attachment_latent": "w/o attach. latent",
}

VARIANT_ORDER = [
    "no_l1_loss",
    "no_mse_loss",
    "no_spectral_loss",
    "mse_only",
    "no_attachment_latent",
]

METRICS = [
    ("rmse_mean", "RMSE\npenalty", "lower"),
    ("psd_distance_mean", "PSD\npenalty", "lower"),
    ("hf_ratio_improvement_mean", "HF loss", "higher"),
    ("acc_norm_rmse", "Acc norm\npenalty", "lower"),
    ("gyr_norm_rmse", "Gyro norm\npenalty", "lower"),
]

PALETTE = {
    "green": "#66BC98",
    "sage": "#AAD09D",
    "lime": "#E3EA96",
    "sand": "#FCD089",
    "orange": "#F6A56E",
    "red": "#E86E67",
    "ink": "#1D2A2E",
    "muted": "#6B7680",
    "grid": "#D8E2EA",
    "paper": "#FBFCFA",
}


def _style() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 7.2,
            "axes.linewidth": 0.72,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 2.4,
            "ytick.major.size": 2.4,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "legend.frameon": False,
        }
    )


def _load_summary(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "variant_name" not in frame.columns and "Variant" in frame.columns:
        frame = _load_figure_source_table(frame)
    required = {"variant_name", "rmse_mean", "psd_distance_mean", "hf_ratio_improvement_mean", "acc_norm_rmse", "gyr_norm_rmse"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    return frame


def _parse_first_number(value: object) -> float:
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(value))
    if not match:
        return float("nan")
    return float(match.group(0))


def _load_figure_source_table(frame: pd.DataFrame) -> pd.DataFrame:
    display_to_variant = {
        "Full model": "full_model",
        "w/o L1 loss": "no_l1_loss",
        "w/o MSE loss": "no_mse_loss",
        "w/o spectral loss": "no_spectral_loss",
        "MSE only": "mse_only",
        "w/o attachment latent": "no_attachment_latent",
    }
    converted = pd.DataFrame(
        {
            "variant_name": frame["Variant"].map(display_to_variant),
            "rmse_mean": pd.to_numeric(frame.get("RMSE Mean", frame["RMSE"].map(_parse_first_number)), errors="coerce"),
            "psd_distance_mean": pd.to_numeric(
                frame.get("PSD Dist. Mean", frame["PSD Dist."].map(_parse_first_number)),
                errors="coerce",
            ),
            "hf_ratio_improvement_mean": pd.to_numeric(
                frame.get("HF Improve. Mean", frame["HF Improve."].map(_parse_first_number)),
                errors="coerce",
            ),
            "acc_norm_rmse": frame["Acc Norm RMSE"].map(_parse_first_number),
            "gyr_norm_rmse": frame["Gyr Norm RMSE"].map(_parse_first_number),
        }
    )
    return converted.dropna(subset=["variant_name"]).reset_index(drop=True)


def _build_fingerprint(summary: pd.DataFrame) -> pd.DataFrame:
    full = summary.loc[summary["variant_name"] == "full_model"].iloc[0]
    rows = []
    for variant in VARIANT_ORDER:
        row = summary.loc[summary["variant_name"] == variant].iloc[0]
        metric_values: dict[str, float] = {}
        for column, label, direction in METRICS:
            if direction == "lower":
                change = (float(row[column]) - float(full[column])) / float(full[column]) * 100.0
            else:
                change = (float(full[column]) - float(row[column])) / float(full[column]) * 100.0
            metric_values[label] = change
        positive_values = [max(value, 0.0) for value in metric_values.values()]
        rows.append(
            {
                "variant_name": variant,
                "variant_label": VARIANT_LABELS[variant],
                **metric_values,
                "Composite degradation": float(np.mean(positive_values)),
            }
        )
    return pd.DataFrame(rows)


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.075,
        1.045,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.5,
        fontweight="bold",
        color=PALETTE["ink"],
    )


def _draw_fingerprint(ax: plt.Axes, fp: pd.DataFrame) -> None:
    metric_labels = [label for _, label, _ in METRICS]
    matrix = fp[metric_labels].to_numpy(dtype=float)
    rows, cols = matrix.shape
    cmap = LinearSegmentedColormap.from_list("impact", [PALETTE["green"], "#F5F5EE", PALETTE["orange"], PALETTE["red"]])
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=22.0)

    for i in range(rows):
        for j in range(cols):
            ax.add_patch(
                Rectangle(
                    (j - 0.5, i - 0.5),
                    1.0,
                    1.0,
                    facecolor=PALETTE["paper"],
                    edgecolor="white",
                    linewidth=1.6,
                    zorder=0,
                )
            )
            value = matrix[i, j]
            size = 170 + min(abs(value), 22.0) / 22.0 * 780
            ax.scatter(
                j,
                i,
                s=size,
                color=cmap(norm(value)),
                edgecolor=PALETTE["ink"],
                linewidth=0.72,
                zorder=2,
            )
            label = f"{value:+.1f}%"
            text_color = "white" if value > 9.0 else PALETTE["ink"]
            ax.text(j, i, label, ha="center", va="center", fontsize=6.5, fontweight="bold", color=text_color, zorder=3)

    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)
    ax.set_xticks(np.arange(cols))
    ax.set_xticklabels(metric_labels, fontsize=7.0)
    ax.set_yticks(np.arange(rows))
    ax.set_yticklabels(fp["variant_label"], fontsize=7.4)
    ax.tick_params(length=0)
    ax.set_title("Ablation impact fingerprint", loc="left", fontsize=9.2, fontweight="bold", color=PALETTE["ink"], pad=13)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.text(
        0.0,
        1.015,
        "Cell value = directional performance loss relative to the full model",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.5,
        color=PALETTE["muted"],
    )
    _panel_label(ax, "a")


def _draw_composite(ax: plt.Axes, fp: pd.DataFrame) -> None:
    plot_frame = fp.sort_values("Composite degradation", ascending=True).reset_index(drop=True)
    y = np.arange(len(plot_frame))
    values = plot_frame["Composite degradation"].to_numpy(dtype=float)
    colors = [PALETTE["red"] if value > 5 else PALETTE["orange"] if value > 1.5 else PALETTE["sage"] for value in values]

    ax.hlines(y, 0, values, color="#B7C5CF", lw=1.0, zorder=1)
    ax.scatter(values, y, s=74, color=colors, edgecolor=PALETTE["ink"], linewidth=0.65, zorder=3)
    for yi, value in zip(y, values):
        ax.text(value + 0.30, yi, f"{value:.1f}%", va="center", ha="left", fontsize=6.8, color=PALETTE["ink"])

    ax.set_yticks(y)
    ax.set_yticklabels(plot_frame["variant_label"], fontsize=7.1)
    ax.set_xlabel("Mean positive penalty (%)")
    ax.set_xlim(0, max(values) * 1.25)
    ax.grid(axis="x", color=PALETTE["grid"], lw=0.55, ls=(0, (2.2, 2.8)), zorder=0)
    ax.set_title("Overall impact", loc="left", fontsize=9.2, fontweight="bold", color=PALETTE["ink"], pad=13)
    _panel_label(ax, "b")


def make_figure(summary_path: Path, output_dir: Path, prefix: str) -> dict[str, Path]:
    _style()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = _load_summary(summary_path)
    fp = _build_fingerprint(summary)

    fig = plt.figure(figsize=(8.7, 3.35), constrained_layout=False)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.78, 0.92], wspace=0.30)
    ax_fp = fig.add_subplot(grid[0, 0])
    ax_comp = fig.add_subplot(grid[0, 1])

    _draw_fingerprint(ax_fp, fp)
    _draw_composite(ax_comp, fp)
    fig.subplots_adjust(left=0.090, right=0.984, top=0.835, bottom=0.175)

    stem = output_dir / prefix
    outputs = {
        "source": stem.with_name(f"{prefix}_source.csv"),
        "svg": stem.with_suffix(".svg"),
        "pdf": stem.with_suffix(".pdf"),
        "png": stem.with_suffix(".png"),
        "tiff": stem.with_suffix(".tiff"),
    }
    fp.to_csv(outputs["source"], index=False)
    fig.savefig(outputs["svg"], bbox_inches="tight")
    fig.savefig(outputs["pdf"], bbox_inches="tight")
    fig.savefig(outputs["png"], dpi=600, bbox_inches="tight")
    fig.savefig(outputs["tiff"], dpi=600, bbox_inches="tight")
    plt.close(fig)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a Nature-style ablation impact fingerprint figure.")
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("outputs/loss_ablation/figures/loss_ablation_three_panel_figure_source.csv"),
        help="Input ablation summary table or figure-source CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/loss_ablation/figures"),
        help="Directory for exported figure files.",
    )
    parser.add_argument("--prefix", type=str, default="loss_ablation_impact_fingerprint_nature")
    args = parser.parse_args()
    outputs = make_figure(args.summary, args.output_dir, args.prefix)
    print("Saved ablation impact fingerprint figure:")
    for kind, path in outputs.items():
        print(f"  {kind}: {path}")


if __name__ == "__main__":
    main()
