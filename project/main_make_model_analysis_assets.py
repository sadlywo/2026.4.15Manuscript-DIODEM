from __future__ import annotations

import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "outputs" / "paper_tables"
FIGURE_DIR = ROOT / "outputs" / "figures"


MODEL_LABELS = {
    "transformer": "Transformer",
    "tcn_causal": "TCN-causal",
    "tcn": "TCN",
    "gru": "GRU",
    "lowpass": "Low-pass",
    "butterworth": "Butterworth",
    "savgol": "Savitzky-Golay",
    "wiener": "Wiener",
    "identity": "Raw input",
}

MODEL_COLORS = {
    "transformer": "#3B6FB6",
    "tcn_causal": "#D6782A",
    "tcn": "#6F8798",
    "gru": "#8A7AA8",
    "lowpass": "#9EA7AD",
    "butterworth": "#C7CDD1",
    "savgol": "#C7CDD1",
    "wiener": "#C7CDD1",
    "identity": "#D7D7D7",
}


def parse_mean(value: object) -> float:
    text = str(value).strip()
    match = re.match(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", text)
    if not match:
        return float("nan")
    return float(match.group(1))


def parse_std(value: object) -> float:
    text = str(value).strip()
    if "+-" not in text:
        return 0.0
    return float(text.split("+-", 1)[1].strip())


def parse_int(value: object) -> int:
    text = str(value).replace(",", "").strip()
    return int(float(text)) if text else 0


def add_panel_label(ax, label: str) -> None:
    ax.text(
        -0.12,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="top",
        ha="left",
    )


def clean_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=6, width=0.6, length=3)


def make_assets() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    table1 = pd.read_csv(TABLE_DIR / "table1_by_experiment_full_comparison.csv")
    table2 = pd.read_csv(TABLE_DIR / "table2_generalization_core_models.csv")
    table3 = pd.read_csv(TABLE_DIR / "table3_deployment_streaming_summary.csv")

    for frame in (table1, table2):
        frame["rmse_mean"] = frame["RMSE"].map(parse_mean)
        frame["rmse_std"] = frame["RMSE"].map(parse_std)
        if "PSD Dist." in frame:
            frame["psd_mean"] = frame["PSD Dist."].map(parse_mean)
        if "HF Improve." in frame:
            frame["hf_mean"] = frame["HF Improve."].map(parse_mean)
    table1["params"] = table1["Params"].map(parse_int)
    table1["cpu_ms"] = table1["CPU ms/window"].map(parse_mean)
    table3["params"] = table3["Params"].map(parse_int)
    table3["stream_mean"] = table3["Streaming mean ms/step"].map(parse_mean)
    table3["stream_p95"] = table3["Streaming p95 ms/step"].map(parse_mean)

    summary_rows = []
    for model in ["transformer", "tcn_causal", "tcn", "gru", "lowpass"]:
        row = table1.loc[table1["Model"] == model].iloc[0]
        summary_rows.append(
            {
                "Model": MODEL_LABELS[model],
                "Role in manuscript": {
                    "transformer": "Offline accuracy reference",
                    "tcn_causal": "Deployment-oriented proposed model",
                    "tcn": "Non-causal TCN comparison",
                    "gru": "Recurrent learning baseline",
                    "lowpass": "Strongest classical filter",
                }[model],
                "RMSE": row["RMSE"],
                "PSD distance": row["PSD Dist."],
                "Parameters": row["Params"],
                "CPU ms/window": row["CPU ms/window"],
            }
        )
    summary_table = pd.DataFrame(summary_rows)
    summary_table.to_csv(FIGURE_DIR / "model_analysis_compact_table.csv", index=False)
    with (FIGURE_DIR / "model_analysis_compact_table.md").open("w", encoding="utf-8") as handle:
        handle.write(summary_table.to_markdown(index=False))
        handle.write("\n")

    source_data = []
    for _, row in table1.iterrows():
        source_data.append(
            {
                "panel": "a,c",
                "setting": "by_experiment",
                "model": row["Model"],
                "rmse_mean": row["rmse_mean"],
                "rmse_std": row["rmse_std"],
                "parameter_count": row["params"],
                "cpu_ms_window": row["cpu_ms"],
                "psd_distance": parse_mean(row["PSD Dist."]),
            }
        )
    for _, row in table2.iterrows():
        source_data.append(
            {
                "panel": "b",
                "setting": row["Setting"],
                "model": row["Model"],
                "rmse_mean": row["rmse_mean"],
                "rmse_std": row["rmse_std"],
                "parameter_count": np.nan,
                "cpu_ms_window": np.nan,
                "psd_distance": parse_mean(row["PSD Dist."]),
            }
        )
    source_frame = pd.DataFrame(source_data)
    source_frame.to_csv(FIGURE_DIR / "model_analysis_figure_source_data.csv", index=False)

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.linewidth": 0.7,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "figure.dpi": 150,
        }
    )

    fig = plt.figure(figsize=(7.2, 4.8), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0], height_ratios=[1.0, 1.0])
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    # Panel a: standard split reconstruction performance.
    order_a = ["transformer", "tcn_causal", "tcn", "gru", "lowpass", "butterworth", "savgol", "wiener", "identity"]
    panel_a = table1.set_index("Model").loc[order_a].reset_index()
    y = np.arange(len(panel_a))
    ax_a.barh(
        y,
        panel_a["rmse_mean"],
        xerr=panel_a["rmse_std"].replace(0, np.nan),
        color=[MODEL_COLORS[m] for m in panel_a["Model"]],
        edgecolor="white",
        linewidth=0.5,
        error_kw={"elinewidth": 0.7, "capsize": 2, "capthick": 0.7},
    )
    ax_a.set_yticks(y)
    ax_a.set_yticklabels([MODEL_LABELS[m] for m in panel_a["Model"]])
    ax_a.invert_yaxis()
    ax_a.set_xlabel("RMSE")
    ax_a.set_title("Standard split")
    ax_a.axvline(0.5575, color="#9EA7AD", linewidth=0.8, linestyle="--")
    ax_a.text(0.565, 4.6, "Low-pass", fontsize=5.8, color="#6A7378", va="center")
    clean_axes(ax_a)
    add_panel_label(ax_a, "a")

    # Panel b: protocol-level generalization.
    settings = ["by_experiment", "by_motion_type", "anomaly_test_only"]
    setting_labels = ["By experiment", "By motion", "Anomaly only"]
    x = np.arange(len(settings))
    for model in ["transformer", "tcn_causal", "tcn", "lowpass"]:
        model_frame = table2.loc[table2["Model"] == model].set_index("Setting").loc[settings]
        ax_b.plot(
            x,
            model_frame["rmse_mean"],
            marker="o",
            markersize=3.4,
            linewidth=1.2 if model == "tcn_causal" else 0.9,
            color=MODEL_COLORS[model],
            label=MODEL_LABELS[model],
        )
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(setting_labels, rotation=20, ha="right")
    ax_b.set_ylabel("RMSE")
    ax_b.set_title("Generalization protocols")
    ax_b.legend(loc="upper left", ncol=1, handlelength=1.5, borderaxespad=0.2)
    clean_axes(ax_b)
    add_panel_label(ax_b, "b")

    # Panel c: accuracy and model footprint.
    panel_c = table1.loc[table1["Model"].isin(["transformer", "tcn_causal", "tcn", "gru", "lowpass"])]
    label_offsets = {
        "transformer": (7, -0.004),
        "tcn_causal": (5, -0.012),
        "tcn": (5, 0.014),
        "gru": (6, 0.010),
        "lowpass": (4, 0.012),
    }
    for _, row in panel_c.iterrows():
        model = row["Model"]
        params_k = row["params"] / 1000.0
        ax_c.scatter(
            params_k,
            row["rmse_mean"],
            s=90 if model == "tcn_causal" else 60,
            color=MODEL_COLORS[model],
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
        dx, dy = label_offsets[model]
        ax_c.text(params_k + dx, row["rmse_mean"] + dy, MODEL_LABELS[model], fontsize=6)
    ax_c.set_xlabel("Trainable parameters (x1000)")
    ax_c.set_ylabel("RMSE")
    ax_c.set_title("Accuracy-footprint trade-off")
    ax_c.set_xlim(-8, 430)
    ax_c.set_ylim(0.32, 0.60)
    ax_c.grid(axis="y", linewidth=0.3, color="#E6E6E6")
    clean_axes(ax_c)
    add_panel_label(ax_c, "c")

    # Panel d: streaming validation for the causal model.
    stream = table3.loc[table3["Model"] == "tcn_causal"].iloc[0]
    latency_names = ["Mean", "p95", "40 Hz budget"]
    latency_values = [stream["stream_mean"], stream["stream_p95"], 25.0]
    latency_colors = [MODEL_COLORS["tcn_causal"], "#E3A05A", "#D7D7D7"]
    ypos = np.arange(len(latency_names))
    ax_d.barh(ypos, latency_values, color=latency_colors, edgecolor="white", linewidth=0.5)
    ax_d.set_yticks(ypos)
    ax_d.set_yticklabels(latency_names)
    ax_d.invert_yaxis()
    ax_d.set_xlabel("Latency (ms/step)")
    ax_d.set_title("Streaming inference")
    ax_d.set_xlim(0, 26)
    for idx, value in enumerate(latency_values):
        ax_d.text(value + 0.35, idx, f"{value:.3g}", va="center", fontsize=6)
    ax_d.text(
        5.2,
        0.06,
        "Streaming vs offline\nRMSE = 3.0e-7\nmax abs = 1.8e-6",
        ha="left",
        va="center",
        fontsize=6,
        color="#333333",
    )
    clean_axes(ax_d)
    add_panel_label(ax_d, "d")

    fig.suptitle(
        "Model performance and deployment trade-off for IMU artifact compensation",
        fontsize=8,
        fontweight="bold",
    )

    output_stem = FIGURE_DIR / "model_analysis_performance_tradeoff"
    fig.savefig(f"{output_stem}.svg", bbox_inches="tight")
    fig.savefig(f"{output_stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{output_stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{output_stem}.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    make_assets()
