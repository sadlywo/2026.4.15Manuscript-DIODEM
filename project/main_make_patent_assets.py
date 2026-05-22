from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


COLORS = {
    "bg": "#FFFFFF",
    "text": "#111827",
    "muted": "#6B7280",
    "line": "#D1D5DB",
    "grid": "#E5E7EB",
    "panel": "#FFFFFF",
    "panel_border": "#D7DCE3",
    "input": "#4C78A8",
    "reference": "#222222",
    "prediction": "#E67E22",
    "header_fill": "#F4F7FB",
}


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    preferred = ["DejaVuSans-Bold.ttf", "arialbd.ttf"] if bold else ["DejaVuSans.ttf", "arial.ttf"]
    for name in preferred:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, title: str, intro: str, rows: List[Dict[str, str]]) -> None:
    headers = list(rows[0].keys())
    lines = [f"**{title}**", "", intro, ""]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row[h] for h in headers) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _draw_text(draw: ImageDraw.ImageDraw, xy: Tuple[int, int], text: str, font, fill: str) -> None:
    draw.text(xy, text, font=font, fill=fill)


def _rounded_panel(draw: ImageDraw.ImageDraw, box: Tuple[int, int, int, int]) -> None:
    draw.rounded_rectangle(box, radius=16, fill=COLORS["panel"], outline=COLORS["panel_border"], width=2)


def _series_to_points(
    values: np.ndarray,
    times: np.ndarray,
    x_left: int,
    x_right: int,
    y_top: int,
    y_bottom: int,
    y_min: float,
    y_max: float,
) -> List[Tuple[int, int]]:
    width = max(1, x_right - x_left)
    height = max(1, y_bottom - y_top)
    t0 = float(times[0])
    t_range = float(times[-1] - times[0]) if len(times) > 1 else 1.0
    y_range = float(y_max - y_min) if abs(y_max - y_min) > 1e-9 else 1.0
    points: List[Tuple[int, int]] = []
    for t, value in zip(times, values):
        x = x_left + int((float(t) - t0) / t_range * width)
        y = y_top + int((y_max - float(value)) / y_range * height)
        points.append((x, y))
    return points


def _compute_norms(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    acc = np.linalg.norm(matrix[:, :3], axis=1)
    gyr = np.linalg.norm(matrix[:, 3:], axis=1)
    return acc, gyr


def _make_patent_signal_figure(
    input_csv: Path,
    reference_csv: Path,
    prediction_csv: Path,
    summary_csv: Path,
    output_path: Path,
) -> None:
    input_rows = _read_csv_rows(input_csv)
    reference_rows = _read_csv_rows(reference_csv)
    prediction_rows = _read_csv_rows(prediction_csv)
    summary_rows = _read_csv_rows(summary_csv)

    input_matrix = np.asarray(
        [[float(row[k]) for k in ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]] for row in input_rows],
        dtype=np.float32,
    )
    reference_matrix = np.asarray(
        [[float(row[k]) for k in ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]] for row in reference_rows],
        dtype=np.float32,
    )
    prediction_matrix = np.asarray(
        [
            [float(row[k]) for k in ["pred_acc_x", "pred_acc_y", "pred_acc_z", "pred_gyr_x", "pred_gyr_y", "pred_gyr_z"]]
            for row in prediction_rows
        ],
        dtype=np.float32,
    )
    times = np.asarray([float(row["time_s"]) for row in input_rows], dtype=np.float32)

    acc_input, gyr_input = _compute_norms(input_matrix)
    acc_ref, gyr_ref = _compute_norms(reference_matrix)
    acc_pred, gyr_pred = _compute_norms(prediction_matrix)

    diff_score = np.abs(acc_input - acc_ref) + np.abs(gyr_input - gyr_ref)
    center = int(np.argmax(diff_score))
    window = min(len(times), 240)
    start = max(0, center - window // 2)
    end = min(len(times), start + window)
    start = max(0, end - window)

    times = times[start:end]
    acc_input = acc_input[start:end]
    acc_ref = acc_ref[start:end]
    acc_pred = acc_pred[start:end]
    gyr_input = gyr_input[start:end]
    gyr_ref = gyr_ref[start:end]
    gyr_pred = gyr_pred[start:end]

    image = Image.new("RGB", (1900, 1180), COLORS["bg"])
    draw = ImageDraw.Draw(image)
    fonts = {
        "title": _load_font(40, bold=True),
        "subtitle": _load_font(20),
        "panel_title": _load_font(24, bold=True),
        "body": _load_font(20),
        "small": _load_font(16),
        "tiny": _load_font(13),
    }

    _draw_text(draw, (70, 40), "Soft-Attached IMU, Rigid Reference, and Compensated Output", fonts["title"], COLORS["text"])
    _draw_text(
        draw,
        (72, 92),
        "Representative compensation example. The compensated output follows the rigid-reference signal more closely than the raw soft-attached input.",
        fonts["subtitle"],
        COLORS["muted"],
    )

    panels = [(70, 160, 1830, 610), (70, 660, 1830, 1110)]
    series_groups = [
        (
            "Acceleration magnitude",
            "||acc|| (m/s²)",
            [
                (acc_input, COLORS["input"], "Soft-attached IMU"),
                (acc_ref, COLORS["reference"], "Rigid reference"),
                (acc_pred, COLORS["prediction"], "Compensated output"),
            ],
        ),
        (
            "Gyroscope magnitude",
            "||gyr|| (rad/s)",
            [
                (gyr_input, COLORS["input"], "Soft-attached IMU"),
                (gyr_ref, COLORS["reference"], "Rigid reference"),
                (gyr_pred, COLORS["prediction"], "Compensated output"),
            ],
        ),
    ]

    for box, (title, y_label, series_list) in zip(panels, series_groups):
        _rounded_panel(draw, box)
        x0, y0, x1, y1 = box
        _draw_text(draw, (x0 + 24, y0 + 16), title, fonts["panel_title"], COLORS["text"])
        left = x0 + 92
        right = x1 - 28
        top = y0 + 60
        bottom = y1 - 54
        width = right - left
        height = bottom - top
        values = np.concatenate([series for series, _, _ in series_list], axis=0)
        y_min = float(np.min(values))
        y_max = float(np.max(values))
        if abs(y_max - y_min) < 1e-6:
            y_min -= 1.0
            y_max += 1.0
        pad = 0.08 * (y_max - y_min)
        y_min -= pad
        y_max += pad

        for frac in np.linspace(0.0, 1.0, 5):
            y = top + int((1.0 - frac) * height)
            draw.line([(left, y), (right, y)], fill=COLORS["grid"], width=1)
            tick = y_min + frac * (y_max - y_min)
            _draw_text(draw, (x0 + 14, y - 9), f"{tick:.2f}", fonts["tiny"], COLORS["muted"])
        for frac in np.linspace(0.0, 1.0, 6):
            x = left + int(frac * width)
            draw.line([(x, top), (x, bottom)], fill=COLORS["grid"], width=1)
            tick = float(times[0] + frac * (times[-1] - times[0]))
            _draw_text(draw, (x - 12, bottom + 10), f"{tick:.1f}", fonts["tiny"], COLORS["muted"])

        draw.line([(left, top), (left, bottom)], fill=COLORS["text"], width=2)
        draw.line([(left, bottom), (right, bottom)], fill=COLORS["text"], width=2)
        _draw_text(draw, (x0 + 18, y0 + 36), y_label, fonts["small"], COLORS["muted"])
        for series, color, _ in series_list:
            draw.line(_series_to_points(series, times, left, right, top, bottom, y_min, y_max), fill=color, width=3)

        legend_x = x0 + 26
        legend_y = y1 - 32
        for _, color, label in series_list:
            draw.line([(legend_x, legend_y + 8), (legend_x + 30, legend_y + 8)], fill=color, width=4)
            _draw_text(draw, (legend_x + 40, legend_y), label, fonts["small"], COLORS["muted"])
            legend_x += 270

    if summary_rows:
        row = summary_rows[0]
        summary_box = (1250, 42, 1830, 128)
        draw.rounded_rectangle(summary_box, radius=14, fill="#FBFCFE", outline=COLORS["line"], width=2)
        _draw_text(draw, (1270, 58), f"RMSE reduction: {row['RMSE Reduction (%)']}%", fonts["body"], COLORS["text"])
        _draw_text(draw, (1270, 88), f"Pearson: {row['Pearson']} | Latency mean: {row['Latency mean (ms)']} ms", fonts["body"], COLORS["text"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _draw_table_png(title: str, rows: List[Dict[str, str]], output_path: Path, col_widths: Sequence[int] | None = None) -> None:
    headers = list(rows[0].keys())
    fonts = {
        "title": _load_font(28, bold=True),
        "header": _load_font(18, bold=True),
        "body": _load_font(17),
    }
    if col_widths is None:
        col_widths = [180] * len(headers)
    width = 60 + sum(col_widths)
    row_h = 46
    height = 110 + row_h * (len(rows) + 1) + 40
    image = Image.new("RGB", (width, height), COLORS["bg"])
    draw = ImageDraw.Draw(image)
    _draw_text(draw, (30, 24), title, fonts["title"], COLORS["text"])

    x = 30
    y = 76
    for idx, header in enumerate(headers):
        w = col_widths[idx]
        draw.rectangle((x, y, x + w, y + row_h), fill=COLORS["header_fill"], outline=COLORS["line"], width=1)
        _draw_text(draw, (x + 8, y + 12), header, fonts["header"], COLORS["text"])
        x += w
    y += row_h

    for row in rows:
        x = 30
        for idx, header in enumerate(headers):
            w = col_widths[idx]
            draw.rectangle((x, y, x + w, y + row_h), fill=COLORS["panel"], outline=COLORS["line"], width=1)
            _draw_text(draw, (x + 8, y + 12), str(row[header]), fonts["body"], COLORS["text"])
            x += w
        y += row_h

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _make_method_comparison_tables(comparison_csv: Path, deployment_csv: Path, output_dir: Path) -> None:
    comparison_rows = _read_csv_rows(comparison_csv)
    deployment_rows = {row["model_name"]: row for row in _read_csv_rows(deployment_csv)}
    target_order = ["lowpass", "butterworth", "gru", "tcn", "transformer", "tcn_causal"]
    display_name = {
        "lowpass": "Low-pass filter",
        "butterworth": "Butterworth",
        "gru": "GRU",
        "tcn": "TCN",
        "transformer": "Transformer",
        "tcn_causal": "Proposed method",
    }
    remarks = {
        "lowpass": "Classical filter",
        "butterworth": "Classical filter",
        "gru": "Learned temporal model",
        "tcn": "Residual TCN baseline",
        "transformer": "Best offline accuracy",
        "tcn_causal": "Causal and online deployable",
    }
    rows: List[Dict[str, str]] = []
    for model_name in target_order:
        row = next(item for item in comparison_rows if item["model_name"] == model_name)
        dep = deployment_rows.get(model_name, {})
        rows.append(
            {
                "Method": display_name[model_name],
                "RMSE": f"{float(row['rmse_mean_mean']):.4f}",
                "Pearson": f"{float(row['pearson_mean_mean']):.4f}",
                "HF Improve.": f"{float(row['hf_ratio_improvement_mean_mean']):.3f}",
                "Latency (ms/window)": f"{float(dep.get('cpu_forward_ms_per_window_mean', 0.0)):.3f}",
                "Remark": remarks[model_name],
            }
        )

    csv_path = output_dir / "patent_method_comparison_table.csv"
    md_path = output_dir / "patent_method_comparison_table.md"
    png_path = output_dir / "patent_method_comparison_table.png"
    _write_csv(csv_path, rows)
    _write_markdown(
        md_path,
        "Table 1. Compensation Performance Comparison Across Methods",
        "This table compares the proposed method with representative filtering and learning-based methods in terms of compensation accuracy, high-frequency consistency improvement, and online deployability.",
        rows,
    )
    _draw_table_png("Table 1. Compensation Performance Comparison Across Methods", rows, png_path, col_widths=[220, 130, 130, 150, 180, 260])


def _make_loss_ablation_tables(ablation_csv: Path, output_dir: Path) -> None:
    raw_rows = _read_csv_rows(ablation_csv)
    rows: List[Dict[str, str]] = []
    for raw in raw_rows:
        rows.append(
            {
                "Variant": raw["Variant"],
                "Deriv.": raw["Deriv."],
                "Spectral": raw["Spectral"],
                "Latent": raw["Latent"],
                "Att-Reg": "Y" if raw["Att-L2"] == "Y" or raw["Att-Temp"] == "Y" else "N",
                "RMSE": raw["RMSE"],
                "PSD Dist.": raw["PSD Dist."],
                "HF Improve.": raw["HF Improve."],
            }
        )

    csv_path = output_dir / "patent_loss_ablation_table.csv"
    md_path = output_dir / "patent_loss_ablation_table.md"
    png_path = output_dir / "patent_loss_ablation_table.png"
    _write_csv(csv_path, rows)
    _write_markdown(
        md_path,
        "Table 2. Ablation of Composite Loss Components",
        "This table shows how different loss-function components affect compensation performance. Lower RMSE and PSD distance together with higher HF improvement indicate better overall behavior.",
        rows,
    )
    _draw_table_png("Table 2. Ablation of Composite Loss Components", rows, png_path, col_widths=[260, 100, 120, 100, 110, 120, 140, 150])


def _write_notes(path: Path, figure_path: Path, method_table_png: Path, ablation_table_png: Path) -> None:
    text = f"""# Patent Figure and Table Insertion Notes

## Figure Recommendation

- Suggested title: Comparison of soft-attached IMU signal, rigid-reference IMU signal, and compensated output signal
- File: [{figure_path.name}]({figure_path.as_posix()})
- Suggested description:
  This figure illustrates the relationship among the soft-attached IMU signal, the rigid-reference IMU signal, and the compensated output produced by the proposed method. The compensated output is visibly closer to the rigid-reference signal in both acceleration and gyroscope domains, indicating that the method effectively reduces measurement deviations caused by soft attachment, local slipping, and compliant coupling.

## Table 1 Recommendation

- Suggested title: Compensation performance comparison across methods
- File: [{method_table_png.name}]({method_table_png.as_posix()})
- Suggested description:
  Table 1 compares the proposed method with representative filtering methods and learning-based models. Relative to conventional filtering, the proposed method achieves better overall performance in error metrics, correlation, and high-frequency consistency while retaining causal and online deployment capability.

## Table 2 Recommendation

- Suggested title: Effect of loss-function components on compensation performance
- File: [{ablation_table_png.name}]({ablation_table_png.as_posix()})
- Suggested description:
  Table 2 shows the contribution of different loss-function components. The results indicate that relying only on a simple reconstruction term is insufficient, whereas introducing derivative consistency, spectral consistency, and attachment-state-related constraints leads to a more balanced performance in time-domain, frequency-domain, and dynamic-consistency metrics.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create patent-ready figures and tables from existing experiment results.")
    parser.add_argument("--input-csv", type=Path, default=Path("outputs/replay_demo/exp01_canonical_seg1_nonrigid_6ch.csv"))
    parser.add_argument("--reference-csv", type=Path, default=Path("outputs/replay_demo/exp01_canonical_seg1_rigid_6ch.csv"))
    parser.add_argument("--prediction-csv", type=Path, default=Path("outputs/replay_demo/exp01_canonical_seg1_predictions.csv"))
    parser.add_argument("--summary-csv", type=Path, default=Path("outputs/replay_demo/ppt_replay_summary_table.csv"))
    parser.add_argument("--comparison-csv", type=Path, default=Path("outputs/supervised_tcn_causal_by_experiment/evaluation/metrics/multiseed_model_comparison.csv"))
    parser.add_argument("--deployment-csv", type=Path, default=Path("outputs/supervised_tcn_causal_by_experiment/evaluation/metrics/multiseed_model_deployment_summary.csv"))
    parser.add_argument("--ablation-csv", type=Path, default=Path("outputs/paper_tables/supplementary_loss_ablation_table.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/patent/assets"))
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    figure_path = output_dir / "patent_signal_comparison_figure.png"
    _make_patent_signal_figure(
        input_csv=args.input_csv.resolve(),
        reference_csv=args.reference_csv.resolve(),
        prediction_csv=args.prediction_csv.resolve(),
        summary_csv=args.summary_csv.resolve(),
        output_path=figure_path,
    )
    _make_method_comparison_tables(args.comparison_csv.resolve(), args.deployment_csv.resolve(), output_dir)
    _make_loss_ablation_tables(args.ablation_csv.resolve(), output_dir)
    _write_notes(
        output_dir / "patent_assets_notes.md",
        figure_path=figure_path,
        method_table_png=output_dir / "patent_method_comparison_table.png",
        ablation_table_png=output_dir / "patent_loss_ablation_table.png",
    )

    print(f"Wrote patent figure and tables to {output_dir}")


if __name__ == "__main__":
    main()
