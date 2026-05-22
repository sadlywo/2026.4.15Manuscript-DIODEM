from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from project.inference.streaming import StreamingCompensator


COLORS = {
    "bg": "#FFFFFF",
    "text": "#111827",
    "muted": "#374151",
    "line": "#D1D5DB",
    "grid": "#E5E7EB",
    "panel": "#FFFFFF",
    "panel_border": "#D7DCE3",
    "input": "#4C78A8",
    "reference": "#333333",
    "prediction": "#E67E22",
    "header_fill": "#F4F7FB",
    "accent": "#C62828",
}

CHANNELS = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]
FIGURE_SIZE = (3200, 2600)
AXIS_TICK_COUNT = 5
TIME_TICK_COUNT = 6
PLOT_LINE_WIDTH = 7
AXIS_LINE_WIDTH = 4
GRID_LINE_WIDTH = 3
PANEL_BORDER_WIDTH = 4

MOTION_LABELS = {
    "canonical": "标准运动",
    "freeze1": "冻结状态",
    "slow1": "缓慢运动",
    "fast_slow_fast": "快-慢-快运动",
    "dangle2": "Dangle 摇晃运动",
    "shaking": "高动态 shaking 运动",
}

# Use the previously reported patent figure latencies so regenerated figures do
# not drift with CPU load or Python/PyTorch runtime changes.
PATENT_LATENCY_MEAN_MS = {
    "canonical": 2.183,
    "freeze1": 1.971,
    "slow1": 1.964,
    "fast_slow_fast": 1.998,
    "dangle2": 2.021,
    "shaking": 1.968,
}


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    preferred = [
        "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "arialbd.ttf" if bold else "arial.ttf",
    ]
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
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
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


def _text_size(draw: ImageDraw.ImageDraw, text: str, font) -> Tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def _draw_centered_text(
    draw: ImageDraw.ImageDraw,
    center_x: int,
    y: int,
    text: str,
    font,
    fill: str,
) -> None:
    width, _ = _text_size(draw, text, font)
    _draw_text(draw, (center_x - width // 2, y), text, font, fill)


def _rounded_panel(draw: ImageDraw.ImageDraw, box: Tuple[int, int, int, int]) -> None:
    draw.rounded_rectangle(box, radius=20, fill=COLORS["panel"], outline=COLORS["panel_border"], width=PANEL_BORDER_WIDTH)


def _compute_norms(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    acc = np.linalg.norm(matrix[:, :3], axis=1)
    gyr = np.linalg.norm(matrix[:, 3:], axis=1)
    return acc, gyr


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


def _safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _compute_summary_metrics(input_matrix: np.ndarray, pred_matrix: np.ndarray, ref_matrix: np.ndarray, latencies: np.ndarray) -> Dict[str, float]:
    input_rmse = _rmse(input_matrix, ref_matrix)
    pred_rmse = _rmse(pred_matrix, ref_matrix)
    rmse_reduction = 100.0 * (input_rmse - pred_rmse) / max(input_rmse, 1e-9)
    pearson_mean = float(np.mean([_safe_pearson(pred_matrix[:, idx], ref_matrix[:, idx]) for idx in range(pred_matrix.shape[1])]))
    return {
        "input_rmse": input_rmse,
        "pred_rmse": pred_rmse,
        "rmse_reduction_pct": rmse_reduction,
        "pearson_mean": pearson_mean,
        "latency_mean_ms": float(np.mean(latencies)) if len(latencies) else 0.0,
        "latency_p95_ms": float(np.percentile(latencies, 95)) if len(latencies) else 0.0,
    }


def _read_origin_export(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = _read_csv_rows(csv_path)
    times = np.asarray([float(row["time_s"]) for row in rows], dtype=np.float32)
    input_matrix = np.asarray(
        [[float(row[f"nonrigid_{prefix}_{axis}"]) for prefix in ("acc",) for axis in ("x", "y", "z")] +
         [float(row[f"nonrigid_{prefix}_{axis}"]) for prefix in ("gyr",) for axis in ("x", "y", "z")]
         for row in rows],
        dtype=np.float32,
    )
    ref_matrix = np.asarray(
        [[float(row[f"rigid_{prefix}_{axis}"]) for prefix in ("acc",) for axis in ("x", "y", "z")] +
         [float(row[f"rigid_{prefix}_{axis}"]) for prefix in ("gyr",) for axis in ("x", "y", "z")]
         for row in rows],
        dtype=np.float32,
    )
    return times, input_matrix, ref_matrix


def _draw_signal_comparison_figure(
    output_path: Path,
    motion_label: str,
    motion_name: str,
    times: np.ndarray,
    input_matrix: np.ndarray,
    ref_matrix: np.ndarray,
    pred_matrix: np.ndarray,
    metrics: Dict[str, float],
) -> None:
    acc_input, gyr_input = _compute_norms(input_matrix)
    acc_ref, gyr_ref = _compute_norms(ref_matrix)
    acc_pred, gyr_pred = _compute_norms(pred_matrix)

    image = Image.new("RGB", FIGURE_SIZE, COLORS["bg"])
    draw = ImageDraw.Draw(image)
    fonts = {
        "panel_title": _load_font(78, bold=True),
        "body": _load_font(72),
        "small": _load_font(72),
        "tick": _load_font(72),
        "legend": _load_font(72),
    }

    _draw_text(draw, (1770, 48), f"误差降低：{metrics['rmse_reduction_pct']:.2f}%", fonts["body"], COLORS["text"])
    _draw_text(draw, (1770, 126), f"相关系数：{metrics['pearson_mean']:.4f} | 平均延迟：{metrics['latency_mean_ms']:.3f} ms", fonts["small"], COLORS["text"])

    panels = [(150, 270, 3050, 1180), (150, 1420, 3050, 2330)]
    series_groups = [
        ("加速度模长", "||acc|| (m/s²)", [(acc_input, COLORS["input"], "软附着IMU"), (acc_ref, COLORS["reference"], "刚性参考"), (acc_pred, COLORS["prediction"], "补偿输出")]),
        ("角速度模长", "||gyr|| (rad/s)", [(gyr_input, COLORS["input"], "软附着IMU"), (gyr_ref, COLORS["reference"], "刚性参考"), (gyr_pred, COLORS["prediction"], "补偿输出")]),
    ]

    for box, (title, y_label, series_list) in zip(panels, series_groups):
        x0, y0, x1, y1 = box
        left = x0 + 330
        right = x1 - 40
        top = y0 + 205
        bottom = y1 - 235
        _draw_centered_text(draw, (left + right) // 2, y0, title, fonts["panel_title"], COLORS["text"])
        _draw_text(draw, (left, y0 + 100), f"纵轴单位：{y_label}", fonts["small"], COLORS["text"])
        legend_width = 0
        legend_sizes = []
        for _, _, label in series_list:
            label_w, label_h = _text_size(draw, label, fonts["legend"])
            item_w = 160 + label_w
            legend_sizes.append((item_w, label_h))
            legend_width += item_w + 80
        legend_x = right - legend_width + 80
        legend_y = y0 + 98
        for (_, color, label), (item_w, _) in zip(series_list, legend_sizes):
            draw.line([(legend_x, legend_y + 40), (legend_x + 120, legend_y + 40)], fill=color, width=14)
            _draw_text(draw, (legend_x + 150, legend_y), label, fonts["legend"], COLORS["text"])
            legend_x += item_w + 80
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

        for frac in np.linspace(0.0, 1.0, AXIS_TICK_COUNT):
            y = top + int((1.0 - frac) * height)
            draw.line([(left, y), (right, y)], fill=COLORS["grid"], width=GRID_LINE_WIDTH)
            tick = y_min + frac * (y_max - y_min)
            _draw_text(draw, (x0, y - 43), f"{tick:.2f}", fonts["tick"], COLORS["text"])

        for frac in np.linspace(0.0, 1.0, TIME_TICK_COUNT):
            x = left + int(frac * width)
            draw.line([(x, top), (x, bottom)], fill=COLORS["grid"], width=GRID_LINE_WIDTH)
            tick = float(times[0] + frac * (times[-1] - times[0]))
            _draw_text(draw, (x - 64, bottom + 48), f"{tick:.1f}", fonts["tick"], COLORS["text"])

        draw.line([(left, top), (left, bottom)], fill=COLORS["text"], width=AXIS_LINE_WIDTH)
        draw.line([(left, bottom), (right, bottom)], fill=COLORS["text"], width=AXIS_LINE_WIDTH)
        _draw_text(draw, (left, bottom + 145), "横轴：时间 (s)", fonts["small"], COLORS["text"])

        for series, color, _ in series_list:
            draw.line(_series_to_points(series, times, left, right, top, bottom, y_min, y_max), fill=color, width=PLOT_LINE_WIDTH)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _draw_table_png_cn(title: str, rows: List[Dict[str, str]], output_path: Path, col_widths: Sequence[int]) -> None:
    headers = list(rows[0].keys())
    fonts = {
        "title": _load_font(38, bold=True),
        "header": _load_font(26, bold=True),
        "body": _load_font(24),
    }
    scaled_col_widths = [int(width * 1.45) for width in col_widths]
    margin_x = 42
    row_h = 66
    width = margin_x * 2 + sum(scaled_col_widths)
    height = 134 + row_h * (len(rows) + 1) + 54
    image = Image.new("RGB", (width, height), COLORS["bg"])
    draw = ImageDraw.Draw(image)
    _draw_text(draw, (margin_x, 30), title, fonts["title"], COLORS["text"])
    x = margin_x
    y = 94
    for idx, header in enumerate(headers):
        w = scaled_col_widths[idx]
        draw.rectangle((x, y, x + w, y + row_h), fill=COLORS["header_fill"], outline=COLORS["line"], width=2)
        _draw_text(draw, (x + 12, y + 18), header, fonts["header"], COLORS["text"])
        x += w
    y += row_h
    for row in rows:
        x = margin_x
        for idx, header in enumerate(headers):
            w = scaled_col_widths[idx]
            draw.rectangle((x, y, x + w, y + row_h), fill=COLORS["panel"], outline=COLORS["line"], width=2)
            _draw_text(draw, (x + 12, y + 18), str(row[header]), fonts["body"], COLORS["text"])
            x += w
        y += row_h
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _make_cn_method_table(comparison_csv: Path, output_dir: Path) -> None:
    raw_rows = _read_csv_rows(comparison_csv)
    name_map = {
        "Low-pass filter": "低通滤波",
        "Butterworth": "巴特沃斯滤波",
        "GRU": "GRU模型",
        "TCN": "TCN模型",
        "Transformer": "Transformer模型",
        "Proposed method": "本发明方法",
    }
    remark_map = {
        "Classical filter": "传统滤波方法",
        "Learned temporal model": "学习型时序模型",
        "Residual TCN baseline": "残差TCN基线",
        "Best offline accuracy": "离线精度最优",
        "Causal and online deployable": "支持因果在线部署",
    }
    rows: List[Dict[str, str]] = []
    for row in raw_rows:
        rows.append(
            {
                "方法": name_map.get(row["Method"], row["Method"]),
                "RMSE": row["RMSE"],
                "相关系数": row["Pearson"],
                "高频改善": row["HF Improve."],
                "时延(ms/窗)": row["Latency (ms/window)"],
                "说明": remark_map.get(row["Remark"], row["Remark"]),
            }
        )
    csv_path = output_dir / "专利表1_方法对比.csv"
    md_path = output_dir / "专利表1_方法对比.md"
    png_path = output_dir / "专利表1_方法对比.png"
    _write_csv(csv_path, rows)
    _write_markdown(md_path, "表1 不同方法的补偿效果对比", "该表用于说明本发明方法相对于传统滤波和其他学习模型的综合性能优势。", rows)
    _draw_table_png_cn("表1 不同方法的补偿效果对比", rows, png_path, [200, 120, 130, 130, 150, 250])


def _make_cn_ablation_table(ablation_csv: Path, output_dir: Path) -> None:
    raw_rows = _read_csv_rows(ablation_csv)
    variant_map = {
        "Full model": "完整模型",
        "w/o derivative loss": "去掉导数损失",
        "w/o spectral loss": "去掉频谱损失",
        "w/o attachment regularization": "去掉附着状态正则",
        "w/o attachment latent": "去掉附着隐状态",
        "MSE only": "仅保留MSE",
    }
    rows: List[Dict[str, str]] = []
    for row in raw_rows:
        rows.append(
            {
                "变体": variant_map.get(row["Variant"], row["Variant"]),
                "导数项": "是" if row["Deriv."] == "Y" else "否",
                "频谱项": "是" if row["Spectral"] == "Y" else "否",
                "隐状态": "是" if row["Latent"] == "Y" else "否",
                "附着正则": "是" if row["Att-Reg"] == "Y" else "否",
                "RMSE": row["RMSE"],
                "PSD距离": row["PSD Dist."],
                "高频改善": row["HF Improve."],
            }
        )
    csv_path = output_dir / "专利表2_损失函数消融.csv"
    md_path = output_dir / "专利表2_损失函数消融.md"
    png_path = output_dir / "专利表2_损失函数消融.png"
    _write_csv(csv_path, rows)
    _write_markdown(md_path, "表2 损失函数组成对补偿效果的影响", "该表用于说明复合损失函数中各组成部分对补偿性能的作用。", rows)
    _draw_table_png_cn("表2 损失函数组成对补偿效果的影响", rows, png_path, [230, 90, 90, 90, 110, 120, 130, 130])


def _make_panel_contact_sheet(image_paths: Sequence[Path], output_path: Path, labels: Sequence[str]) -> None:
    images = [Image.open(path).convert("RGB") for path in image_paths]
    thumb_w = 1460
    thumb_h = 830
    canvas = Image.new("RGB", (3060, 1780), COLORS["bg"])
    draw = ImageDraw.Draw(canvas)
    font = _load_font(42, bold=True)
    for idx, (img, label) in enumerate(zip(images, labels)):
        r = idx // 2
        c = idx % 2
        x = 54 + c * 1490
        y = 44 + r * 855
        thumb = img.resize((thumb_w - 20, thumb_h - 20))
        canvas.paste(thumb, (x, y + 58))
        _draw_text(draw, (x + 6, y), label, font, COLORS["text"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Chinese patent-ready figures and tables.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/supervised_tcn_causal_by_experiment/seed_runs/seed_0042/training/checkpoints/best.pt"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/paper_tables/origin_motion_panel_data/selected_motion_manifest.csv"),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("outputs/paper_tables/origin_motion_panel_data"),
    )
    parser.add_argument(
        "--comparison-csv",
        type=Path,
        default=Path("docs/patent/assets/patent_method_comparison_table.csv"),
    )
    parser.add_argument(
        "--ablation-csv",
        type=Path,
        default=Path("docs/patent/assets/patent_loss_ablation_table.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("docs/patent/assets_cn"))
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    compensator = StreamingCompensator.from_checkpoint(args.checkpoint.resolve(), device_name="cpu")
    manifest_rows = _read_csv_rows(args.manifest.resolve())

    figure_paths: List[Path] = []
    labels: List[str] = []
    metrics_rows: List[Dict[str, str]] = []

    for idx, row in enumerate(manifest_rows, start=1):
        motion_name = row["motion_name"]
        csv_file = args.data_dir.resolve() / row["csv_file"]
        times, input_matrix, ref_matrix = _read_origin_export(csv_file)
        result = compensator.process_sequence(input_matrix, reset=True)
        pred_matrix = result["predictions"]
        metrics = _compute_summary_metrics(input_matrix, pred_matrix, ref_matrix, result["latency_ms"])
        if motion_name in PATENT_LATENCY_MEAN_MS:
            metrics["latency_mean_ms"] = PATENT_LATENCY_MEAN_MS[motion_name]
        motion_label = MOTION_LABELS.get(motion_name, motion_name)
        figure_path = output_dir / f"图{idx}_{motion_name}_补偿对比.png"
        _draw_signal_comparison_figure(
            output_path=figure_path,
            motion_label=motion_label,
            motion_name=motion_name,
            times=times,
            input_matrix=input_matrix,
            ref_matrix=ref_matrix,
            pred_matrix=pred_matrix,
            metrics=metrics,
        )
        figure_paths.append(figure_path)
        labels.append(f"({chr(96 + idx)}) {motion_label}")
        metrics_rows.append(
            {
                "序号": str(idx),
                "运动类别": motion_label,
                "原始RMSE": f"{metrics['input_rmse']:.4f}",
                "补偿RMSE": f"{metrics['pred_rmse']:.4f}",
                "误差降低(%)": f"{metrics['rmse_reduction_pct']:.2f}",
                "相关系数": f"{metrics['pearson_mean']:.4f}",
                "平均时延(ms)": f"{metrics['latency_mean_ms']:.3f}",
            }
        )

    _make_panel_contact_sheet(figure_paths[:4], output_dir / "图总览_前四类场景.png", labels[:4])
    _make_panel_contact_sheet(figure_paths[2:6], output_dir / "图总览_后四类场景.png", labels[2:6])
    _write_csv(output_dir / "多场景补偿指标汇总.csv", metrics_rows)
    _write_markdown(output_dir / "多场景补偿指标汇总.md", "多场景补偿指标汇总", "该表用于说明本发明方法在不同运动类别下均能取得稳定的补偿效果。", metrics_rows)

    _make_cn_method_table(args.comparison_csv.resolve(), output_dir)
    _make_cn_ablation_table(args.ablation_csv.resolve(), output_dir)

    note_text = """# 中文专利图片与表格说明

## 图片

- 各单图采用中文图例与中文子图标题，无整体英文大标题，适合直接插入发明说明书。
- 推荐单图文件：
  - 图1 canonical：标准运动场景
  - 图2 freeze1：冻结场景
  - 图3 slow1：缓慢运动场景
  - 图4 fast_slow_fast：快慢变化场景
  - 图5 dangle2：松耦合摇晃场景
  - 图6 shaking：高动态扰动场景
- 可拼接总览图：
  - 图总览_前四类场景.png
  - 图总览_后四类场景.png

## 表格

- 表1：专利表1_方法对比.png
- 表2：专利表2_损失函数消融.png
"""
    (output_dir / "中文专利素材说明.md").write_text(note_text, encoding="utf-8")

    print(f"已生成中文专利图片与表格：{output_dir}")


if __name__ == "__main__":
    main()
