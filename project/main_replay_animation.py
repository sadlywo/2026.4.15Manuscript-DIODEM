from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from project.inference import StreamingCompensator
from project.utils.torch_compat import require_torch


CHANNELS = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]
PREDICTION_COLUMNS = ["pred_acc_x", "pred_acc_y", "pred_acc_z", "pred_gyr_x", "pred_gyr_y", "pred_gyr_z"]
COLORS = {
    "input": "#4C78A8",
    "prediction": "#E67E22",
    "reference": "#222222",
    "grid": "#D6DADF",
    "axis": "#4B5563",
    "text": "#111827",
    "subtle": "#6B7280",
    "panel_fill": "#FFFFFF",
    "panel_border": "#D1D5DB",
    "accent_fill": "#F8FAFC",
}


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_matrix(rows: List[Dict[str, str]], columns: Sequence[str]) -> np.ndarray:
    if not rows:
        return np.zeros((0, len(columns)), dtype=np.float32)
    missing = [column for column in columns if column not in rows[0]]
    if missing:
        raise KeyError(f"{missing} not found in CSV columns.")
    matrix = []
    for row in rows:
        matrix.append([float(row[column]) for column in columns])
    return np.asarray(matrix, dtype=np.float32)


def _compute_norm(values: np.ndarray, start: int, end: int) -> np.ndarray:
    return np.linalg.norm(values[:, start:end], axis=1)


def _compute_rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((prediction - target) ** 2)))


def _compute_pearson(prediction: np.ndarray, target: np.ndarray) -> float:
    if prediction.std() < 1e-12 or target.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(prediction, target)[0, 1])


def _fmt(value: float, decimals: int = 3) -> str:
    if value is None or math.isnan(value):
        return ""
    return f"{float(value):.{decimals}f}"


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    preferred = []
    if bold:
        preferred.extend(["arialbd.ttf", "Arial Bold.ttf", "DejaVuSans-Bold.ttf"])
    else:
        preferred.extend(["arial.ttf", "Arial.ttf", "DejaVuSans.ttf"])
    for name in preferred:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_text(draw: ImageDraw.ImageDraw, xy: Tuple[int, int], text: str, font, fill: str) -> None:
    draw.text(xy, text, font=font, fill=fill)


def _draw_panel(
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    title: str,
    times: np.ndarray,
    series: Dict[str, np.ndarray],
    y_label: str,
    current_time: float,
    fonts: Dict[str, ImageFont.ImageFont],
) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=10, fill=COLORS["panel_fill"], outline=COLORS["panel_border"], width=2)
    _draw_text(draw, (x0 + 18, y0 + 14), title, fonts["panel_title"], COLORS["text"])
    _draw_text(draw, (x1 - 140, y0 + 16), f"t = {_fmt(current_time, 2)} s", fonts["small"], COLORS["subtle"])

    left = x0 + 78
    right = x1 - 24
    top = y0 + 56
    bottom = y1 - 50
    width = max(right - left, 1)
    height = max(bottom - top, 1)

    all_values = np.concatenate([values for values in series.values() if values.size > 0], axis=0)
    if all_values.size == 0:
        return
    y_min = float(np.min(all_values))
    y_max = float(np.max(all_values))
    if abs(y_max - y_min) < 1e-6:
        y_min -= 1.0
        y_max += 1.0
    padding = 0.08 * (y_max - y_min)
    y_min -= padding
    y_max += padding

    time_min = float(times[0]) if len(times) else 0.0
    time_max = float(times[-1]) if len(times) else max(current_time, 1.0)
    if time_max - time_min < 1e-6:
        time_max = time_min + 1.0

    for fraction in np.linspace(0.0, 1.0, 5):
        y = top + int((1.0 - fraction) * height)
        draw.line([(left, y), (right, y)], fill=COLORS["grid"], width=1)
        tick_value = y_min + fraction * (y_max - y_min)
        _draw_text(draw, (x0 + 18, y - 8), _fmt(tick_value, 2), fonts["small"], COLORS["subtle"])

    for fraction in np.linspace(0.0, 1.0, 6):
        x = left + int(fraction * width)
        draw.line([(x, top), (x, bottom)], fill=COLORS["grid"], width=1)
        tick_time = time_min + fraction * (time_max - time_min)
        _draw_text(draw, (x - 12, bottom + 12), _fmt(tick_time, 1), fonts["small"], COLORS["subtle"])

    draw.line([(left, top), (left, bottom)], fill=COLORS["axis"], width=2)
    draw.line([(left, bottom), (right, bottom)], fill=COLORS["axis"], width=2)
    _draw_text(draw, (x0 + 18, y0 + 34), y_label, fonts["small"], COLORS["subtle"])

    def to_xy(local_times: np.ndarray, local_values: np.ndarray) -> List[Tuple[int, int]]:
        points: List[Tuple[int, int]] = []
        for time_value, y_value in zip(local_times, local_values):
            x = left + int((float(time_value) - time_min) / (time_max - time_min) * width)
            y = top + int((y_max - float(y_value)) / (y_max - y_min) * height)
            points.append((x, y))
        return points

    for key in ("reference", "input", "prediction"):
        values = series.get(key)
        if values is None or len(values) < 2:
            continue
        points = to_xy(times, values)
        draw.line(points, fill=COLORS[key], width=4 if key == "prediction" else 3)

    cursor_x = left + int((current_time - time_min) / (time_max - time_min) * width)
    draw.line([(cursor_x, top), (cursor_x, bottom)], fill="#9CA3AF", width=2)

    legend_y = y1 - 28
    legend_x = x0 + 20
    legend_items = [("input", "Nonrigid input"), ("prediction", "Compensated output"), ("reference", "Rigid reference")]
    for key, label in legend_items:
        draw.line([(legend_x, legend_y + 8), (legend_x + 28, legend_y + 8)], fill=COLORS[key], width=4)
        _draw_text(draw, (legend_x + 36, legend_y), label, fonts["small"], COLORS["subtle"])
        legend_x += 190


def _draw_metrics_box(
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    summary_rows: List[Dict[str, str]],
    current_index: int,
    latencies: np.ndarray,
    fonts: Dict[str, ImageFont.ImageFont],
) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=10, fill=COLORS["accent_fill"], outline=COLORS["panel_border"], width=2)
    _draw_text(draw, (x0 + 18, y0 + 14), "Replay Summary", fonts["panel_title"], COLORS["text"])

    if summary_rows:
        row = summary_rows[0]
        lines = [
            f"Input RMSE: {row.get('Input RMSE', '')}",
            f"Compensated RMSE: {row.get('Compensated RMSE', '')}",
            f"RMSE Reduction: {row.get('RMSE Reduction (%)', '')}%",
            f"Pearson: {row.get('Pearson', '')}",
            f"HF Improve.: {row.get('HF Improve.', '')}",
        ]
    else:
        lines = []

    if latencies.size:
        current_latency = float(latencies[min(current_index, len(latencies) - 1)])
        lines.extend(
            [
                f"Current latency: {_fmt(current_latency, 3)} ms",
                f"Mean latency: {_fmt(float(np.mean(latencies)), 3)} ms",
                f"P95 latency: {_fmt(float(np.percentile(latencies, 95)), 3)} ms",
            ]
        )

    y = y0 + 54
    for line in lines:
        _draw_text(draw, (x0 + 18, y), line, fonts["body"], COLORS["text"])
        y += 28


def _draw_value_table(
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    input_step: np.ndarray,
    prediction_step: np.ndarray,
    reference_step: np.ndarray,
    fonts: Dict[str, ImageFont.ImageFont],
) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=10, fill=COLORS["panel_fill"], outline=COLORS["panel_border"], width=2)
    _draw_text(draw, (x0 + 18, y0 + 14), "Current Sample Values", fonts["panel_title"], COLORS["text"])

    header_y = y0 + 52
    channel_x = x0 + 18
    input_x = x0 + 170
    pred_x = x0 + 320
    ref_x = x0 + 470

    _draw_text(draw, (channel_x, header_y), "Channel", fonts["body"], COLORS["subtle"])
    _draw_text(draw, (input_x, header_y), "Input", fonts["body"], COLORS["input"])
    _draw_text(draw, (pred_x, header_y), "Compensated", fonts["body"], COLORS["prediction"])
    _draw_text(draw, (ref_x, header_y), "Reference", fonts["body"], COLORS["reference"])

    row_y = header_y + 32
    for index, channel in enumerate(CHANNELS):
        draw.line([(x0 + 18, row_y - 8), (x1 - 18, row_y - 8)], fill=COLORS["grid"], width=1)
        _draw_text(draw, (channel_x, row_y), channel, fonts["small"], COLORS["text"])
        _draw_text(draw, (input_x, row_y), _fmt(float(input_step[index]), 4), fonts["small"], COLORS["text"])
        _draw_text(draw, (pred_x, row_y), _fmt(float(prediction_step[index]), 4), fonts["small"], COLORS["text"])
        _draw_text(draw, (ref_x, row_y), _fmt(float(reference_step[index]), 4), fonts["small"], COLORS["text"])
        row_y += 34


def _draw_status_box(
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    summary_rows: List[Dict[str, str]],
    current_index: int,
    current_time: float,
    times: np.ndarray,
    latencies: np.ndarray,
    fonts: Dict[str, ImageFont.ImageFont],
) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=10, fill=COLORS["accent_fill"], outline=COLORS["panel_border"], width=2)
    _draw_text(draw, (x0 + 18, y0 + 14), "Runtime Status and Key Metrics", fonts["panel_title"], COLORS["text"])

    sampling_hz = float("nan")
    if len(times) >= 2:
        step = float(np.median(np.diff(times)))
        if step > 0:
            sampling_hz = 1.0 / step

    if summary_rows:
        row = summary_rows[0]
        lines = [
            f"Current time: {_fmt(current_time, 3)} s",
            f"Current sample index: {current_index + 1}/{len(times)}",
            f"Sampling frequency: {_fmt(sampling_hz, 1)} Hz",
            "",
            f"Input RMSE: {row.get('Input RMSE', '')}",
            f"Compensated RMSE: {row.get('Compensated RMSE', '')}",
            f"RMSE Reduction: {row.get('RMSE Reduction (%)', '')}%",
            f"Pearson: {row.get('Pearson', '')}",
            f"PSD Distance: {row.get('PSD Distance', '')}",
            f"HF Improve.: {row.get('HF Improve.', '')}",
        ]
    else:
        lines = [
            f"Current time: {_fmt(current_time, 3)} s",
            f"Current sample index: {current_index + 1}/{len(times)}",
            f"Sampling frequency: {_fmt(sampling_hz, 1)} Hz",
        ]

    if latencies.size:
        current_latency = float(latencies[min(current_index, len(latencies) - 1)])
        lines.extend(
            [
                "",
                f"Current latency: {_fmt(current_latency, 3)} ms",
                f"Mean latency: {_fmt(float(np.mean(latencies)), 3)} ms",
                f"P95 latency: {_fmt(float(np.percentile(latencies, 95)), 3)} ms",
            ]
        )

    y = y0 + 52
    for line in lines:
        if line:
            _draw_text(draw, (x0 + 18, y), line, fonts["body"], COLORS["text"])
        y += 28


def _build_frames_stacked(
    times: np.ndarray,
    input_values: np.ndarray,
    prediction_values: np.ndarray,
    reference_values: np.ndarray,
    latencies: np.ndarray,
    summary_rows: List[Dict[str, str]],
    history_seconds: float,
    frame_stride: int,
    width: int,
    height: int,
) -> List[Image.Image]:
    fonts = {
        "title": _load_font(28, bold=True),
        "subtitle": _load_font(15, bold=False),
        "panel_title": _load_font(19, bold=True),
        "body": _load_font(17, bold=False),
        "small": _load_font(14, bold=False),
    }

    frames: List[Image.Image] = []
    acc_input = _compute_norm(input_values, 0, 3)
    acc_prediction = _compute_norm(prediction_values, 0, 3)
    acc_reference = _compute_norm(reference_values, 0, 3)
    gyr_input = _compute_norm(input_values, 3, 6)
    gyr_prediction = _compute_norm(prediction_values, 3, 6)
    gyr_reference = _compute_norm(reference_values, 3, 6)

    frame_indices = list(range(0, len(times), max(int(frame_stride), 1)))
    if not frame_indices or frame_indices[-1] != len(times) - 1:
        frame_indices.append(len(times) - 1)

    for frame_number, index in enumerate(frame_indices, start=1):
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)

        _draw_text(draw, (40, 28), "Real-Time IMU Compensation Replay", fonts["title"], COLORS["text"])
        _draw_text(
            draw,
            (40, 66),
            "Streaming nonrigid input through TCN-causal and comparing the compensated output against the rigid reference.",
            fonts["subtitle"],
            COLORS["subtle"],
        )
        _draw_text(
            draw,
            (width - 220, 34),
            f"Frame {frame_number}/{len(frame_indices)}",
            fonts["subtitle"],
            COLORS["subtle"],
        )

        current_time = float(times[index])
        history_start = max(current_time - history_seconds, float(times[0]))
        history_mask = (times >= history_start) & (times <= current_time)
        local_times = times[history_mask]

        acc_series = {
            "input": acc_input[history_mask],
            "prediction": acc_prediction[history_mask],
            "reference": acc_reference[history_mask],
        }
        gyr_series = {
            "input": gyr_input[history_mask],
            "prediction": gyr_prediction[history_mask],
            "reference": gyr_reference[history_mask],
        }

        _draw_panel(
            draw,
            box=(36, 108, width - 36, 440),
            title="Acceleration magnitude",
            times=local_times,
            series=acc_series,
            y_label="||acc|| (m/s^2)",
            current_time=current_time,
            fonts=fonts,
        )
        _draw_panel(
            draw,
            box=(36, 468, width - 36, 800),
            title="Gyroscope magnitude",
            times=local_times,
            series=gyr_series,
            y_label="||gyr|| (rad/s)",
            current_time=current_time,
            fonts=fonts,
        )
        _draw_metrics_box(
            draw,
            box=(width - 350, 118, width - 56, 320),
            summary_rows=summary_rows,
            current_index=index,
            latencies=latencies,
            fonts=fonts,
        )

        frames.append(image)
    return frames


def _build_frames_dashboard(
    times: np.ndarray,
    input_values: np.ndarray,
    prediction_values: np.ndarray,
    reference_values: np.ndarray,
    latencies: np.ndarray,
    summary_rows: List[Dict[str, str]],
    history_seconds: float,
    frame_stride: int,
    width: int,
    height: int,
) -> List[Image.Image]:
    fonts = {
        "title": _load_font(28, bold=True),
        "subtitle": _load_font(15, bold=False),
        "panel_title": _load_font(19, bold=True),
        "body": _load_font(17, bold=False),
        "small": _load_font(14, bold=False),
    }

    frames: List[Image.Image] = []
    acc_input = _compute_norm(input_values, 0, 3)
    acc_prediction = _compute_norm(prediction_values, 0, 3)
    acc_reference = _compute_norm(reference_values, 0, 3)
    gyr_input = _compute_norm(input_values, 3, 6)
    gyr_prediction = _compute_norm(prediction_values, 3, 6)
    gyr_reference = _compute_norm(reference_values, 3, 6)

    frame_indices = list(range(0, len(times), max(int(frame_stride), 1)))
    if not frame_indices or frame_indices[-1] != len(times) - 1:
        frame_indices.append(len(times) - 1)

    left_col = 860
    right_col_x0 = 920
    right_col_x1 = width - 36

    for frame_number, index in enumerate(frame_indices, start=1):
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)

        _draw_text(draw, (40, 28), "Online IMU Compensation Dashboard", fonts["title"], COLORS["text"])
        _draw_text(
            draw,
            (40, 66),
            "Simulated real-time stream: nonrigid IMU input, causal model inference, and rigid-reference comparison.",
            fonts["subtitle"],
            COLORS["subtle"],
        )
        _draw_text(draw, (width - 220, 34), f"Frame {frame_number}/{len(frame_indices)}", fonts["subtitle"], COLORS["subtle"])

        current_time = float(times[index])
        history_start = max(current_time - history_seconds, float(times[0]))
        history_mask = (times >= history_start) & (times <= current_time)
        local_times = times[history_mask]

        acc_series = {
            "input": acc_input[history_mask],
            "prediction": acc_prediction[history_mask],
            "reference": acc_reference[history_mask],
        }
        gyr_series = {
            "input": gyr_input[history_mask],
            "prediction": gyr_prediction[history_mask],
            "reference": gyr_reference[history_mask],
        }

        _draw_panel(
            draw,
            box=(36, 108, left_col, 430),
            title="Acceleration magnitude",
            times=local_times,
            series=acc_series,
            y_label="||acc|| (m/s^2)",
            current_time=current_time,
            fonts=fonts,
        )
        _draw_panel(
            draw,
            box=(36, 458, left_col, 780),
            title="Gyroscope magnitude",
            times=local_times,
            series=gyr_series,
            y_label="||gyr|| (rad/s)",
            current_time=current_time,
            fonts=fonts,
        )
        _draw_value_table(
            draw,
            box=(right_col_x0, 108, right_col_x1, 430),
            input_step=input_values[index],
            prediction_step=prediction_values[index],
            reference_step=reference_values[index],
            fonts=fonts,
        )
        _draw_status_box(
            draw,
            box=(right_col_x0, 458, right_col_x1, 780),
            summary_rows=summary_rows,
            current_index=index,
            current_time=current_time,
            times=times,
            latencies=latencies,
            fonts=fonts,
        )

        frames.append(image)
    return frames


def _maybe_generate_predictions(
    checkpoint: Path | None,
    input_values: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    if checkpoint is None:
        raise ValueError("Either --prediction-csv or --checkpoint must be provided.")
    require_torch()
    compensator = StreamingCompensator.from_checkpoint(checkpoint, device_name="cpu")
    predictions = []
    latencies = []
    for sample in input_values:
        result = compensator.push(sample)
        predictions.append(result["prediction"])
        latencies.append(result["latency_ms"])
    return np.asarray(predictions, dtype=np.float32), np.asarray(latencies, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a real-time replay GIF for IMU compensation results.")
    parser.add_argument("--input-csv", type=Path, required=True, help="Replay-ready nonrigid 6-channel CSV.")
    parser.add_argument("--reference-csv", type=Path, required=True, help="Replay-ready rigid 6-channel CSV.")
    parser.add_argument("--prediction-csv", type=Path, default=None, help="Prediction CSV from main_realtime_infer.py.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional causal checkpoint to generate predictions on the fly.")
    parser.add_argument("--summary-csv", type=Path, default=None, help="Optional PPT summary CSV to display in the animation.")
    parser.add_argument("--output-gif", type=Path, default=Path("outputs/replay_demo/replay_animation.gif"))
    parser.add_argument("--history-seconds", type=float, default=4.0, help="Trailing history window shown in each panel.")
    parser.add_argument("--frame-stride", type=int, default=20, help="Use every Nth sample as one animation frame.")
    parser.add_argument("--fps", type=int, default=12, help="GIF playback frames per second.")
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--height", type=int, default=860)
    parser.add_argument(
        "--layout",
        type=str,
        choices=["dashboard-2x2", "stacked"],
        default="dashboard-2x2",
        help="Animation layout style.",
    )
    args = parser.parse_args()

    input_rows = _read_csv_rows(args.input_csv.resolve())
    reference_rows = _read_csv_rows(args.reference_csv.resolve())
    summary_rows = _read_csv_rows(args.summary_csv.resolve()) if args.summary_csv else []

    input_values = _load_matrix(input_rows, CHANNELS)
    reference_values = _load_matrix(reference_rows, CHANNELS)
    times = _load_matrix(input_rows, ["time_s"]).reshape(-1)

    if args.prediction_csv is not None:
        prediction_rows = _read_csv_rows(args.prediction_csv.resolve())
        prediction_values = _load_matrix(prediction_rows, PREDICTION_COLUMNS)
        latencies = _load_matrix(prediction_rows, ["latency_ms"]).reshape(-1)
    else:
        prediction_values, latencies = _maybe_generate_predictions(args.checkpoint.resolve() if args.checkpoint else None, input_values)

    length = min(len(times), len(input_values), len(reference_values), len(prediction_values), len(latencies))
    if length < 4:
        raise ValueError("Need at least 4 aligned samples to create an animation.")

    times = times[:length]
    input_values = input_values[:length]
    reference_values = reference_values[:length]
    prediction_values = prediction_values[:length]
    latencies = latencies[:length]

    if not summary_rows:
        rmse_input = _compute_rmse(input_values, reference_values)
        rmse_prediction = _compute_rmse(prediction_values, reference_values)
        pearson = _compute_pearson(prediction_values[:, 0], reference_values[:, 0])
        summary_rows = [
            {
                "Input RMSE": _fmt(rmse_input, 4),
                "Compensated RMSE": _fmt(rmse_prediction, 4),
                "RMSE Reduction (%)": _fmt(100.0 * (rmse_input - rmse_prediction) / max(rmse_input, 1e-8), 2),
                "Pearson": _fmt(pearson, 4),
                "HF Improve.": "",
            }
        ]

    if args.layout == "stacked":
        frames = _build_frames_stacked(
            times=times,
            input_values=input_values,
            prediction_values=prediction_values,
            reference_values=reference_values,
            latencies=latencies,
            summary_rows=summary_rows,
            history_seconds=float(args.history_seconds),
            frame_stride=int(args.frame_stride),
            width=int(args.width),
            height=int(args.height),
        )
    else:
        frames = _build_frames_dashboard(
            times=times,
            input_values=input_values,
            prediction_values=prediction_values,
            reference_values=reference_values,
            latencies=latencies,
            summary_rows=summary_rows,
            history_seconds=float(args.history_seconds),
            frame_stride=int(args.frame_stride),
            width=int(args.width),
            height=int(args.height),
        )

    output_path = args.output_gif.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = int(round(1000.0 / max(int(args.fps), 1)))
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        optimize=False,
        duration=duration_ms,
        loop=0,
        disposal=2,
    )
    print(f"Saved replay animation GIF to {output_path}")


if __name__ == "__main__":
    main()
