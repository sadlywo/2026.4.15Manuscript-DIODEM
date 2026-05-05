from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


CHANNELS = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]
COLORS = {
    "bg": "#FFFFFF",
    "text": "#111827",
    "muted": "#6B7280",
    "line": "#D1D5DB",
    "grid": "#E5E7EB",
    "panel": "#FFFFFF",
    "panel_border": "#D7DCE3",
    "blue": "#4C78A8",
    "orange": "#E67E22",
    "green": "#54A24B",
    "red": "#E45756",
    "slate": "#374151",
    "chip_fill": "#F8FAFC",
}


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    preferred = ["DejaVuSans-Bold.ttf", "arialbd.ttf"] if bold else ["DejaVuSans.ttf", "arial.ttf"]
    for name in preferred:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_text(draw: ImageDraw.ImageDraw, xy: Tuple[int, int], text: str, font, fill: str) -> None:
    draw.text(xy, text, font=font, fill=fill)


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _extract_sampling_frequency(csv_path: Path) -> float:
    with csv_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for _ in range(10):
            line = handle.readline()
            if not line:
                break
            if "sampling frequency" in line.lower():
                return float(line.split(":")[-1].strip())
    raise ValueError(f"Could not parse sampling frequency from {csv_path}")


def _read_raw_csv(csv_path: Path) -> Tuple[List[str], List[List[float]]]:
    with csv_path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        filtered_lines = [line for line in handle if not line.lstrip().startswith("#")]
    reader = csv.reader(filtered_lines)
    try:
        header = [item.strip() for item in next(reader)]
    except StopIteration as exc:
        raise ValueError(f"{csv_path} does not contain tabular data.") from exc
    values: List[List[float]] = []
    for row in reader:
        if row:
            values.append([float(value) for value in row])
    return header, values


def _build_segment_matrix(csv_path: Path, segment_id: str) -> Tuple[float, np.ndarray]:
    sampling_frequency = _extract_sampling_frequency(csv_path)
    header, values = _read_raw_csv(csv_path)
    header_to_index = {name: idx for idx, name in enumerate(header)}
    required = [f"{segment_id}_{channel}" for channel in CHANNELS]
    missing = [name for name in required if name not in header_to_index]
    if missing:
        raise KeyError(f"Missing columns in {csv_path}: {missing}")
    matrix = []
    for row in values:
        matrix.append([row[header_to_index[f"{segment_id}_{channel}"]] for channel in CHANNELS])
    return sampling_frequency, np.asarray(matrix, dtype=np.float32)


def _summarize_dataset(metadata_rows: List[Dict[str, str]], pair_rows: List[Dict[str, str]]) -> Dict[str, object]:
    recording_sets = Counter()
    experiments = defaultdict(set)
    motions = defaultdict(set)
    file_type_counts = Counter()
    sampling_frequency = defaultdict(set)
    for row in metadata_rows:
        kc_type = row["kc_type"]
        experiments[kc_type].add(row["experiment_id"])
        motions[kc_type].add(row["motion_name"])
        file_type_counts[row["file_type"]] += 1
        sampling_frequency[row["file_type"]].add(float(row["sampling_frequency"]))
        if row["file_type"] == "imu_nonrigid":
            recording_sets[kc_type] += 1

    split_counts = Counter(row["split"] for row in pair_rows)
    segment_ids = sorted({row["segment_id"] for row in pair_rows})
    representative_motions = [
        "canonical",
        "pause1",
        "slow1",
        "fast",
        "fast_slow",
        "shaking",
        "rotation",
        "gait_slow",
        "gait_fast",
        "explosiv",
    ]

    return {
        "recording_sets": recording_sets,
        "experiments": {key: len(value) for key, value in experiments.items()},
        "motions": {key: len(value) for key, value in motions.items()},
        "file_type_counts": file_type_counts,
        "sampling_frequency": {key: sorted(value) for key, value in sampling_frequency.items()},
        "total_recording_sets": int(sum(recording_sets.values())),
        "paired_segments": len(pair_rows),
        "split_counts": split_counts,
        "segment_ids": segment_ids,
        "representative_motions": representative_motions,
    }


def _format_frequency(values: Sequence[float]) -> str:
    unique = sorted({float(value) for value in values})
    if not unique:
        return ""
    if len(unique) == 1:
        return f"{unique[0]:.0f} Hz"
    return f"{unique[0]:.0f}-{unique[-1]:.0f} Hz"


def _rounded_panel(draw: ImageDraw.ImageDraw, box: Tuple[int, int, int, int]) -> None:
    draw.rounded_rectangle(box, radius=18, fill=COLORS["panel"], outline=COLORS["panel_border"], width=2)


def _draw_chip(draw: ImageDraw.ImageDraw, xy: Tuple[int, int], label: str, font) -> int:
    x, y = xy
    text_box = draw.textbbox((0, 0), label, font=font)
    width = text_box[2] - text_box[0] + 26
    height = text_box[3] - text_box[1] + 16
    draw.rounded_rectangle((x, y, x + width, y + height), radius=14, fill=COLORS["chip_fill"], outline=COLORS["line"])
    _draw_text(draw, (x + 13, y + 8), label, font, COLORS["slate"])
    return width


def _make_overview_figure(summary: Dict[str, object], output_path: Path) -> None:
    image = Image.new("RGB", (1800, 1120), COLORS["bg"])
    draw = ImageDraw.Draw(image)
    fonts = {
        "title": _load_font(42, bold=True),
        "subtitle": _load_font(20),
        "panel_title": _load_font(24, bold=True),
        "metric": _load_font(34, bold=True),
        "body": _load_font(20),
        "small": _load_font(17),
        "chip": _load_font(18),
    }

    _draw_text(draw, (70, 52), "DIODEM Dataset Used in This Study", fonts["title"], COLORS["text"])
    subtitle = (
        "Synchronized rigid IMU, non-rigid IMU, and optical motion capture recordings; "
        "used here to build segment-level supervised compensation pairs."
    )
    _draw_text(draw, (72, 108), subtitle, fonts["subtitle"], COLORS["muted"])

    cards = [
        (70, 170, 560, 650),
        (620, 170, 1110, 650),
        (1170, 170, 1730, 650),
        (70, 710, 1730, 1030),
    ]
    for box in cards:
        _rounded_panel(draw, box)

    # Coverage card.
    x0, y0, x1, y1 = cards[0]
    _draw_text(draw, (x0 + 28, y0 + 22), "Coverage", fonts["panel_title"], COLORS["text"])
    total_recording_sets = int(summary["total_recording_sets"])
    _draw_text(draw, (x0 + 28, y0 + 78), f"{total_recording_sets}", fonts["metric"], COLORS["blue"])
    _draw_text(draw, (x0 + 140, y0 + 90), "synchronized recording sets", fonts["body"], COLORS["text"])

    recording_sets = summary["recording_sets"]
    experiments = summary["experiments"]
    motions = summary["motions"]
    max_count = max(recording_sets.values())
    bar_left = x0 + 28
    bar_right = x1 - 40
    labels = [("arm", "Arm chain", COLORS["blue"]), ("gait", "Gait chain", COLORS["green"])]
    for idx, (key, label, color) in enumerate(labels):
        y = y0 + 170 + idx * 118
        _draw_text(draw, (bar_left, y), label, fonts["body"], COLORS["text"])
        draw.rounded_rectangle((bar_left, y + 38, bar_right, y + 62), radius=12, fill="#EEF2F7", outline=None)
        width = int((recording_sets[key] / max_count) * (bar_right - bar_left))
        draw.rounded_rectangle((bar_left, y + 38, bar_left + width, y + 62), radius=12, fill=color, outline=None)
        detail = f"{recording_sets[key]} recordings | {experiments[key]} experiments | {motions[key]} motion labels"
        _draw_text(draw, (bar_left, y + 74), detail, fonts["small"], COLORS["muted"])

    # Modalities card.
    x0, y0, x1, y1 = cards[1]
    _draw_text(draw, (x0 + 28, y0 + 22), "Measurement modalities", fonts["panel_title"], COLORS["text"])
    modalities = [
        ("Non-rigid IMU", "88 files", _format_frequency(summary["sampling_frequency"]["imu_nonrigid"]), COLORS["blue"]),
        ("Rigid IMU", "88 files", _format_frequency(summary["sampling_frequency"]["imu_rigid"]), COLORS["orange"]),
        ("OMC reference", "88 files", _format_frequency(summary["sampling_frequency"]["omc"]), COLORS["slate"]),
    ]
    for idx, (name, count_text, freq_text, color) in enumerate(modalities):
        top = y0 + 90 + idx * 118
        draw.rounded_rectangle((x0 + 28, top, x1 - 28, top + 88), radius=14, fill="#FBFCFE", outline=COLORS["line"])
        draw.rounded_rectangle((x0 + 44, top + 18, x0 + 70, top + 44), radius=8, fill=color, outline=None)
        _draw_text(draw, (x0 + 88, top + 16), name, fonts["body"], COLORS["text"])
        _draw_text(draw, (x0 + 88, top + 48), count_text, fonts["small"], COLORS["muted"])
        _draw_text(draw, (x1 - 118, top + 30), freq_text, fonts["body"], COLORS["slate"])

    # Usage card.
    x0, y0, x1, y1 = cards[2]
    _draw_text(draw, (x0 + 28, y0 + 22), "Usage in this study", fonts["panel_title"], COLORS["text"])
    _draw_text(draw, (x0 + 28, y0 + 86), "Supervised mapping", fonts["body"], COLORS["muted"])
    _draw_text(draw, (x0 + 28, y0 + 118), "X_nr  ->  X_r", fonts["metric"], COLORS["red"])
    _draw_text(draw, (x0 + 28, y0 + 204), "Segment-level pairing", fonts["body"], COLORS["muted"])
    _draw_text(draw, (x0 + 28, y0 + 238), f"{summary['paired_segments']} paired segment samples", fonts["metric"], COLORS["blue"])
    _draw_text(
        draw,
        (x0 + 28, y0 + 318),
        "Segment IDs: " + ", ".join(summary["segment_ids"]),
        fonts["body"],
        COLORS["text"],
    )
    split_counts = summary["split_counts"]
    split_text = f"Split counts: train {split_counts['train']} | val {split_counts['val']} | test {split_counts['test']}"
    _draw_text(draw, (x0 + 28, y0 + 362), split_text, fonts["small"], COLORS["muted"])

    # Motion patterns card.
    x0, y0, x1, y1 = cards[3]
    _draw_text(draw, (x0 + 28, y0 + 22), "Representative motion patterns", fonts["panel_title"], COLORS["text"])
    _draw_text(
        draw,
        (x0 + 28, y0 + 64),
        "The recordings span quasi-static, transition, and highly dynamic movements across arm and gait contexts.",
        fonts["body"],
        COLORS["muted"],
    )
    chip_x = x0 + 28
    chip_y = y0 + 120
    for label in summary["representative_motions"]:
        width = _draw_chip(draw, (chip_x, chip_y), label, fonts["chip"])
        chip_x += width + 14
        if chip_x > x1 - 220:
            chip_x = x0 + 28
            chip_y += 58

    footnote = (
        "Dataset summary derived from local metadata tables in this project; IMU streams are sampled at 40 Hz, "
        "while OMC streams are sampled at 30-120 Hz depending on the subset."
    )
    _draw_text(draw, (x0 + 28, y1 - 40), footnote, fonts["small"], COLORS["muted"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _select_example_row(selected_rows: List[Dict[str, str]]) -> Dict[str, str]:
    if selected_rows:
        return selected_rows[0]
    raise ValueError("selected_examples.csv is empty; cannot build paired example figure.")


def _choose_window(nonrigid: np.ndarray, rigid: np.ndarray, window_size: int) -> Tuple[int, int]:
    acc_diff = np.abs(np.linalg.norm(nonrigid[:, :3], axis=1) - np.linalg.norm(rigid[:, :3], axis=1))
    gyr_diff = np.abs(np.linalg.norm(nonrigid[:, 3:], axis=1) - np.linalg.norm(rigid[:, 3:], axis=1))
    acc_scale = float(np.std(acc_diff) + 1e-6)
    gyr_scale = float(np.std(gyr_diff) + 1e-6)
    score = acc_diff / acc_scale + gyr_diff / gyr_scale
    center = int(np.argmax(score))
    start = max(0, center - window_size // 2)
    end = min(len(score), start + window_size)
    start = max(0, end - window_size)
    return start, end


def _draw_time_series_panel(
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    times: np.ndarray,
    series: Sequence[Tuple[str, np.ndarray, str]],
    title: str,
    y_label: str,
    fonts: Dict[str, ImageFont.ImageFont],
) -> None:
    _rounded_panel(draw, box)
    x0, y0, x1, y1 = box
    _draw_text(draw, (x0 + 24, y0 + 18), title, fonts["panel_title"], COLORS["text"])
    left = x0 + 78
    right = x1 - 24
    top = y0 + 62
    bottom = y1 - 60
    width = right - left
    height = bottom - top

    all_values = np.concatenate([values for _, values, _ in series], axis=0)
    y_min = float(np.min(all_values))
    y_max = float(np.max(all_values))
    if abs(y_max - y_min) < 1e-6:
        y_min -= 1.0
        y_max += 1.0
    padding = 0.08 * (y_max - y_min)
    y_min -= padding
    y_max += padding

    for frac in np.linspace(0.0, 1.0, 5):
        y = top + int((1.0 - frac) * height)
        draw.line([(left, y), (right, y)], fill=COLORS["grid"], width=1)
        tick_value = y_min + frac * (y_max - y_min)
        _draw_text(draw, (x0 + 16, y - 10), f"{tick_value:.2f}", fonts["small"], COLORS["muted"])

    for frac in np.linspace(0.0, 1.0, 6):
        x = left + int(frac * width)
        draw.line([(x, top), (x, bottom)], fill=COLORS["grid"], width=1)
        tick_time = float(times[0] + frac * (times[-1] - times[0]))
        _draw_text(draw, (x - 12, bottom + 12), f"{tick_time:.1f}", fonts["small"], COLORS["muted"])

    draw.line([(left, top), (left, bottom)], fill=COLORS["slate"], width=2)
    draw.line([(left, bottom), (right, bottom)], fill=COLORS["slate"], width=2)
    _draw_text(draw, (x0 + 18, y0 + 36), y_label, fonts["small"], COLORS["muted"])

    def to_points(values: np.ndarray) -> List[Tuple[int, int]]:
        points: List[Tuple[int, int]] = []
        t0 = float(times[0])
        tr = float(times[-1] - times[0]) if len(times) > 1 else 1.0
        yr = float(y_max - y_min)
        for t, v in zip(times, values):
            x = left + int((float(t) - t0) / tr * width)
            y = top + int((y_max - float(v)) / yr * height)
            points.append((x, y))
        return points

    legend_x = x0 + 26
    legend_y = y1 - 34
    for idx, (label, values, color) in enumerate(series):
        draw.line(to_points(values), fill=color, width=4 if idx == 0 else 3)
        draw.line([(legend_x, legend_y + 8), (legend_x + 30, legend_y + 8)], fill=color, width=4)
        _draw_text(draw, (legend_x + 38, legend_y), label, fonts["small"], COLORS["muted"])
        legend_x += 210


def _make_pair_example_figure(
    dataset_root: Path,
    selected_rows: List[Dict[str, str]],
    output_path: Path,
) -> None:
    example = _select_example_row(selected_rows)
    kc_type = example["kc_type"]
    experiment_id = example["experiment_id"]
    motion_folder = example["motion_folder"]
    motion_index = example["motion_index"]
    motion_name = example["motion_name"]
    segment = example["segment"].strip().lower()
    base_dir = dataset_root / kc_type / experiment_id / motion_folder
    nonrigid_path = base_dir / f"{experiment_id}_{motion_index}_imu_nonrigid.csv"
    rigid_path = base_dir / f"{experiment_id}_{motion_index}_imu_rigid.csv"

    sampling_frequency, nonrigid = _build_segment_matrix(nonrigid_path, segment)
    _, rigid = _build_segment_matrix(rigid_path, segment)
    window_size = int(round(6.0 * sampling_frequency))
    start, end = _choose_window(nonrigid, rigid, window_size=window_size)

    nonrigid = nonrigid[start:end]
    rigid = rigid[start:end]
    times = np.arange(start, end, dtype=np.float32) / float(sampling_frequency)
    acc_nonrigid = np.linalg.norm(nonrigid[:, :3], axis=1)
    acc_rigid = np.linalg.norm(rigid[:, :3], axis=1)
    gyr_nonrigid = np.linalg.norm(nonrigid[:, 3:], axis=1)
    gyr_rigid = np.linalg.norm(rigid[:, 3:], axis=1)

    image = Image.new("RGB", (1800, 1160), COLORS["bg"])
    draw = ImageDraw.Draw(image)
    fonts = {
        "title": _load_font(40, bold=True),
        "subtitle": _load_font(20),
        "panel_title": _load_font(24, bold=True),
        "body": _load_font(20),
        "small": _load_font(17),
        "metric": _load_font(26, bold=True),
    }

    _draw_text(draw, (70, 50), "Example of Paired Measurements in DIODEM", fonts["title"], COLORS["text"])
    subtitle = (
        f"{kc_type} | {experiment_id} | {motion_name} | {segment} | "
        f"{sampling_frequency:.0f} Hz IMU | selected 6 s window"
    )
    _draw_text(draw, (72, 104), subtitle, fonts["subtitle"], COLORS["muted"])

    top_panel = (70, 170, 1730, 610)
    bottom_panel = (70, 650, 1730, 1090)
    _draw_time_series_panel(
        draw,
        top_panel,
        times,
        [
            ("Non-rigid IMU", acc_nonrigid, COLORS["blue"]),
            ("Rigid IMU", acc_rigid, COLORS["orange"]),
        ],
        title="Acceleration magnitude",
        y_label="||acc||",
        fonts=fonts,
    )
    _draw_time_series_panel(
        draw,
        bottom_panel,
        times,
        [
            ("Non-rigid IMU", gyr_nonrigid, COLORS["blue"]),
            ("Rigid IMU", gyr_rigid, COLORS["orange"]),
        ],
        title="Gyroscope magnitude",
        y_label="||gyr||",
        fonts=fonts,
    )

    acc_rmse = float(np.sqrt(np.mean((acc_nonrigid - acc_rigid) ** 2)))
    gyr_rmse = float(np.sqrt(np.mean((gyr_nonrigid - gyr_rigid) ** 2)))
    info_box = (1250, 68, 1730, 152)
    draw.rounded_rectangle(info_box, radius=14, fill="#FBFCFE", outline=COLORS["line"], width=2)
    _draw_text(draw, (1270, 84), f"Acc RMSE: {acc_rmse:.3f}", fonts["body"], COLORS["text"])
    _draw_text(draw, (1270, 114), f"Gyr RMSE: {gyr_rmse:.3f}", fonts["body"], COLORS["text"])

    footer = (
        "The paired rigid/non-rigid structure provides a direct reference for supervised compensation: "
        "the same underlying motion is observed under different attachment conditions."
    )
    _draw_text(draw, (72, 1110), footer, fonts["body"], COLORS["muted"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create scientific-style dataset overview assets for the DIODEM section.")
    parser.add_argument("--metadata-csv", type=Path, default=Path("outputs/metadata_summary.csv"))
    parser.add_argument("--pair-table", type=Path, default=Path("processed_by_experiment/pair_table.csv"))
    parser.add_argument("--selected-examples-csv", type=Path, default=Path("outputs/selected_examples.csv"))
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/figures"))
    args = parser.parse_args()

    metadata_rows = _read_csv_rows(args.metadata_csv.resolve())
    pair_rows = _read_csv_rows(args.pair_table.resolve())
    selected_rows = _read_csv_rows(args.selected_examples_csv.resolve())
    summary = _summarize_dataset(metadata_rows, pair_rows)

    output_dir = args.output_dir.resolve()
    overview_path = output_dir / "diodem_dataset_overview.png"
    pair_example_path = output_dir / "diodem_pair_example.png"
    _make_overview_figure(summary, overview_path)
    _make_pair_example_figure(args.dataset_root.resolve(), selected_rows, pair_example_path)

    print(f"Wrote overview figure to {overview_path}")
    print(f"Wrote paired-example figure to {pair_example_path}")


if __name__ == "__main__":
    main()
