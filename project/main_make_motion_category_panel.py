from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


TARGET_MOTIONS = ["canonical", "freeze1", "slow1", "fast_slow_fast", "dangle2", "shaking"]
SEGMENTS = ["seg1", "seg2", "seg3", "seg4", "seg5"]
CHANNELS = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]
COLORS = {
    "bg": "#FFFFFF",
    "text": "#111827",
    "muted": "#6B7280",
    "line": "#D1D5DB",
    "grid": "#E5E7EB",
    "panel": "#FFFFFF",
    "panel_border": "#D7DCE3",
    "acc_nonrigid": "#4C78A8",
    "acc_rigid": "#9EC1E6",
    "gyr_nonrigid": "#F58518",
    "gyr_rigid": "#FFB56B",
    "axis": "#374151",
}


@dataclass
class MotionWindow:
    motion_name: str
    kc_type: str
    experiment_id: str
    motion_folder: str
    motion_index: str
    segment_id: str
    sampling_frequency: float
    start_index: int
    end_index: int
    score: float
    time_s: np.ndarray
    nonrigid: np.ndarray
    rigid: np.ndarray
    acc_nonrigid: np.ndarray
    acc_rigid: np.ndarray
    gyr_nonrigid: np.ndarray
    gyr_rigid: np.ndarray


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
    matrix = np.asarray(
        [[row[header_to_index[f"{segment_id}_{channel}"]] for channel in CHANNELS] for row in values],
        dtype=np.float32,
    )
    return sampling_frequency, matrix


def _compute_norms(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    acc = np.linalg.norm(values[:, :3], axis=1)
    gyr = np.linalg.norm(values[:, 3:], axis=1)
    return acc, gyr


def _choose_window(nonrigid: np.ndarray, rigid: np.ndarray, sampling_frequency: float) -> Tuple[int, int]:
    acc_nr, gyr_nr = _compute_norms(nonrigid)
    acc_rg, gyr_rg = _compute_norms(rigid)
    acc_diff = np.abs(acc_nr - acc_rg)
    gyr_diff = np.abs(gyr_nr - gyr_rg)
    acc_scale = float(np.std(acc_diff) + 1e-6)
    gyr_scale = float(np.std(gyr_diff) + 1e-6)
    score = acc_diff / acc_scale + gyr_diff / gyr_scale
    center = int(np.argmax(score))
    target_window = max(32, int(round(min(len(score), 6.0 * sampling_frequency))))
    start = max(0, center - target_window // 2)
    end = min(len(score), start + target_window)
    start = max(0, end - target_window)
    return start, end


def _candidate_score(nonrigid: np.ndarray, rigid: np.ndarray) -> float:
    acc_nr, gyr_nr = _compute_norms(nonrigid)
    acc_rg, gyr_rg = _compute_norms(rigid)
    acc_rmse = float(np.sqrt(np.mean((acc_nr - acc_rg) ** 2)))
    gyr_rmse = float(np.sqrt(np.mean((gyr_nr - gyr_rg) ** 2)))
    acc_std = float(np.std(acc_rg) + 1e-6)
    gyr_std = float(np.std(gyr_rg) + 1e-6)
    return acc_rmse / acc_std + gyr_rmse / gyr_std


def _select_motion_window(dataset_root: Path, metadata_rows: List[Dict[str, str]], motion_name: str) -> MotionWindow:
    candidates = [row for row in metadata_rows if row["file_type"] == "imu_nonrigid" and row["motion_name"] == motion_name]
    if not candidates:
        raise ValueError(f"No candidate recordings found for motion '{motion_name}'.")

    best: MotionWindow | None = None
    for row in candidates:
        nonrigid_path = Path(row["path"])
        rigid_path = nonrigid_path.with_name(nonrigid_path.name.replace("imu_nonrigid", "imu_rigid"))
        for segment_id in SEGMENTS:
            try:
                sampling_frequency, nonrigid = _build_segment_matrix(nonrigid_path, segment_id)
                _, rigid = _build_segment_matrix(rigid_path, segment_id)
            except (KeyError, ValueError):
                continue
            score = _candidate_score(nonrigid, rigid)
            start, end = _choose_window(nonrigid, rigid, sampling_frequency=sampling_frequency)
            nr_slice = nonrigid[start:end]
            rg_slice = rigid[start:end]
            times = np.arange(start, end, dtype=np.float32) / float(sampling_frequency)
            acc_nr, gyr_nr = _compute_norms(nr_slice)
            acc_rg, gyr_rg = _compute_norms(rg_slice)
            current = MotionWindow(
                motion_name=motion_name,
                kc_type=row["kc_type"],
                experiment_id=row["experiment_id"],
                motion_folder=row["motion_folder"],
                motion_index=row["motion_index"],
                segment_id=segment_id,
                sampling_frequency=sampling_frequency,
                start_index=start,
                end_index=end,
                score=score,
                time_s=times,
                nonrigid=nr_slice,
                rigid=rg_slice,
                acc_nonrigid=acc_nr,
                acc_rigid=acc_rg,
                gyr_nonrigid=gyr_nr,
                gyr_rigid=gyr_rg,
            )
            if best is None or current.score > best.score:
                best = current
    if best is None:
        raise ValueError(f"Could not extract any valid segment data for motion '{motion_name}'.")
    return best


def _draw_dashed_line(draw: ImageDraw.ImageDraw, points: Sequence[Tuple[int, int]], fill: str, width: int = 3, dash: int = 8) -> None:
    if len(points) < 2:
        return
    for idx in range(len(points) - 1):
        (x0, y0), (x1, y1) = points[idx], points[idx + 1]
        dx = x1 - x0
        dy = y1 - y0
        distance = max(1.0, float(np.hypot(dx, dy)))
        steps = int(distance // dash) + 1
        for step in range(0, steps, 2):
            start_frac = min(1.0, (step * dash) / distance)
            end_frac = min(1.0, ((step + 1) * dash) / distance)
            sx = int(round(x0 + dx * start_frac))
            sy = int(round(y0 + dy * start_frac))
            ex = int(round(x0 + dx * end_frac))
            ey = int(round(y0 + dy * end_frac))
            draw.line([(sx, sy), (ex, ey)], fill=fill, width=width)


def _series_to_points(values: np.ndarray, times: np.ndarray, x_left: int, x_right: int, y_top: int, y_bottom: int, y_min: float, y_max: float) -> List[Tuple[int, int]]:
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


def _draw_subplot(
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    data: MotionWindow,
    fonts: Dict[str, ImageFont.ImageFont],
) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=16, fill=COLORS["panel"], outline=COLORS["panel_border"], width=2)

    title = data.motion_name.replace("_", " ")
    _draw_text(draw, (x0 + 22, y0 + 16), title, fonts["panel_title"], COLORS["text"])
    sub = f"{data.kc_type} | {data.experiment_id} | {data.segment_id} | {data.sampling_frequency:.0f} Hz"
    _draw_text(draw, (x0 + 22, y0 + 48), sub, fonts["small"], COLORS["muted"])

    left = x0 + 70
    right = x1 - 76
    top = y0 + 82
    bottom = y1 - 62
    width = right - left
    height = bottom - top

    acc_min = min(float(np.min(data.acc_nonrigid)), float(np.min(data.acc_rigid)))
    acc_max = max(float(np.max(data.acc_nonrigid)), float(np.max(data.acc_rigid)))
    gyr_min = min(float(np.min(data.gyr_nonrigid)), float(np.min(data.gyr_rigid)))
    gyr_max = max(float(np.max(data.gyr_nonrigid)), float(np.max(data.gyr_rigid)))
    if abs(acc_max - acc_min) < 1e-6:
        acc_min -= 1.0
        acc_max += 1.0
    if abs(gyr_max - gyr_min) < 1e-6:
        gyr_min -= 1.0
        gyr_max += 1.0
    acc_pad = 0.08 * (acc_max - acc_min)
    gyr_pad = 0.08 * (gyr_max - gyr_min)
    acc_min -= acc_pad
    acc_max += acc_pad
    gyr_min -= gyr_pad
    gyr_max += gyr_pad

    for frac in np.linspace(0.0, 1.0, 5):
        y = top + int((1.0 - frac) * height)
        draw.line([(left, y), (right, y)], fill=COLORS["grid"], width=1)
        acc_tick = acc_min + frac * (acc_max - acc_min)
        gyr_tick = gyr_min + frac * (gyr_max - gyr_min)
        _draw_text(draw, (x0 + 8, y - 9), f"{acc_tick:.2f}", fonts["tiny"], COLORS["acc_nonrigid"])
        _draw_text(draw, (right + 12, y - 9), f"{gyr_tick:.2f}", fonts["tiny"], COLORS["gyr_nonrigid"])
    for frac in np.linspace(0.0, 1.0, 6):
        x = left + int(frac * width)
        draw.line([(x, top), (x, bottom)], fill=COLORS["grid"], width=1)
        tick_time = float(data.time_s[0] + frac * (data.time_s[-1] - data.time_s[0]))
        _draw_text(draw, (x - 12, bottom + 10), f"{tick_time:.1f}", fonts["tiny"], COLORS["muted"])

    draw.line([(left, top), (left, bottom)], fill=COLORS["axis"], width=2)
    draw.line([(right, top), (right, bottom)], fill=COLORS["axis"], width=2)
    draw.line([(left, bottom), (right, bottom)], fill=COLORS["axis"], width=2)
    _draw_text(draw, (x0 + 12, y0 + 78), "||acc||", fonts["tiny"], COLORS["acc_nonrigid"])
    _draw_text(draw, (right + 10, y0 + 78), "||gyr||", fonts["tiny"], COLORS["gyr_nonrigid"])

    acc_nr_points = _series_to_points(data.acc_nonrigid, data.time_s, left, right, top, bottom, acc_min, acc_max)
    acc_rg_points = _series_to_points(data.acc_rigid, data.time_s, left, right, top, bottom, acc_min, acc_max)
    gyr_nr_points = _series_to_points(data.gyr_nonrigid, data.time_s, left, right, top, bottom, gyr_min, gyr_max)
    gyr_rg_points = _series_to_points(data.gyr_rigid, data.time_s, left, right, top, bottom, gyr_min, gyr_max)

    draw.line(acc_nr_points, fill=COLORS["acc_nonrigid"], width=3)
    _draw_dashed_line(draw, acc_rg_points, fill=COLORS["acc_rigid"], width=3, dash=10)
    draw.line(gyr_nr_points, fill=COLORS["gyr_nonrigid"], width=3)
    _draw_dashed_line(draw, gyr_rg_points, fill=COLORS["gyr_rigid"], width=3, dash=10)

    info = f"window: {data.start_index / data.sampling_frequency:.1f}-{data.end_index / data.sampling_frequency:.1f} s"
    _draw_text(draw, (x0 + 22, y1 - 36), info, fonts["tiny"], COLORS["muted"])


def _make_panel_figure(windows: Sequence[MotionWindow], output_path: Path) -> None:
    image = Image.new("RGB", (2400, 1650), COLORS["bg"])
    draw = ImageDraw.Draw(image)
    fonts = {
        "title": _load_font(46, bold=True),
        "subtitle": _load_font(22),
        "panel_title": _load_font(25, bold=True),
        "small": _load_font(16),
        "tiny": _load_font(13),
        "legend": _load_font(18),
    }

    _draw_text(draw, (70, 46), "Representative Paired Rigid/Non-Rigid IMU Patterns", fonts["title"], COLORS["text"])
    subtitle = (
        "Each panel shows acceleration magnitude on the left axis and gyroscope magnitude on the right axis; "
        "solid lines denote non-rigid signals and dashed lines denote rigid-reference signals."
    )
    _draw_text(draw, (72, 104), subtitle, fonts["subtitle"], COLORS["muted"])

    legend_y = 144
    legend_items = [
        ("Acc non-rigid", COLORS["acc_nonrigid"], False),
        ("Acc rigid", COLORS["acc_rigid"], True),
        ("Gyr non-rigid", COLORS["gyr_nonrigid"], False),
        ("Gyr rigid", COLORS["gyr_rigid"], True),
    ]
    legend_x = 72
    for label, color, dashed in legend_items:
        if dashed:
            _draw_dashed_line(draw, [(legend_x, legend_y + 10), (legend_x + 36, legend_y + 10)], fill=color, width=4, dash=8)
        else:
            draw.line([(legend_x, legend_y + 10), (legend_x + 36, legend_y + 10)], fill=color, width=4)
        _draw_text(draw, (legend_x + 48, legend_y), label, fonts["legend"], COLORS["muted"])
        legend_x += 255

    left_margin = 70
    right_margin = 70
    top_margin = 200
    bottom_margin = 60
    gap_x = 30
    gap_y = 30
    panel_width = (2400 - left_margin - right_margin - 2 * gap_x) // 3
    panel_height = (1650 - top_margin - bottom_margin - gap_y) // 2
    for idx, window in enumerate(windows):
        row = idx // 3
        col = idx % 3
        x0 = left_margin + col * (panel_width + gap_x)
        y0 = top_margin + row * (panel_height + gap_y)
        box = (x0, y0, x0 + panel_width, y0 + panel_height)
        _draw_subplot(draw, box, window, fonts)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _write_origin_exports(windows: Sequence[MotionWindow], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: List[Dict[str, str]] = []
    combined_rows: List[Dict[str, str]] = []
    for window in windows:
        rows: List[Dict[str, str]] = []
        for idx in range(len(window.time_s)):
            row = {
                "motion_name": window.motion_name,
                "kc_type": window.kc_type,
                "experiment_id": window.experiment_id,
                "motion_folder": window.motion_folder,
                "motion_index": window.motion_index,
                "segment_id": window.segment_id,
                "sampling_frequency_hz": f"{window.sampling_frequency:.1f}",
                "time_s": f"{float(window.time_s[idx]):.6f}",
                "acc_nonrigid_norm": f"{float(window.acc_nonrigid[idx]):.8f}",
                "acc_rigid_norm": f"{float(window.acc_rigid[idx]):.8f}",
                "gyr_nonrigid_norm": f"{float(window.gyr_nonrigid[idx]):.8f}",
                "gyr_rigid_norm": f"{float(window.gyr_rigid[idx]):.8f}",
            }
            for ch_idx, channel in enumerate(CHANNELS):
                row[f"nonrigid_{channel}"] = f"{float(window.nonrigid[idx, ch_idx]):.8f}"
                row[f"rigid_{channel}"] = f"{float(window.rigid[idx, ch_idx]):.8f}"
            rows.append(row)
            combined_rows.append(row)

        csv_path = output_dir / f"{window.motion_name}_origin_export.csv"
        fieldnames = list(rows[0].keys())
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        manifest_rows.append(
            {
                "motion_name": window.motion_name,
                "kc_type": window.kc_type,
                "experiment_id": window.experiment_id,
                "motion_folder": window.motion_folder,
                "motion_index": window.motion_index,
                "segment_id": window.segment_id,
                "sampling_frequency_hz": f"{window.sampling_frequency:.1f}",
                "window_start_s": f"{window.start_index / window.sampling_frequency:.3f}",
                "window_end_s": f"{window.end_index / window.sampling_frequency:.3f}",
                "selection_score": f"{window.score:.6f}",
                "csv_file": csv_path.name,
            }
        )

    if combined_rows:
        combined_path = output_dir / "all_selected_motions_origin_export.csv"
        with combined_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(combined_rows[0].keys()))
            writer.writeheader()
            writer.writerows(combined_rows)

    if manifest_rows:
        manifest_path = output_dir / "selected_motion_manifest.csv"
        with manifest_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0].keys()))
            writer.writeheader()
            writer.writerows(manifest_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a 2x3 scientific panel of representative rigid/non-rigid motion categories.")
    parser.add_argument("--metadata-csv", type=Path, default=Path("outputs/metadata_summary.csv"))
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--output-figure", type=Path, default=Path("outputs/paper_figures/figure_motion_category_panel.png"))
    parser.add_argument("--origin-output-dir", type=Path, default=Path("outputs/paper_tables/origin_motion_panel_data"))
    args = parser.parse_args()

    metadata_rows = _read_csv_rows(args.metadata_csv.resolve())
    windows = [_select_motion_window(args.dataset_root.resolve(), metadata_rows, motion) for motion in TARGET_MOTIONS]
    _make_panel_figure(windows, args.output_figure.resolve())
    _write_origin_exports(windows, args.origin_output_dir.resolve())

    print(f"Wrote figure to {args.output_figure.resolve()}")
    print(f"Wrote Origin-friendly data to {args.origin_output_dir.resolve()}")


if __name__ == "__main__":
    main()
