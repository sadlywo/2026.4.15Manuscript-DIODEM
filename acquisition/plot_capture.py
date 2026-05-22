from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from acquisition.live_plot import (
    CHANNEL_LABELS,
    CHANNEL_UNITS,
    PREDICTION_CHANNELS,
    PRED_COLOR,
    RAW_COLOR,
    apply_nature_realtime_style,
)
from acquisition.sinct485 import INPUT_CHANNELS


DEFAULT_FORMATS = ("svg", "pdf", "png")


def _read_capture_csv(csv_path: Path) -> Dict[str, np.ndarray]:
    with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = ["elapsed_s", *INPUT_CHANNELS, *PREDICTION_CHANNELS]
        missing = [field for field in required if field not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{csv_path} is missing required columns: {missing}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{csv_path} does not contain any capture rows.")
    return {
        "time_s": np.asarray([float(row["elapsed_s"]) for row in rows], dtype=np.float32),
        "raw": np.asarray([[float(row[channel]) for channel in INPUT_CHANNELS] for row in rows], dtype=np.float32),
        "pred": np.asarray([[float(row[channel]) for channel in PREDICTION_CHANNELS] for row in rows], dtype=np.float32),
    }


def create_capture_figure(
    *,
    csv_path: Path,
    output_stem: Path,
    formats: Sequence[str] = DEFAULT_FORMATS,
    max_points: int | None = None,
) -> List[Path]:
    """Create a Nature-style multi-panel figure from a live capture CSV."""
    data = _read_capture_csv(Path(csv_path))
    time_s = data["time_s"]
    raw = data["raw"]
    pred = data["pred"]
    if max_points is not None and max_points > 0 and len(time_s) > int(max_points):
        indices = np.linspace(0, len(time_s) - 1, int(max_points)).astype(int)
        time_s = time_s[indices]
        raw = raw[indices]
        pred = pred[indices]

    apply_nature_realtime_style()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.2), sharex=True)
    axes_flat = list(axes.ravel())
    for index, axis in enumerate(axes_flat):
        axis.plot(time_s, raw[:, index], color=RAW_COLOR, linewidth=0.75, label="Raw")
        axis.plot(time_s, pred[:, index], color=PRED_COLOR, linewidth=0.75, label="Compensated")
        axis.set_title(CHANNEL_LABELS[index], fontsize=7.2, fontweight="bold")
        axis.set_ylabel(CHANNEL_UNITS[index])
        axis.grid(True, color="#e9e9e9", linewidth=0.45)
    for axis in axes_flat[3:]:
        axis.set_xlabel("Time (s)")
    axes_flat[0].legend(loc="upper left", ncols=2)
    fig.suptitle("Real-time SINCT-485 artifact compensation", fontsize=8.5, fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))

    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    outputs: List[Path] = []
    for item in formats:
        suffix = str(item).lower().lstrip(".")
        output_path = output_stem.with_suffix(f".{suffix}")
        save_kwargs = {"bbox_inches": "tight"}
        if suffix in {"png", "tif", "tiff"}:
            save_kwargs["dpi"] = 600
        fig.savefig(output_path, **save_kwargs)
        outputs.append(output_path)
    plt.close(fig)
    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a Nature-style figure from a live SINCT-485 capture CSV.")
    parser.add_argument("--input-csv", type=Path, required=True, help="CSV produced by acquisition/live_compensate.py.")
    parser.add_argument("--output-stem", type=Path, default=None, help="Output path without suffix.")
    parser.add_argument(
        "--formats",
        default="svg,pdf,png",
        help="Comma-separated export formats. Example: svg,pdf,png,tiff",
    )
    parser.add_argument("--max-points", type=int, default=None, help="Downsample very long traces for plotting.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_stem = args.output_stem
    if output_stem is None:
        output_stem = args.input_csv.with_name(args.input_csv.stem + "_nature")
    formats = tuple(item.strip() for item in str(args.formats).split(",") if item.strip())
    outputs = create_capture_figure(
        csv_path=args.input_csv,
        output_stem=output_stem,
        formats=formats,
        max_points=args.max_points,
    )
    for output in outputs:
        print(output.resolve())


if __name__ == "__main__":
    main()
