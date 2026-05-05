from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


CHANNELS = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]


def _load_pair_row(pair_table: Path, row_index: int) -> Dict[str, str]:
    with pair_table.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for current_index, row in enumerate(reader):
            if current_index == row_index:
                return {str(key): str(value) for key, value in row.items()}
    raise IndexError(f"Row index {row_index} is outside pair table length.")


def _extract_sampling_frequency(csv_path: Path) -> float:
    with csv_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for _ in range(10):
            line = handle.readline()
            if not line:
                break
            if "sampling frequency" in line.lower():
                value = line.split(":")[-1].strip()
                return float(value)
    raise ValueError(f"Could not parse sampling frequency from {csv_path}")


def _read_raw_csv(csv_path: Path) -> Tuple[List[str], List[List[float]]]:
    rows: List[List[float]] = []
    with csv_path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        filtered_lines = [line for line in handle if not line.lstrip().startswith("#")]
    reader = csv.reader(filtered_lines)
    try:
        header = [item.strip() for item in next(reader)]
    except StopIteration as exc:
        raise ValueError(f"{csv_path} does not contain tabular data.") from exc
    for row in reader:
        if not row:
            continue
        rows.append([float(value) for value in row])
    return header, rows


def _build_six_channel_rows(csv_path: Path, segment_id: str) -> Tuple[float, List[Dict[str, float]]]:
    sampling_frequency = _extract_sampling_frequency(csv_path)
    header, values = _read_raw_csv(csv_path)
    header_to_index = {name: idx for idx, name in enumerate(header)}
    segment = str(segment_id).strip().lower()
    required = [f"{segment}_{channel}" for channel in CHANNELS]
    missing = [name for name in required if name not in header_to_index]
    if missing:
        raise KeyError(f"Missing required columns in {csv_path}: {missing}")

    output_rows: List[Dict[str, float]] = []
    for row_index, row_values in enumerate(values):
        item: Dict[str, float] = {"time_s": float(row_index) / float(sampling_frequency)}
        for channel in CHANNELS:
            item[channel] = float(row_values[header_to_index[f"{segment}_{channel}"]])
        output_rows.append(item)
    return sampling_frequency, output_rows


def _write_six_channel_csv(output_path: Path, rows: List[Dict[str, float]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["time_s", *CHANNELS]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert raw DIODEM IMU CSVs into 6-channel replay-ready files.")
    parser.add_argument("--nonrigid-csv", type=Path, default=None, help="Path to a raw DIODEM nonrigid IMU CSV.")
    parser.add_argument("--rigid-csv", type=Path, default=None, help="Path to a raw DIODEM rigid IMU CSV.")
    parser.add_argument("--segment-id", type=str, default=None, help="Segment id like seg1, seg2, ...")
    parser.add_argument(
        "--pair-table",
        type=Path,
        default=None,
        help="Optional pair_table.csv path. When provided, nonrigid/rigid/segment can be pulled from one row.",
    )
    parser.add_argument(
        "--pair-row",
        type=int,
        default=None,
        help="Zero-based row index inside pair_table.csv to prepare for replay.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("dataset"),
        help="Dataset root used to resolve paths from pair_table.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/replay_demo"),
        help="Directory where replay-ready CSV files will be written.",
    )
    args = parser.parse_args()

    nonrigid_csv = args.nonrigid_csv
    rigid_csv = args.rigid_csv
    segment_id = args.segment_id
    base_name = "replay_sample"

    if args.pair_table is not None or args.pair_row is not None:
        if args.pair_table is None or args.pair_row is None:
            raise ValueError("--pair-table and --pair-row must be provided together.")
        pair_row = _load_pair_row(args.pair_table.resolve(), int(args.pair_row))
        dataset_root = args.dataset_root.resolve()
        nonrigid_csv = dataset_root / pair_row["nonrigid_path"]
        rigid_csv = dataset_root / pair_row["rigid_path"]
        segment_id = pair_row["segment_id"]
        base_name = f"{pair_row.get('experiment_id', '')}_{pair_row.get('motion_name', '')}_{segment_id}".strip("_")

    if nonrigid_csv is None or rigid_csv is None or segment_id is None:
        raise ValueError(
            "Provide either --pair-table/--pair-row or all of --nonrigid-csv, --rigid-csv, and --segment-id."
        )

    output_dir = args.output_dir.resolve()
    _, nonrigid_rows = _build_six_channel_rows(nonrigid_csv.resolve(), segment_id=segment_id)
    _, rigid_rows = _build_six_channel_rows(rigid_csv.resolve(), segment_id=segment_id)

    nonrigid_output = output_dir / f"{base_name}_nonrigid_6ch.csv"
    rigid_output = output_dir / f"{base_name}_rigid_6ch.csv"
    _write_six_channel_csv(nonrigid_output, nonrigid_rows)
    _write_six_channel_csv(rigid_output, rigid_rows)

    print(f"Prepared replay-ready nonrigid CSV: {nonrigid_output}")
    print(f"Prepared replay-ready rigid CSV: {rigid_output}")


if __name__ == "__main__":
    main()
