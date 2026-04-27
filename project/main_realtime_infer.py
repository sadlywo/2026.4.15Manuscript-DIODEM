from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from project.inference import StreamingCompensator
from project.utils.torch_compat import require_torch, torch


def _parse_csv_columns(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _resolve_input_columns(args, compensator: StreamingCompensator) -> List[str]:
    if args.input_columns:
        columns = _parse_csv_columns(args.input_columns)
    else:
        columns = list(getattr(args, "checkpoint_input_channels", []))
    if len(columns) != compensator.input_dim:
        raise ValueError(
            f"Expected {compensator.input_dim} input columns, got {len(columns)}: {columns}"
        )
    return columns


def _format_prediction_row(
    prediction: np.ndarray,
    latency_ms: float,
    prediction_columns: Sequence[str],
    input_row: dict | None = None,
    include_input: bool = False,
) -> dict:
    output_row = {}
    if include_input and input_row is not None:
        output_row.update(input_row)
    for key, value in zip(prediction_columns, prediction.tolist()):
        output_row[key] = float(value)
    output_row["latency_ms"] = float(latency_ms)
    return output_row


def _read_csv_rows(
    path: Path,
    input_columns: Sequence[str],
) -> Iterable[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in input_columns if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"CSV file {path} is missing required columns: {missing}")
        for row in reader:
            if not row:
                continue
            yield row


def _row_to_sample(row: dict, input_columns: Sequence[str]) -> np.ndarray:
    values = [float(row[column]) for column in input_columns]
    return np.asarray(values, dtype=np.float32)


def _run_csv_file(
    compensator: StreamingCompensator,
    input_path: Path,
    output_path: Path,
    input_columns: Sequence[str],
    prediction_columns: Sequence[str],
    include_input: bool,
) -> None:
    rows = list(_read_csv_rows(input_path, input_columns))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = []
        if include_input and rows:
            fieldnames.extend(list(rows[0].keys()))
        fieldnames.extend(prediction_columns)
        fieldnames.append("latency_ms")
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            result = compensator.push(_row_to_sample(row, input_columns))
            writer.writerow(
                _format_prediction_row(
                    prediction=result["prediction"],
                    latency_ms=result["latency_ms"],
                    prediction_columns=prediction_columns,
                    input_row=row,
                    include_input=include_input,
                )
            )


def _iter_stdin_lines() -> Iterable[str]:
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        yield line


def _parse_stdin_sample(line: str, input_columns: Sequence[str], input_format: str) -> tuple[np.ndarray, dict | None]:
    if input_format == "jsonl":
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError("JSONL input must contain one JSON object per line.")
        return _row_to_sample(row, input_columns), row
    values = [float(item.strip()) for item in line.split(",")]
    if len(values) != len(input_columns):
        raise ValueError(f"Expected {len(input_columns)} comma-separated values, got {len(values)}")
    row = {key: value for key, value in zip(input_columns, values)}
    return np.asarray(values, dtype=np.float32), row


def _run_stdin_stream(
    compensator: StreamingCompensator,
    input_columns: Sequence[str],
    prediction_columns: Sequence[str],
    input_format: str,
    output_format: str,
    include_input: bool,
) -> None:
    for line in _iter_stdin_lines():
        sample, input_row = _parse_stdin_sample(line, input_columns=input_columns, input_format=input_format)
        result = compensator.push(sample)
        output_row = _format_prediction_row(
            prediction=result["prediction"],
            latency_ms=result["latency_ms"],
            prediction_columns=prediction_columns,
            input_row=input_row,
            include_input=include_input,
        )
        if output_format == "jsonl":
            sys.stdout.write(json.dumps(output_row, ensure_ascii=True) + "\n")
        else:
            values = [str(output_row[name]) for name in prediction_columns]
            values.append(str(output_row["latency_ms"]))
            sys.stdout.write(",".join(values) + "\n")
        sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run real-time causal IMU compensation on CSV files or a live stdin stream."
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to a trained causal checkpoint.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device, e.g. cpu or cuda.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["csv-file", "stdin-stream"],
        default="stdin-stream",
        help="Replay a CSV file or consume a live sample stream from stdin.",
    )
    parser.add_argument("--input-csv", type=Path, default=None, help="Input CSV path for csv-file mode.")
    parser.add_argument("--output-csv", type=Path, default=None, help="Output CSV path for csv-file mode.")
    parser.add_argument(
        "--input-columns",
        type=str,
        default=None,
        help="Comma-separated input column names. Defaults to the training input channel order.",
    )
    parser.add_argument(
        "--prediction-columns",
        type=str,
        default="pred_acc_x,pred_acc_y,pred_acc_z,pred_gyr_x,pred_gyr_y,pred_gyr_z",
        help="Comma-separated output column names.",
    )
    parser.add_argument(
        "--stdin-format",
        type=str,
        choices=["csv", "jsonl"],
        default="csv",
        help="How each stdin sample is encoded.",
    )
    parser.add_argument(
        "--stdout-format",
        type=str,
        choices=["csv", "jsonl"],
        default="jsonl",
        help="How each prediction row is emitted in stdin-stream mode.",
    )
    parser.add_argument(
        "--include-input",
        action="store_true",
        help="Echo the input fields alongside each prediction row.",
    )
    args = parser.parse_args()

    require_torch()
    compensator = StreamingCompensator.from_checkpoint(args.checkpoint, device_name=args.device)
    checkpoint = torch.load(Path(args.checkpoint).resolve(), map_location=args.device)
    config = dict(checkpoint.get("config") or {})
    if not bool(dict(config.get("model", {})).get("causal", False)):
        raise ValueError(
            "The provided checkpoint is not configured as causal. "
            "Use a `tcn_causal` checkpoint for live deployment."
        )
    args.checkpoint_input_channels = list(config.get("input_channels", []))
    args.checkpoint_target_channels = list(config.get("target_channels", []))

    input_columns = _resolve_input_columns(args, compensator)
    prediction_columns = _parse_csv_columns(args.prediction_columns)
    if len(prediction_columns) != len(args.checkpoint_target_channels):
        raise ValueError(
            f"Expected {len(args.checkpoint_target_channels)} prediction columns, got {len(prediction_columns)}"
        )

    if args.mode == "csv-file":
        if args.input_csv is None:
            raise ValueError("--input-csv is required when --mode csv-file is used.")
        output_path = args.output_csv
        if output_path is None:
            output_path = args.input_csv.with_name(args.input_csv.stem + "_predictions.csv")
        _run_csv_file(
            compensator=compensator,
            input_path=args.input_csv.resolve(),
            output_path=output_path.resolve(),
            input_columns=input_columns,
            prediction_columns=prediction_columns,
            include_input=bool(args.include_input),
        )
        print(f"Saved predictions to {output_path.resolve()}")
        return

    _run_stdin_stream(
        compensator=compensator,
        input_columns=input_columns,
        prediction_columns=prediction_columns,
        input_format=args.stdin_format,
        output_format=args.stdout_format,
        include_input=bool(args.include_input),
    )


if __name__ == "__main__":
    main()
