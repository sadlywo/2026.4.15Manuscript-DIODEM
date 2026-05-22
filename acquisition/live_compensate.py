from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Sequence, TextIO

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from acquisition.sinct485 import (  # noqa: E402
    DEFAULT_BAUDRATE,
    DEFAULT_SLAVE_ID,
    DEFAULT_TIMEOUT_SEC,
    INPUT_CHANNELS,
    Sinct485Error,
    Sinct485Reader,
)


PREDICTION_CHANNELS = ["pred_acc_x", "pred_acc_y", "pred_acc_z", "pred_gyr_x", "pred_gyr_y", "pred_gyr_z"]
CSV_FIELDS = [
    "timestamp_iso",
    "elapsed_s",
    "seq",
    "warmup",
    *INPUT_CHANNELS,
    *PREDICTION_CHANNELS,
    "read_latency_ms",
    "infer_latency_ms",
    "loop_latency_ms",
    "late_by_ms",
]
DEFAULT_CHECKPOINT = Path("outputs/supervised_tcn_causal_by_experiment/seed_runs/seed_0042/training/checkpoints/best.pt")
DEFAULT_OUTPUT_DIR = Path("outputs/live_capture")
DEFAULT_RATE_HZ = 40.0
DEFAULT_WARMUP_SAMPLES = 61


def _parse_int_auto(value: str) -> int:
    return int(str(value), 0)


def default_output_path(now: datetime | None = None) -> Path:
    current = now or datetime.now()
    return DEFAULT_OUTPUT_DIR / f"sinct485_{current.strftime('%Y%m%d_%H%M%S')}.csv"


def _sample_to_array(sample: Dict[str, float]) -> np.ndarray:
    return np.asarray([float(sample[channel]) for channel in INPUT_CHANNELS], dtype=np.float32)


def _format_row(
    *,
    timestamp_iso: str,
    elapsed_s: float,
    seq: int,
    warmup: bool,
    sample: Dict[str, float],
    prediction: Sequence[float],
    read_latency_ms: float,
    infer_latency_ms: float,
    loop_latency_ms: float,
    late_by_ms: float,
) -> Dict[str, float | int | bool | str]:
    row: Dict[str, float | int | bool | str] = {
        "timestamp_iso": timestamp_iso,
        "elapsed_s": float(elapsed_s),
        "seq": int(seq),
        "warmup": bool(warmup),
        "read_latency_ms": float(read_latency_ms),
        "infer_latency_ms": float(infer_latency_ms),
        "loop_latency_ms": float(loop_latency_ms),
        "late_by_ms": float(late_by_ms),
    }
    for channel in INPUT_CHANNELS:
        row[channel] = float(sample[channel])
    for channel, value in zip(PREDICTION_CHANNELS, prediction):
        row[channel] = float(value)
    return row


def _print_status(output_stream: TextIO | None, text: str) -> None:
    if output_stream is None:
        return
    print(text, file=output_stream, flush=True)


def run_capture(
    *,
    reader,
    compensator,
    output_csv: Path,
    rate_hz: float = DEFAULT_RATE_HZ,
    duration_sec: float | None = None,
    max_samples: int | None = None,
    max_consecutive_errors: int = 20,
    print_every: int = 40,
    warmup_samples: int = DEFAULT_WARMUP_SAMPLES,
    sleep_fn: Callable[[float], None] = time.sleep,
    output_stream: TextIO | None = sys.stdout,
    sample_callback: Callable[[Dict[str, float | int | bool | str]], None] | None = None,
) -> Dict[str, float | int | str]:
    """Run the live acquisition/inference loop and write a CSV log."""
    if rate_hz <= 0:
        raise ValueError("rate_hz must be positive.")
    if duration_sec is not None and duration_sec < 0:
        raise ValueError("duration_sec must be non-negative.")
    if max_samples is not None and max_samples < 0:
        raise ValueError("max_samples must be non-negative.")
    if max_consecutive_errors <= 0:
        raise ValueError("max_consecutive_errors must be positive.")

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    period_sec = 1.0 / float(rate_hz)
    start_time = time.perf_counter()
    samples_written = 0
    consecutive_errors = 0
    total_errors = 0

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        try:
            while True:
                if max_samples is not None and samples_written >= max_samples:
                    break
                elapsed_before = time.perf_counter() - start_time
                if duration_sec is not None and elapsed_before >= duration_sec:
                    break

                loop_start = time.perf_counter()
                timestamp_iso = datetime.now().astimezone().isoformat(timespec="milliseconds")
                try:
                    read_start = time.perf_counter()
                    sample = reader.read_sample()
                    read_latency_ms = (time.perf_counter() - read_start) * 1000.0
                except Sinct485Error as exc:
                    consecutive_errors += 1
                    total_errors += 1
                    _print_status(
                        output_stream,
                        f"[WARN] read failed ({consecutive_errors}/{max_consecutive_errors}): {exc}",
                    )
                    if consecutive_errors >= max_consecutive_errors:
                        raise RuntimeError(
                            f"Stopping after {consecutive_errors} consecutive SINCT-485 read errors."
                        ) from exc
                    sleep_fn(period_sec)
                    continue

                consecutive_errors = 0
                model_input = _sample_to_array(sample)
                inference = compensator.push(model_input)
                prediction = np.asarray(inference["prediction"], dtype=np.float32)
                infer_latency_ms = float(inference["latency_ms"])

                loop_latency_ms = (time.perf_counter() - loop_start) * 1000.0
                late_by_ms = max(0.0, loop_latency_ms - period_sec * 1000.0)
                row = _format_row(
                    timestamp_iso=timestamp_iso,
                    elapsed_s=time.perf_counter() - start_time,
                    seq=samples_written,
                    warmup=samples_written < int(warmup_samples),
                    sample=sample,
                    prediction=prediction,
                    read_latency_ms=read_latency_ms,
                    infer_latency_ms=infer_latency_ms,
                    loop_latency_ms=loop_latency_ms,
                    late_by_ms=late_by_ms,
                )
                writer.writerow(row)
                handle.flush()
                if sample_callback is not None:
                    sample_callback(row)
                samples_written += 1

                if print_every > 0 and samples_written % int(print_every) == 0:
                    _print_status(
                        output_stream,
                        (
                            f"[INFO] samples={samples_written} "
                            f"infer={infer_latency_ms:.3f} ms "
                            f"loop={loop_latency_ms:.3f} ms "
                            f"late={late_by_ms:.3f} ms"
                        ),
                    )

                next_tick = start_time + samples_written * period_sec
                remaining = next_tick - time.perf_counter()
                if remaining > 0:
                    sleep_fn(remaining)
        except KeyboardInterrupt:
            _print_status(output_stream, "\n[INFO] Interrupted by user; closing capture.")
        finally:
            reader.close()

    return {
        "output_csv": str(output_csv.resolve()),
        "samples_written": int(samples_written),
        "total_errors": int(total_errors),
        "elapsed_s": float(time.perf_counter() - start_time),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Acquire SINCT-485 IMU samples and run live DIODEM compensation.")
    parser.add_argument("--port", required=True, help="Serial port, e.g. COM3 on Windows or /dev/ttyUSB0 on Linux.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="Path to a causal TCN checkpoint.")
    parser.add_argument("--device", default="cpu", help="Torch device for inference, e.g. cpu or cuda.")
    parser.add_argument("--duration-sec", type=float, default=None, help="Capture duration. Omit to run until Ctrl+C.")
    parser.add_argument("--output-csv", type=Path, default=None, help="Output CSV path.")
    parser.add_argument("--baudrate", type=int, default=DEFAULT_BAUDRATE, help="Serial baud rate.")
    parser.add_argument("--slave-id", type=_parse_int_auto, default=DEFAULT_SLAVE_ID, help="Modbus slave id, e.g. 0x50.")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SEC, help="Serial read/write timeout seconds.")
    parser.add_argument("--rate-hz", type=float, default=DEFAULT_RATE_HZ, help="Polling rate in Hz.")
    parser.add_argument("--print-every", type=int, default=40, help="Print one status line every N samples; 0 disables.")
    parser.add_argument("--max-consecutive-errors", type=int, default=20, help="Stop after this many read errors in a row.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional finite sample limit for bench testing.")
    parser.add_argument("--plot", action="store_true", help="Open a real-time matplotlib window with raw and compensated streams.")
    parser.add_argument("--plot-window-sec", type=float, default=10.0, help="Visible time span for the live plot.")
    parser.add_argument(
        "--plot-update-interval-ms",
        type=float,
        default=100.0,
        help="Minimum redraw interval for the live plot.",
    )
    parser.add_argument(
        "--save-summary-figure",
        action="store_true",
        help="After capture, export a Nature-style static figure from the CSV.",
    )
    parser.add_argument("--summary-figure-stem", type=Path, default=None, help="Output stem for the static summary figure.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    from project.inference import StreamingCompensator
    from project.utils.torch_compat import require_torch

    require_torch()
    output_csv = args.output_csv or default_output_path()
    reader = Sinct485Reader(
        port=args.port,
        baudrate=args.baudrate,
        slave_id=args.slave_id,
        timeout=args.timeout,
    )
    compensator = StreamingCompensator.from_checkpoint(args.checkpoint, device_name=args.device)
    plotter = None
    sample_callback = None
    if args.plot:
        from acquisition.live_plot import LiveSignalPlotter

        plotter = LiveSignalPlotter(
            window_sec=args.plot_window_sec,
            rate_hz=args.rate_hz,
            update_interval_ms=args.plot_update_interval_ms,
        )
        sample_callback = plotter.push_row
    try:
        summary = run_capture(
            reader=reader,
            compensator=compensator,
            output_csv=output_csv,
            rate_hz=args.rate_hz,
            duration_sec=args.duration_sec,
            max_samples=args.max_samples,
            max_consecutive_errors=args.max_consecutive_errors,
            print_every=args.print_every,
            sample_callback=sample_callback,
        )
    finally:
        if plotter is not None:
            plotter.close()
    if args.save_summary_figure:
        from acquisition.plot_capture import create_capture_figure

        figure_stem = args.summary_figure_stem
        if figure_stem is None:
            figure_stem = Path(str(summary["output_csv"])).with_name(Path(str(summary["output_csv"])).stem + "_nature")
        figure_outputs = create_capture_figure(csv_path=Path(str(summary["output_csv"])), output_stem=figure_stem)
        for figure_output in figure_outputs:
            print(f"[INFO] Summary figure saved to {figure_output.resolve()}")
    print(
        f"[INFO] Capture saved to {summary['output_csv']} "
        f"({summary['samples_written']} samples, {summary['total_errors']} read errors)."
    )


if __name__ == "__main__":
    main()
