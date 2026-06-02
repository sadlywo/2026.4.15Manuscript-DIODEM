from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List

import numpy as np

from compare_stm32_output import compute_error_metrics
from deploy_common import DEFAULT_OUTPUT_ROOT, PREDICTION_CHANNELS, repo_root_from_file, save_json


@dataclass
class ParsedOkLine:
    seq: int
    values: np.ndarray
    inference_us: int
    raw: str


def parse_ok_line(line: str) -> ParsedOkLine | None:
    parts = [part.strip() for part in line.strip().split(",")]
    if len(parts) != 9 or parts[1] != "OK":
        return None
    try:
        values = np.asarray([float(item) for item in parts[2:8]], dtype=np.float32)
        inference_us = int(float(parts[8]))
        return ParsedOkLine(seq=int(parts[0]), values=values, inference_us=inference_us, raw=line.strip())
    except ValueError:
        return None


def is_firmware_response(line: str) -> bool:
    parts = [part.strip() for part in line.strip().split(",")]
    if parts and parts[0] == "ERR":
        return True
    return len(parts) >= 2 and parts[1] in {"WARMUP", "OK", "ERR"}


def format_sample(sample: np.ndarray) -> str:
    values = np.asarray(sample, dtype=np.float32).reshape(-1)
    if values.shape[0] != 6:
        raise ValueError(f"Expected six IMU channels, got {values.shape[0]}")
    return ",".join(f"{float(value):.9g}" for value in values) + "\n"


def default_openocd_paths() -> tuple[Path, Path]:
    package_root = Path.home() / ".platformio" / "packages" / "tool-openocd"
    exe = package_root / "bin" / "openocd.exe"
    scripts = package_root / "scripts"
    if not scripts.exists():
        scripts = package_root / "openocd" / "scripts"
    return exe, scripts


def reset_target_with_openocd() -> None:
    exe, scripts = default_openocd_paths()
    if not exe.exists() or not scripts.exists():
        raise FileNotFoundError(
            "OpenOCD was not found under PlatformIO packages. "
            "Install/build the PlatformIO project first, or reset the board manually."
        )
    command = [
        str(exe),
        "-s",
        str(scripts),
        "-f",
        "interface/stlink.cfg",
        "-f",
        "target/stm32h7x.cfg",
        "-c",
        "init; reset run; sleep 300; shutdown",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=15, check=False)
    if completed.returncode != 0:
        detail = "\n".join([completed.stdout.strip(), completed.stderr.strip()]).strip()
        raise RuntimeError(f"OpenOCD target reset failed:\n{detail}")


def read_until(
    serial_port,
    predicate: Callable[[str], bool],
    timeout_s: float,
    log: List[str],
) -> str | None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        raw = serial_port.readline()
        if not raw:
            continue
        line = raw.decode("utf-8", errors="replace").strip()
        if not line:
            continue
        log.append(line)
        if predicate(line):
            return line
    return None


def replay_one_window(
    vectors_path: Path,
    port: str,
    baud: int,
    window_index: int,
    output_json: Path,
    reset_target: bool = False,
    response_timeout_s: float = 2.0,
    inference_timeout_s: float = 20.0,
    sample_delay_s: float = 0.02,
) -> dict:
    try:
        import serial  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on local runtime
        raise RuntimeError("pyserial is required. Install it in the active Python environment.") from exc

    vectors = np.load(vectors_path)
    raw_windows = np.asarray(vectors["raw_windows"], dtype=np.float32)
    references = np.asarray(vectors["physical_reference"], dtype=np.float32)
    if raw_windows.ndim != 3 or raw_windows.shape[1:] != (64, 6):
        raise ValueError(f"Expected raw_windows shape [N, 64, 6], got {raw_windows.shape}")
    if references.ndim != 2 or references.shape[1] != 6:
        raise ValueError(f"Expected physical_reference shape [N, 6], got {references.shape}")
    if window_index < 0 or window_index >= raw_windows.shape[0]:
        raise IndexError(f"window_index {window_index} is outside available range 0..{raw_windows.shape[0] - 1}")

    if reset_target:
        reset_target_with_openocd()
        time.sleep(0.5)

    serial_log: List[str] = []
    ok_line: ParsedOkLine | None = None
    with serial.Serial(port=port, baudrate=baud, timeout=0.1, write_timeout=2.0) as serial_port:
        serial_port.reset_input_buffer()
        serial_port.reset_output_buffer()
        for sample_index, sample in enumerate(raw_windows[window_index]):
            serial_port.write(format_sample(sample).encode("ascii"))
            serial_port.flush()
            timeout = inference_timeout_s if sample_index == 63 else response_timeout_s
            line = read_until(serial_port, is_firmware_response, timeout, serial_log)
            if line is None:
                raise TimeoutError(f"No firmware response after sample {sample_index} within {timeout:.1f}s")
            parsed = parse_ok_line(line)
            if parsed is not None:
                ok_line = parsed
                break
            if line.startswith("ERR") or ",ERR" in line:
                raise RuntimeError(f"Firmware returned error after sample {sample_index}: {line}")
            if sample_delay_s > 0:
                time.sleep(sample_delay_s)

    if ok_line is None:
        raise RuntimeError("The board never returned an OK line; the 64-sample window did not complete.")

    reference = references[window_index].reshape(1, 6)
    actual = ok_line.values.reshape(1, 6)
    metrics = compute_error_metrics(reference, actual, channel_names=PREDICTION_CHANNELS)
    report = {
        "port": port,
        "baud": baud,
        "vectors_path": str(vectors_path.resolve()),
        "window_index": int(window_index),
        "fresh_reset_expected_seq": 63,
        "observed_seq": int(ok_line.seq),
        "fresh_reset_likely": bool(ok_line.seq == 63),
        "inference_us": int(ok_line.inference_us),
        "reference": {name: float(reference[0, index]) for index, name in enumerate(PREDICTION_CHANNELS)},
        "stm32": {name: float(actual[0, index]) for index, name in enumerate(PREDICTION_CHANNELS)},
        "metrics": metrics,
        "serial_log": serial_log,
    }
    save_json(report, output_json)
    return report


def main() -> None:
    repo_root = repo_root_from_file()
    parser = argparse.ArgumentParser(description="Replay one 64-sample golden IMU window to the STM32 firmware.")
    parser.add_argument("--port", default="COM6", help="ST-LINK virtual COM port, for example COM6.")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument(
        "--vectors",
        type=Path,
        default=repo_root / DEFAULT_OUTPUT_ROOT / "test_vectors" / "stm32_golden_vectors.npz",
    )
    parser.add_argument("--window-index", type=int, default=0)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=repo_root / DEFAULT_OUTPUT_ROOT / "stm32_serial_replay_window0.json",
    )
    parser.add_argument(
        "--reset-target",
        action="store_true",
        help="Reset the NUCLEO-H723ZG with PlatformIO OpenOCD before replaying the window.",
    )
    parser.add_argument("--response-timeout-s", type=float, default=2.0)
    parser.add_argument("--inference-timeout-s", type=float, default=20.0)
    parser.add_argument("--sample-delay-s", type=float, default=0.02)
    args = parser.parse_args()

    try:
        report = replay_one_window(
            vectors_path=args.vectors,
            port=args.port,
            baud=args.baud,
            window_index=args.window_index,
            output_json=args.output_json,
            reset_target=args.reset_target,
            response_timeout_s=args.response_timeout_s,
            inference_timeout_s=args.inference_timeout_s,
            sample_delay_s=args.sample_delay_s,
        )
    except Exception as exc:
        print(f"[ERROR] STM32 serial replay failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(f"Saved report: {args.output_json.resolve()}")
    print(f"Observed seq: {report['observed_seq']} (fresh reset likely: {report['fresh_reset_likely']})")
    print(f"Inference time: {report['inference_us']} us")
    print(f"Max abs error: {report['metrics']['max_abs_error']:.9g}")
    print(f"RMSE: {report['metrics']['rmse']:.9g}")


if __name__ == "__main__":
    main()
