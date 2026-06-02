from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from deploy_common import DEFAULT_OUTPUT_ROOT, repo_root_from_file, save_json


DEFAULT_PIO_EXE = Path.home() / ".platformio" / "penv" / "Scripts" / "pio.exe"
DEFAULT_PINN_IMU_PYTHON = Path("D:/Anaconda3/envs/pinn_imu/python.exe")
DEFAULT_FIRMWARE_ROOT = Path("firmware/platformio_nucleo_h723zg")
DEFAULT_REPORT = Path(DEFAULT_OUTPUT_ROOT) / "streaming_tcn_speed_report.md"
DEFAULT_EXPERIMENT_ROOT = Path(DEFAULT_OUTPUT_ROOT) / "experiments"
DEFAULT_CUBEAI_WINDOW_LATENCY_MS = 93.615
DEFAULT_SAMPLING_HZ = 40.0
REQUESTED_COMPARISON_ARCHITECTURES = ["gru", "lstm", "transformer", "mlp"]


def quote_command(command: list[str]) -> str:
    return subprocess.list2cmdline([str(item) for item in command])


def resolve_executable(path: Path, fallback: Path) -> Path:
    return path if path.exists() else fallback


def run_command(command: list[str], cwd: Path, timeout_s: float | None = None) -> dict[str, Any]:
    start = time.perf_counter()
    completed = subprocess.run(
        [str(item) for item in command],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    duration_s = time.perf_counter() - start
    return {
        "ran": True,
        "command": quote_command(command),
        "cwd": str(cwd.resolve()),
        "returncode": int(completed.returncode),
        "duration_s": float(duration_s),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def build_replay_command(
    python_exe: Path,
    repo_root: Path,
    port: str,
    baud: int,
    window_index: int,
    sample_delay_s: float,
    response_timeout_s: float,
    inference_timeout_s: float,
    output_json: Path,
) -> list[str]:
    return [
        str(python_exe),
        str(repo_root / "tools" / "deploy_stm32" / "serial_replay_stm32.py"),
        "--port",
        str(port),
        "--baud",
        str(baud),
        "--window-index",
        str(window_index),
        "--sample-delay-s",
        str(sample_delay_s),
        "--response-timeout-s",
        str(response_timeout_s),
        "--inference-timeout-s",
        str(inference_timeout_s),
        "--reset-target",
        "--output-json",
        str(output_json),
    ]


def run_replay_trial(
    python_exe: Path,
    repo_root: Path,
    port: str,
    baud: int,
    window_index: int,
    sample_delay_s: float,
    response_timeout_s: float,
    inference_timeout_s: float,
    output_json: Path,
    command_timeout_s: float,
    max_retries: int,
    command_runner=run_command,
) -> dict[str, Any]:
    failed_attempts: list[dict[str, Any]] = []
    attempts = int(max_retries) + 1
    replay_command = build_replay_command(
        python_exe=python_exe,
        repo_root=repo_root,
        port=port,
        baud=baud,
        window_index=window_index,
        sample_delay_s=sample_delay_s,
        response_timeout_s=response_timeout_s,
        inference_timeout_s=inference_timeout_s,
        output_json=output_json,
    )
    for attempt_index in range(attempts):
        result = command_runner(replay_command, cwd=repo_root, timeout_s=command_timeout_s)
        if int(result.get("returncode", 1)) == 0:
            with output_json.open("r", encoding="utf-8") as handle:
                trial = json.load(handle)
            trial["replay_command"] = quote_command(replay_command)
            trial["replay_returncode"] = int(result["returncode"])
            trial["replay_attempts"] = attempt_index + 1
            trial["failed_replay_attempts"] = failed_attempts
            return trial
        failed_attempts.append(
            {
                "attempt": attempt_index + 1,
                "returncode": int(result.get("returncode", 1)),
                "stdout_tail": str(result.get("stdout") or "")[-1000:],
                "stderr_tail": str(result.get("stderr") or "")[-1000:],
            }
        )
        if output_json.exists():
            output_json.unlink()
    _raise_if_failed(failed_attempts[-1], "STM32 replay")
    raise RuntimeError("unreachable")


def parse_platformio_memory_usage(output: str) -> dict[str, dict[str, float | int]]:
    usage: dict[str, dict[str, float | int]] = {}
    pattern = re.compile(
        r"^\s*(RAM|Flash):.*?([0-9.]+)%\s*"
        r"\(used\s+([0-9]+)\s+bytes\s+from\s+([0-9]+)\s+bytes\)",
        re.IGNORECASE | re.MULTILINE,
    )
    for match in pattern.finditer(output):
        key = match.group(1).lower()
        usage[key] = {
            "used_percent": float(match.group(2)),
            "used_bytes": int(match.group(3)),
            "total_bytes": int(match.group(4)),
        }
    return usage


def stat_block(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("Cannot summarize an empty value list.")
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "mean": float(statistics.mean(values)),
        "std": float(std),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def summarize_trials(trials: list[dict[str, Any]], sampling_hz: float = DEFAULT_SAMPLING_HZ) -> dict[str, Any]:
    if not trials:
        raise ValueError("At least one STM32 trial is required to summarize latency.")
    latency_us = [float(trial["inference_us"]) for trial in trials]
    max_abs_errors = [float(trial["metrics"]["max_abs_error"]) for trial in trials]
    rmses = [float(trial["metrics"]["rmse"]) for trial in trials]
    latency_us_stats = stat_block(latency_us)
    latency_ms = [value / 1000.0 for value in latency_us]
    period_ms = 1000.0 / float(sampling_hz)
    mean_latency_us = latency_us_stats["mean"]
    utilization_percent = mean_latency_us / (1_000_000.0 / float(sampling_hz)) * 100.0
    effective_hz = 1_000_000.0 / mean_latency_us if mean_latency_us > 0 else 0.0
    return {
        "num_trials": int(len(trials)),
        "latency_us": latency_us_stats,
        "latency_ms": stat_block(latency_ms),
        "accuracy": {
            "max_abs_error": stat_block(max_abs_errors),
            "rmse": stat_block(rmses),
        },
        "realtime": {
            "sampling_hz": float(sampling_hz),
            "period_ms": float(period_ms),
            "utilization_percent": float(utilization_percent),
            "effective_max_sampling_hz": float(effective_hz),
            "supports_sampling_hz": bool(mean_latency_us <= 1_000_000.0 / float(sampling_hz)),
        },
    }


def _first_float(row: dict[str, str], keys: list[str]) -> float | None:
    for key in keys:
        value = row.get(key)
        if value not in {None, ""}:
            return float(value)
    return None


def _first_int(row: dict[str, str], keys: list[str]) -> int | None:
    value = _first_float(row, keys)
    return int(round(value)) if value is not None else None


def _display_architecture(architecture: str) -> str:
    labels = {
        "tcn": "TCN-causal",
        "tcn_causal": "TCN-causal",
        "gru": "GRU-causal",
        "lstm": "LSTM-causal",
        "transformer": "Transformer-causal",
        "mlp": "MLP-causal",
    }
    return labels.get(architecture, architecture)


def _architecture_summary_candidates(repo_root: Path, architecture: str) -> list[Path]:
    base = repo_root / "outputs" / "causal_model_comparison"
    preferred = [
        base / f"{architecture}_causal_by_motion_type" / "evaluation" / "metrics" / "multiseed_model_deployment_summary.csv",
        base / f"{architecture}_causal_by_experiment" / "evaluation" / "metrics" / "multiseed_model_deployment_summary.csv",
        base / f"{architecture}_causal_anomaly_test_only" / "evaluation" / "metrics" / "multiseed_model_deployment_summary.csv",
    ]
    if architecture == "tcn":
        preferred.append(
            repo_root / "outputs" / "supervised_by_motion_type" / "evaluation" / "metrics" / "multiseed_model_deployment_summary.csv"
        )
    discovered = sorted(base.rglob("multiseed_model_deployment_summary.csv")) if base.exists() else []
    matching = [
        path
        for path in discovered
        if f"{architecture}_causal_" in str(path).replace("\\", "/")
    ]
    return [path for path in [*preferred, *matching] if path.exists()]


def _relative_source(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _row_from_csv(path: Path, architecture: str, repo_root: Path) -> dict[str, Any] | None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row_arch = str(row.get("architecture_name") or row.get("model_name") or "").lower()
            if row_arch and architecture not in row_arch:
                continue
            return {
                "architecture": _display_architecture(architecture),
                "scope": "Model-level comparison",
                "parameter_count": _first_int(row, ["parameter_count_mean", "parameter_count"]),
                "fp32_size_mb": _first_float(row, ["parameter_size_mb_fp32_mean", "parameter_size_mb_fp32"]),
                "int8_size_mb": None,
                "cpu_forward_ms_per_window": _first_float(
                    row,
                    ["cpu_forward_ms_per_window_mean", "cpu_forward_ms_per_window"],
                ),
                "stm32_latency_ms": None,
                "notes": f"From {_relative_source(path, repo_root)}; not measured on STM32 in this stage.",
            }
    return None


def _audit_fallback_row(repo_root: Path, architecture: str) -> dict[str, Any] | None:
    audit_path = repo_root / DEFAULT_OUTPUT_ROOT / "model_audit_pinn_imu.json"
    if not audit_path.exists():
        return None
    with audit_path.open("r", encoding="utf-8") as handle:
        audit = json.load(handle)
    for model in audit.get("deployment_summary", {}).get("models", []):
        row_arch = str(model.get("architecture_name") or model.get("model_name") or "").lower()
        if architecture not in row_arch:
            continue
        return {
            "architecture": _display_architecture(architecture),
            "scope": "Model-level comparison",
            "parameter_count": int(model["parameter_count"]),
            "fp32_size_mb": float(model["parameter_size_mb_fp32"]),
            "int8_size_mb": None,
            "cpu_forward_ms_per_window": float(model["cpu_forward_ms_per_window"]),
            "stm32_latency_ms": None,
            "notes": f"From {audit_path.relative_to(repo_root)}; not measured on STM32 in this stage.",
        }
    return None


def load_architecture_comparison(
    repo_root: Path,
    requested_architectures: list[str] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for architecture in requested_architectures or REQUESTED_COMPARISON_ARCHITECTURES:
        selected: dict[str, Any] | None = None
        for candidate in _architecture_summary_candidates(repo_root, architecture):
            selected = _row_from_csv(candidate, architecture, repo_root)
            if selected is not None:
                break
        if selected is None:
            selected = _audit_fallback_row(repo_root, architecture)
        if selected is not None:
            rows.append(selected)
    return rows


def measured_tcn_row(trial_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "architecture": "TCN-causal",
        "scope": "Measured on STM32",
        "parameter_count": 101326,
        "fp32_size_mb": 0.38652801513671875,
        "int8_size_mb": None,
        "cpu_forward_ms_per_window": None,
        "stm32_latency_ms": float(trial_summary["latency_ms"]["mean"]),
        "notes": "Handwritten streaming forward_step on NUCLEO-H723ZG.",
    }


def _format_number(value: Any, precision: int = 4) -> str:
    if value is None:
        return "not measured"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{precision}g}"
    return str(value)


def _format_mean_std(block: dict[str, float], precision: int = 4) -> str:
    return f"{block['mean']:.{precision}g} +/- {block['std']:.{precision}g}"


def _memory_line(label: str, usage: dict[str, Any] | None) -> str:
    if not usage:
        return f"- {label}: not parsed"
    return (
        f"- {label}: {usage['used_percent']:.1f}% "
        f"({usage['used_bytes']} / {usage['total_bytes']} bytes)"
    )


def render_markdown_report(
    config: dict[str, Any],
    pytest_result: dict[str, Any],
    build_result: dict[str, Any],
    upload_result: dict[str, Any],
    memory_usage: dict[str, dict[str, float | int]],
    trial_summary: dict[str, Any],
    architecture_rows: list[dict[str, Any]],
    cubeai_window_latency_ms: float = DEFAULT_CUBEAI_WINDOW_LATENCY_MS,
    generated_at: str | None = None,
) -> str:
    generated_at = generated_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    streaming_latency_ms = float(trial_summary["latency_ms"]["mean"])
    speedup = cubeai_window_latency_ms / streaming_latency_ms if streaming_latency_ms > 0 else 0.0
    realtime = trial_summary["realtime"]

    rows = [
        "| Architecture | Scope | Params | FP32 MB | INT8 MB | PC per-window ms | STM32 latency ms | Notes |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in architecture_rows:
        rows.append(
            "| "
            + " | ".join(
                [
                    str(row["architecture"]),
                    str(row["scope"]),
                    _format_number(row.get("parameter_count"), 0),
                    _format_number(row.get("fp32_size_mb"), 4),
                    _format_number(row.get("int8_size_mb"), 4),
                    _format_number(row.get("cpu_forward_ms_per_window"), 4),
                    _format_number(row.get("stm32_latency_ms"), 4),
                    str(row.get("notes") or ""),
                ]
            )
            + " |"
        )

    status = "PASS" if realtime["supports_sampling_hz"] else "FAIL"
    return "\n".join(
        [
            "# Streaming Causal TCN STM32 Speed Report",
            "",
            f"Generated at: {generated_at}",
            "",
            "## Experiment Configuration",
            "",
            f"- Board: NUCLEO-H723ZG",
            f"- Serial port: {config['port']} @ {config['baud']} baud",
            f"- Trials: {config['trials']}",
            f"- Input stream: 6-axis IMU at {config['sampling_hz']:.0f} Hz",
            f"- Window index: {config['window_index']}",
            f"- Host sample delay during replay: {config['sample_delay_s']} s",
            f"- Pytest command: `{config['pytest_command']}`",
            f"- PlatformIO build command: `{config['platformio_build_command']}`",
            f"- PlatformIO upload command: `{config['platformio_upload_command']}`",
            "",
            "## Toolchain Results",
            "",
            f"- pytest: return code {pytest_result.get('returncode', 'not run')}",
            f"- PlatformIO build: return code {build_result.get('returncode', 'not run')}",
            f"- PlatformIO upload: return code {upload_result.get('returncode', 'not run')}",
            _memory_line("RAM", memory_usage.get("ram")),
            _memory_line("Flash", memory_usage.get("flash")),
            "",
            "## Measured on STM32",
            "",
            f"- Streaming TCN latency: {_format_mean_std(trial_summary['latency_ms'])} ms",
            f"- Raw inference time: {_format_mean_std(trial_summary['latency_us'])} us",
            f"- Max abs error: {_format_mean_std(trial_summary['accuracy']['max_abs_error'], 3)}",
            f"- RMSE: {_format_mean_std(trial_summary['accuracy']['rmse'], 3)}",
            f"- Effective max sampling frequency: {realtime['effective_max_sampling_hz']:.2f} Hz",
            f"- 40 Hz period: {realtime['period_ms']:.2f} ms",
            f"- Compute utilization at 40 Hz: {realtime['utilization_percent']:.2f}%",
            f"- 40 Hz real-time verdict: {status}",
            "",
            "## Previous measured baseline",
            "",
            f"- Cube.AI 64-point window latency: {cubeai_window_latency_ms:.3f} ms",
            f"- Streaming forward_step speed-up: {speedup:.1f}x",
            "",
            "## Model-level comparison",
            "",
            *rows,
            "",
            "## INT8 Status",
            "",
            "- No measured INT8 STM32 artifact is included in this experiment stage.",
            "- INT8 model sizes are therefore reported as `not measured`, not inferred measurements.",
            "- PTQ/QAT should be evaluated only after the FP32 streaming path is locked and compared against the same golden vectors.",
            "",
            "## Interpretation",
            "",
            f"- The measured computation time is far below the {realtime['period_ms']:.2f} ms period required for 40 Hz streaming.",
            "- The TCN inference itself is not the bottleneck for this setup.",
            "- The serial protocol and host-to-board data transfer can dominate replay wall time, especially with text CSV frames.",
            "- Recommendation: keep the handwritten streaming causal TCN path for NUCLEO-H723ZG FP32 deployment, then evaluate binary serial framing and INT8 only if system-level bandwidth or memory demands require it.",
            "",
        ]
    )


def _raise_if_failed(result: dict[str, Any], label: str) -> None:
    if int(result.get("returncode", 0)) != 0:
        stdout = str(result.get("stdout") or "").strip()
        stderr = str(result.get("stderr") or "").strip()
        detail = "\n".join([item for item in [stdout, stderr] if item])
        raise RuntimeError(f"{label} failed with return code {result['returncode']}:\n{detail}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build, upload, replay, and report STM32 streaming causal TCN real-time performance."
    )
    parser.add_argument("--port", default="COM6")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--sample-delay-s", type=float, default=0.05)
    parser.add_argument("--response-timeout-s", type=float, default=5.0)
    parser.add_argument("--inference-timeout-s", type=float, default=20.0)
    parser.add_argument("--trial-retries", type=int, default=2)
    parser.add_argument("--inter-trial-delay-s", type=float, default=1.0)
    parser.add_argument("--window-index", type=int, default=0)
    parser.add_argument("--sampling-hz", type=float, default=DEFAULT_SAMPLING_HZ)
    parser.add_argument("--pio-exe", type=Path, default=DEFAULT_PIO_EXE)
    parser.add_argument("--python-exe", type=Path, default=DEFAULT_PINN_IMU_PYTHON)
    parser.add_argument("--firmware-root", type=Path, default=DEFAULT_FIRMWARE_ROOT)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--cubeai-window-latency-ms", type=float, default=DEFAULT_CUBEAI_WINDOW_LATENCY_MS)
    parser.add_argument("--skip-pytest", action="store_true")
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    return parser


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = repo_root_from_file()
    firmware_root = (repo_root / args.firmware_root).resolve()
    experiment_root = (repo_root / args.experiment_root).resolve()
    report_path = (repo_root / args.report).resolve()
    experiment_root.mkdir(parents=True, exist_ok=True)

    python_exe = resolve_executable(Path(args.python_exe), Path(sys.executable))
    pio_exe = Path(args.pio_exe)
    if not pio_exe.exists():
        raise FileNotFoundError(f"PlatformIO executable was not found: {pio_exe}")
    if int(args.trials) < 1:
        raise ValueError("--trials must be at least 1.")

    pytest_command = [
        str(python_exe),
        "-m",
        "pytest",
        "tests/test_deploy_stm32_tools.py",
        "tests/test_platformio_nucleo_h723zg_project.py",
        "tests/test_streaming_tcn_experiment_runner.py",
        "-q",
    ]
    build_command = [str(pio_exe), "run"]
    upload_command = [str(pio_exe), "run", "-t", "upload"]
    config = {
        "port": args.port,
        "baud": int(args.baud),
        "trials": int(args.trials),
        "sampling_hz": float(args.sampling_hz),
        "sample_delay_s": float(args.sample_delay_s),
        "response_timeout_s": float(args.response_timeout_s),
        "inference_timeout_s": float(args.inference_timeout_s),
        "trial_retries": int(args.trial_retries),
        "inter_trial_delay_s": float(args.inter_trial_delay_s),
        "window_index": int(args.window_index),
        "pytest_command": quote_command(pytest_command),
        "platformio_build_command": quote_command(build_command),
        "platformio_upload_command": quote_command(upload_command),
    }

    if args.skip_pytest:
        pytest_result = {"ran": False, "returncode": "skipped"}
    else:
        print("[1/4] Running pytest validation...")
        pytest_result = run_command(pytest_command, cwd=repo_root, timeout_s=180)
        _raise_if_failed(pytest_result, "pytest validation")

    if args.skip_build:
        build_result = {"ran": False, "returncode": "skipped", "stdout": "", "stderr": ""}
        memory_usage: dict[str, dict[str, float | int]] = {}
    else:
        print("[2/4] Building PlatformIO firmware...")
        build_result = run_command(build_command, cwd=firmware_root, timeout_s=300)
        _raise_if_failed(build_result, "PlatformIO build")
        memory_usage = parse_platformio_memory_usage(
            str(build_result.get("stdout") or "") + "\n" + str(build_result.get("stderr") or "")
        )

    if args.skip_upload:
        upload_result = {"ran": False, "returncode": "skipped"}
    else:
        print("[3/4] Uploading firmware to NUCLEO-H723ZG...")
        upload_result = run_command(upload_command, cwd=firmware_root, timeout_s=300)
        _raise_if_failed(upload_result, "PlatformIO upload")

    print(f"[4/4] Replaying {args.trials} STM32 trial(s)...")
    trials: list[dict[str, Any]] = []
    for trial_index in range(int(args.trials)):
        if trial_index > 0 and float(args.inter_trial_delay_s) > 0:
            time.sleep(float(args.inter_trial_delay_s))
        output_json = experiment_root / f"streaming_tcn_trial_{trial_index:03d}.json"
        print(f"  trial {trial_index + 1}/{args.trials}: {output_json.name}")
        trial = run_replay_trial(
            python_exe=python_exe,
            repo_root=repo_root,
            port=str(args.port),
            baud=int(args.baud),
            window_index=int(args.window_index),
            sample_delay_s=float(args.sample_delay_s),
            response_timeout_s=float(args.response_timeout_s),
            inference_timeout_s=float(args.inference_timeout_s),
            output_json=output_json,
            command_timeout_s=120.0,
            max_retries=int(args.trial_retries),
        )
        trials.append(trial)

    trial_summary = summarize_trials(trials, sampling_hz=float(args.sampling_hz))
    architecture_rows = [
        measured_tcn_row(trial_summary),
        *load_architecture_comparison(repo_root, REQUESTED_COMPARISON_ARCHITECTURES),
    ]
    summary = {
        "config": config,
        "pytest_result": pytest_result,
        "build_result": build_result,
        "upload_result": upload_result,
        "memory_usage": memory_usage,
        "trials": trials,
        "trial_summary": trial_summary,
        "architecture_rows": architecture_rows,
        "cubeai_window_latency_ms": float(args.cubeai_window_latency_ms),
    }
    summary_path = experiment_root / "streaming_tcn_speed_summary.json"
    save_json(summary, summary_path)

    report = render_markdown_report(
        config=config,
        pytest_result=pytest_result,
        build_result=build_result,
        upload_result=upload_result,
        memory_usage=memory_usage,
        trial_summary=trial_summary,
        architecture_rows=architecture_rows,
        cubeai_window_latency_ms=float(args.cubeai_window_latency_ms),
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(f"Saved summary: {summary_path}")
    print(f"Saved report: {report_path}")
    return summary


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        run_experiment(args)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
