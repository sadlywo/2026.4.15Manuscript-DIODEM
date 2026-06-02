import math
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = REPO_ROOT / "tools" / "deploy_stm32"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))


def test_platformio_memory_parser_reads_percentages_and_bytes():
    from run_streaming_tcn_experiments import parse_platformio_memory_usage

    output = """
    RAM:   [=         ]  10.6% (used 34848 bytes from 327680 bytes)
    Flash: [====      ]  41.4% (used 434564 bytes from 1048576 bytes)
    """

    usage = parse_platformio_memory_usage(output)

    assert usage["ram"]["used_bytes"] == 34848
    assert usage["ram"]["total_bytes"] == 327680
    assert math.isclose(usage["ram"]["used_percent"], 10.6)
    assert usage["flash"]["used_bytes"] == 434564
    assert usage["flash"]["total_bytes"] == 1048576
    assert math.isclose(usage["flash"]["used_percent"], 41.4)


def test_trial_summary_reports_mean_std_and_40hz_margin():
    from run_streaming_tcn_experiments import summarize_trials

    trials = [
        {"inference_us": 1390, "metrics": {"max_abs_error": 4.0e-7, "rmse": 2.0e-7}},
        {"inference_us": 1410, "metrics": {"max_abs_error": 5.0e-7, "rmse": 2.4e-7}},
        {"inference_us": 1400, "metrics": {"max_abs_error": 4.5e-7, "rmse": 2.2e-7}},
    ]

    summary = summarize_trials(trials, sampling_hz=40.0)

    assert summary["num_trials"] == 3
    assert math.isclose(summary["latency_us"]["mean"], 1400.0)
    assert math.isclose(summary["latency_us"]["std"], 10.0)
    assert math.isclose(summary["latency_ms"]["mean"], 1.4)
    assert math.isclose(summary["realtime"]["period_ms"], 25.0)
    assert math.isclose(summary["realtime"]["utilization_percent"], 5.6)
    assert math.isclose(summary["realtime"]["effective_max_sampling_hz"], 714.2857142857143)
    assert summary["realtime"]["supports_sampling_hz"] is True
    assert math.isclose(summary["accuracy"]["max_abs_error"]["mean"], 4.5e-7)
    assert math.isclose(summary["accuracy"]["rmse"]["mean"], 2.2e-7)


def test_report_names_measured_and_model_level_comparisons():
    from run_streaming_tcn_experiments import render_markdown_report

    report = render_markdown_report(
        config={
            "port": "COM6",
            "baud": 115200,
            "trials": 3,
            "sampling_hz": 40.0,
            "sample_delay_s": 0.05,
            "window_index": 0,
            "pytest_command": "python -m pytest tests/test_streaming_tcn_experiment_runner.py -q",
            "platformio_build_command": "pio run",
            "platformio_upload_command": "pio run -t upload",
        },
        pytest_result={"ran": True, "returncode": 0},
        build_result={"ran": True, "returncode": 0},
        upload_result={"ran": True, "returncode": 0},
        memory_usage={
            "ram": {"used_bytes": 34848, "total_bytes": 327680, "used_percent": 10.6},
            "flash": {"used_bytes": 434564, "total_bytes": 1048576, "used_percent": 41.4},
        },
        trial_summary={
            "num_trials": 3,
            "latency_us": {"mean": 1390.0, "std": 1.0, "min": 1389.0, "max": 1391.0},
            "latency_ms": {"mean": 1.39, "std": 0.001, "min": 1.389, "max": 1.391},
            "accuracy": {
                "max_abs_error": {"mean": 4.77e-7, "std": 0.0, "min": 4.77e-7, "max": 4.77e-7},
                "rmse": {"mean": 2.41e-7, "std": 0.0, "min": 2.41e-7, "max": 2.41e-7},
            },
            "realtime": {
                "sampling_hz": 40.0,
                "period_ms": 25.0,
                "utilization_percent": 5.56,
                "effective_max_sampling_hz": 719.42,
                "supports_sampling_hz": True,
            },
        },
        architecture_rows=[
            {
                "architecture": "TCN-causal",
                "scope": "Measured on STM32",
                "parameter_count": 101326,
                "fp32_size_mb": 0.3865,
                "int8_size_mb": None,
                "cpu_forward_ms_per_window": None,
                "stm32_latency_ms": 1.39,
                "notes": "Streaming forward_step.",
            },
            {
                "architecture": "GRU-causal",
                "scope": "Model-level comparison",
                "parameter_count": 152070,
                "fp32_size_mb": 0.5801,
                "int8_size_mb": None,
                "cpu_forward_ms_per_window": 3.01,
                "stm32_latency_ms": None,
                "notes": "Not measured on STM32 in this stage.",
            },
        ],
        cubeai_window_latency_ms=93.615,
        generated_at="2026-06-02 12:00:00",
    )

    assert "Measured on STM32" in report
    assert "Previous measured baseline" in report
    assert "Model-level comparison" in report
    assert "INT8" in report
    assert "40 Hz" in report
    assert "serial protocol" in report
    assert "67.3x" in report
    assert "GRU-causal" in report


def test_architecture_loader_reads_multiseed_model_summaries(tmp_path):
    from run_streaming_tcn_experiments import load_architecture_comparison

    metrics_dir = (
        tmp_path
        / "outputs"
        / "causal_model_comparison"
        / "gru_causal_by_motion_type"
        / "evaluation"
        / "metrics"
    )
    metrics_dir.mkdir(parents=True)
    csv_path = metrics_dir / "multiseed_model_deployment_summary.csv"
    csv_path.write_text(
        "\n".join(
            [
                "model_name,model_role,architecture_name,num_seeds,parameter_count_mean,parameter_count_std,"
                "trainable_parameter_count_mean,trainable_parameter_count_std,parameter_size_mb_fp32_mean,"
                "parameter_size_mb_fp32_std,cpu_forward_ms_per_window_mean,cpu_forward_ms_per_window_std,"
                "embedded_deployment_verdict",
                "gru,trained_comparison,gru,3,152070,0,152070,0,0.580101,0,3.0151,0.03,"
                "possible_on_embedded_linux",
            ]
        ),
        encoding="utf-8",
    )

    rows = load_architecture_comparison(tmp_path, requested_architectures=["gru"])

    assert len(rows) == 1
    assert rows[0]["architecture"] == "GRU-causal"
    assert rows[0]["scope"] == "Model-level comparison"
    assert rows[0]["parameter_count"] == 152070
    assert math.isclose(rows[0]["fp32_size_mb"], 0.580101)
    assert math.isclose(rows[0]["cpu_forward_ms_per_window"], 3.0151)
    assert rows[0]["stm32_latency_ms"] is None


def test_replay_command_includes_serial_timeout_controls(tmp_path):
    from run_streaming_tcn_experiments import build_replay_command

    command = build_replay_command(
        python_exe=Path("python.exe"),
        repo_root=tmp_path,
        port="COM6",
        baud=115200,
        window_index=0,
        sample_delay_s=0.1,
        response_timeout_s=5.0,
        inference_timeout_s=20.0,
        output_json=tmp_path / "trial.json",
    )

    assert "--response-timeout-s" in command
    assert "5.0" in command
    assert "--inference-timeout-s" in command
    assert "20.0" in command


def test_replay_trial_retries_transient_serial_failures(tmp_path):
    from run_streaming_tcn_experiments import run_replay_trial

    calls = []

    def fake_run_command(command, cwd, timeout_s):
        calls.append(command)
        if len(calls) == 1:
            return {"returncode": 1, "stdout": "", "stderr": "transient serial timeout"}
        output_json = Path(command[-1])
        output_json.write_text(
            json.dumps(
                {
                    "inference_us": 1390,
                    "metrics": {"max_abs_error": 4.0e-7, "rmse": 2.0e-7},
                }
            ),
            encoding="utf-8",
        )
        return {"returncode": 0, "stdout": "ok", "stderr": ""}

    trial = run_replay_trial(
        python_exe=Path("python.exe"),
        repo_root=tmp_path,
        port="COM6",
        baud=115200,
        window_index=0,
        sample_delay_s=0.1,
        response_timeout_s=5.0,
        inference_timeout_s=20.0,
        output_json=tmp_path / "trial.json",
        command_timeout_s=120.0,
        max_retries=1,
        command_runner=fake_run_command,
    )

    assert len(calls) == 2
    assert trial["inference_us"] == 1390
    assert trial["replay_attempts"] == 2
    assert trial["failed_replay_attempts"][0]["returncode"] == 1
