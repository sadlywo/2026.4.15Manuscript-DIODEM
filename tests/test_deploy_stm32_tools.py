import csv
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = REPO_ROOT / "tools" / "deploy_stm32"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))


def test_normalization_round_trip_uses_per_channel_stats():
    from deploy_common import denormalize_outputs, normalize_inputs

    stats = {
        "mode": "per_channel_zscore",
        "input_mean": [1, 2, 3, 4, 5, 6],
        "input_std": [1, 2, 4, 5, 10, 20],
        "target_mean": [-1, -2, -3, -4, -5, -6],
        "target_std": [2, 4, 8, 10, 20, 40],
    }
    raw = np.asarray([[2, 6, 11, 14, 25, 46]], dtype=np.float32)

    normalized = normalize_inputs(raw, stats)
    restored = denormalize_outputs(normalized, {
        **stats,
        "target_mean": stats["input_mean"],
        "target_std": stats["input_std"],
    })

    assert normalized.dtype == np.float32
    assert np.allclose(restored, raw)


def test_compare_metrics_reports_channel_errors():
    from compare_stm32_output import compute_error_metrics

    reference = np.asarray([[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7]], dtype=np.float32)
    actual = reference + np.asarray([0, 1, 0, -1, 0, 2], dtype=np.float32)

    metrics = compute_error_metrics(reference, actual, channel_names=["a", "b", "c", "d", "e", "f"])

    assert metrics["num_samples"] == 2
    assert metrics["max_abs_error"] == 2.0
    assert metrics["per_channel"]["b"]["rmse"] == 1.0
    assert metrics["per_channel"]["d"]["max_abs_error"] == 1.0


def test_stm32_csv_loader_accepts_prediction_columns(tmp_path):
    from compare_stm32_output import load_prediction_csv

    csv_path = tmp_path / "stm32.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["seq", "pred_acc_x", "pred_acc_y", "pred_acc_z", "pred_gyr_x", "pred_gyr_y", "pred_gyr_z"],
        )
        writer.writeheader()
        writer.writerow({
            "seq": 0,
            "pred_acc_x": 1,
            "pred_acc_y": 2,
            "pred_acc_z": 3,
            "pred_gyr_x": 4,
            "pred_gyr_y": 5,
            "pred_gyr_z": 6,
        })

    loaded = load_prediction_csv(csv_path)

    assert loaded.shape == (1, 6)
    assert loaded.dtype == np.float32
    assert np.allclose(loaded[0], [1, 2, 3, 4, 5, 6])


def test_serial_replay_parses_firmware_ok_line():
    from serial_replay_stm32 import format_sample, is_firmware_response, parse_ok_line

    parsed = parse_ok_line("63,OK,0.1,-0.2,9.8,0.01,0.02,-0.03,12345")

    assert parsed is not None
    assert parsed.seq == 63
    assert parsed.inference_us == 12345
    assert np.allclose(parsed.values, [0.1, -0.2, 9.8, 0.01, 0.02, -0.03])
    assert format_sample(np.arange(6, dtype=np.float32)) == "0,1,2,3,4,5\n"
    assert is_firmware_response("ERR,bad_csv")


def test_audit_report_records_missing_torch_without_crashing(tmp_path, monkeypatch):
    from audit_model import build_audit_report

    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"not a real torch checkpoint")
    stats_path = tmp_path / "normalization_stats.json"
    stats_path.write_text(json.dumps({"mode": "none"}), encoding="utf-8")

    report = build_audit_report(
        repo_root=tmp_path,
        checkpoint=checkpoint,
        processed_root=tmp_path,
        try_torch=False,
    )

    assert report["checkpoint"]["exists"] is True
    assert report["normalization"]["path"] == str(stats_path)
    assert report["torch"]["attempted"] is False
