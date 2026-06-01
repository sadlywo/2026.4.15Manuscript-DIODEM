from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

from deploy_common import (
    DEFAULT_CHECKPOINT,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PROCESSED_ROOT,
    denormalize_outputs,
    load_checkpoint_and_model,
    load_json,
    load_test_windows,
    normalize_inputs,
    repo_root_from_file,
    save_json,
)
from export_onnx import build_export_wrapper


def compute_metrics(reference: np.ndarray, actual: np.ndarray) -> Dict[str, Any]:
    delta = np.asarray(actual, dtype=np.float32) - np.asarray(reference, dtype=np.float32)
    return {
        "max_abs_error": float(np.max(np.abs(delta))) if delta.size else 0.0,
        "rmse": float(np.sqrt(np.mean(np.square(delta)))) if delta.size else 0.0,
        "mean_abs_error": float(np.mean(np.abs(delta))) if delta.size else 0.0,
        "per_channel_rmse": np.sqrt(np.mean(np.square(delta), axis=0)).astype(float).tolist()
        if delta.ndim == 2 and delta.shape[0] > 0
        else [],
    }


def run_onnx_inference(session: Any, input_name: str, normalized: np.ndarray) -> np.ndarray:
    expected_shape = session.get_inputs()[0].shape
    expected_batch = expected_shape[0] if expected_shape else None
    normalized = normalized.astype(np.float32)

    if isinstance(expected_batch, int) and expected_batch > 0 and expected_batch != normalized.shape[0]:
        outputs = []
        for index in range(normalized.shape[0]):
            window = normalized[index : index + 1]
            outputs.append(session.run(None, {input_name: window})[0])
        return np.concatenate(outputs, axis=0)

    return session.run(None, {input_name: normalized})[0]


def validate_onnx(
    checkpoint: Path,
    onnx_model: Path,
    processed_root: Path,
    split: str = "test",
    max_windows: int = 16,
    output_mode: str = "last_step",
    device: str = "cpu",
) -> Dict[str, Any]:
    import onnx
    import onnxruntime as ort

    torch, _, config, model = load_checkpoint_and_model(checkpoint, device_name=device)
    stats = load_json(Path(processed_root) / "normalization_stats.json")
    raw_windows = load_test_windows(processed_root, split=split, max_windows=max_windows)
    normalized = normalize_inputs(raw_windows, stats)

    wrapper = build_export_wrapper(torch, model, output_mode=output_mode)
    with torch.no_grad():
        torch_output = wrapper(torch.from_numpy(normalized).to(device)).detach().cpu().numpy()
    if output_mode == "last_step":
        torch_physical = denormalize_outputs(torch_output, stats)
    else:
        torch_physical = denormalize_outputs(torch_output, stats)

    onnx.checker.check_model(str(onnx_model))
    session = ort.InferenceSession(str(onnx_model), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    onnx_input_shape = session.get_inputs()[0].shape
    onnx_output = run_onnx_inference(session, input_name, normalized)
    onnx_physical = denormalize_outputs(onnx_output, stats)

    return {
        "checkpoint": str(Path(checkpoint).resolve()),
        "onnx_model": str(Path(onnx_model).resolve()),
        "split": split,
        "num_windows": int(raw_windows.shape[0]),
        "input_shape": list(normalized.shape),
        "onnx_input_shape": list(onnx_input_shape),
        "output_shape": list(onnx_output.shape),
        "model_name": str(config.get("model_name")),
        "output_mode": output_mode,
        "normalized_metrics": compute_metrics(torch_output.reshape(raw_windows.shape[0], -1), onnx_output.reshape(raw_windows.shape[0], -1)),
        "physical_metrics": compute_metrics(torch_physical.reshape(raw_windows.shape[0], -1), onnx_physical.reshape(raw_windows.shape[0], -1)),
    }


def main() -> None:
    repo_root = repo_root_from_file()
    parser = argparse.ArgumentParser(description="Validate ONNX output against PyTorch FP32 output.")
    parser.add_argument("--checkpoint", type=Path, default=repo_root / DEFAULT_CHECKPOINT)
    parser.add_argument("--onnx-model", type=Path, default=repo_root / DEFAULT_OUTPUT_ROOT / "onnx" / "tcn_causal_last_step.onnx")
    parser.add_argument("--processed-root", type=Path, default=repo_root / DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--max-windows", type=int, default=16)
    parser.add_argument("--output-mode", choices=["last_step", "sequence"], default="last_step")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=repo_root / DEFAULT_OUTPUT_ROOT / "onnx_validation_report.json")
    args = parser.parse_args()
    try:
        report = validate_onnx(
            checkpoint=args.checkpoint,
            onnx_model=args.onnx_model,
            processed_root=args.processed_root,
            split=args.split,
            max_windows=args.max_windows,
            output_mode=args.output_mode,
            device=args.device,
        )
    except Exception as exc:
        print(f"[ERROR] ONNX validation failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    save_json(report, args.output)
    print(f"Saved ONNX validation report to {args.output.resolve()}")


if __name__ == "__main__":
    main()
