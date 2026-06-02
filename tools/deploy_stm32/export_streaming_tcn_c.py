from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable

import numpy as np

from deploy_common import DEFAULT_CHECKPOINT, DEFAULT_OUTPUT_ROOT, load_checkpoint_and_model, repo_root_from_file, save_json


ARCHIVED_DEFAULT_CHECKPOINT = Path(
    "outputs/_archive/2026-06-01_result_cleanup/"
    "supervised_tcn_causal_by_experiment/seed_runs/seed_0042/training/checkpoints/best.pt"
)


def resolve_checkpoint(repo_root: Path, requested: Path | None) -> Path:
    candidates = []
    if requested is not None:
        candidates.append(requested)
    candidates.extend([repo_root / DEFAULT_CHECKPOINT, repo_root / ARCHIVED_DEFAULT_CHECKPOINT])
    for candidate in candidates:
        path = Path(candidate)
        if not path.is_absolute():
            path = repo_root / path
        if path.exists():
            return path.resolve()
    tried = "\n".join(str(Path(item)) for item in candidates)
    raise FileNotFoundError(f"Could not locate a TCN checkpoint. Tried:\n{tried}")


def tensor_dict(model) -> Dict[str, np.ndarray]:
    return {
        name: value.detach().cpu().numpy().astype(np.float32)
        for name, value in model.state_dict().items()
    }


def linear(weight: np.ndarray, bias: np.ndarray, inputs: np.ndarray) -> np.ndarray:
    return (weight @ inputs + bias).astype(np.float32)


def sigmoid(values: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-values))).astype(np.float32)


def conv_step(
    weight: np.ndarray,
    bias: np.ndarray,
    history: np.ndarray,
    current: np.ndarray,
    dilation: int,
) -> np.ndarray:
    out_dim, in_dim, kernel = weight.shape
    if kernel != 3:
        raise ValueError(f"Only kernel_size=3 is supported for the STM32 streaming path, got {kernel}")
    outputs = np.array(bias, dtype=np.float32, copy=True)
    for out_ch in range(out_dim):
        acc = np.float32(outputs[out_ch])
        for in_ch in range(in_dim):
            acc += weight[out_ch, in_ch, 0] * history[in_ch, 0]
            acc += weight[out_ch, in_ch, 1] * history[in_ch, dilation]
            acc += weight[out_ch, in_ch, 2] * current[in_ch]
        outputs[out_ch] = acc
    return outputs


def update_history(history: np.ndarray, current: np.ndarray, history_len: int) -> None:
    history[:, : history_len - 1] = history[:, 1:history_len]
    history[:, history_len - 1] = current


def run_numpy_stream(weights: Dict[str, np.ndarray], sequence: np.ndarray) -> np.ndarray:
    hidden_dim = weights["input_projection.bias"].shape[0]
    history_max = 16
    conv1_history = np.zeros((4, hidden_dim, history_max), dtype=np.float32)
    conv2_history = np.zeros((4, hidden_dim, history_max), dtype=np.float32)
    predictions = []
    for sample in np.asarray(sequence, dtype=np.float32):
        base = sample.astype(np.float32)
        features = linear(weights["input_projection.weight"][:, :, 0], weights["input_projection.bias"], base)
        for block_index in range(4):
            dilation = 1 << block_index
            history_len = 2 * dilation
            residual = features.copy()
            conv1 = conv_step(
                weights[f"blocks.{block_index}.conv1.weight"],
                weights[f"blocks.{block_index}.conv1.bias"],
                conv1_history[block_index, :, :history_len],
                features,
                dilation,
            )
            conv1 = np.maximum(conv1, 0.0).astype(np.float32)
            update_history(conv1_history[block_index], features, history_len)

            conv2 = conv_step(
                weights[f"blocks.{block_index}.conv2.weight"],
                weights[f"blocks.{block_index}.conv2.bias"],
                conv2_history[block_index, :, :history_len],
                conv1,
                dilation,
            )
            conv2 = np.maximum(conv2, 0.0).astype(np.float32)
            update_history(conv2_history[block_index], conv1, history_len)
            features = (conv2 + residual).astype(np.float32)

        latent = linear(
            weights["attachment_encoder.latent_projection.weight"][:, :, 0],
            weights["attachment_encoder.latent_projection.bias"],
            features,
        )
        gate = sigmoid(linear(weights["feature_gate.weight"][:, :, 0], weights["feature_gate.bias"], latent))
        shift = linear(weights["feature_shift.weight"][:, :, 0], weights["feature_shift.bias"], latent)
        conditioned = (features * (1.0 + gate) + shift).astype(np.float32)
        residual = linear(
            weights["residual_projection.weight"][:, :, 0],
            weights["residual_projection.bias"],
            conditioned,
        )
        predictions.append((base + residual).astype(np.float32))
    return np.asarray(predictions, dtype=np.float32)


def c_float(value: float) -> str:
    return f"{np.float32(value).item():.9g}f"


def nested_initializer(array: np.ndarray, indent: int = 0) -> str:
    values = np.asarray(array, dtype=np.float32)
    pad = " " * indent
    if values.ndim == 1:
        return "{" + ", ".join(c_float(float(item)) for item in values) + "}"
    lines = [pad + "{"]
    for index, child in enumerate(values):
        suffix = "," if index < values.shape[0] - 1 else ""
        lines.append(nested_initializer(child, indent + 2) + suffix)
    lines.append(pad + "}")
    return "\n".join(lines)


def write_header(path: Path) -> None:
    text = """#ifndef STREAMING_TCN_WEIGHTS_H
#define STREAMING_TCN_WEIGHTS_H

#include <stdint.h>

#define STCN_INPUT_DIM 6U
#define STCN_HIDDEN_DIM 64U
#define STCN_OUTPUT_DIM 6U
#define STCN_LATENT_DIM 8U
#define STCN_NUM_BLOCKS 4U
#define STCN_KERNEL_SIZE 3U
#define STCN_HISTORY_MAX 16U

extern const float g_stcn_input_projection_weight[STCN_HIDDEN_DIM][STCN_INPUT_DIM];
extern const float g_stcn_input_projection_bias[STCN_HIDDEN_DIM];
extern const float g_stcn_block_conv1_weight[STCN_NUM_BLOCKS][STCN_HIDDEN_DIM][STCN_HIDDEN_DIM][STCN_KERNEL_SIZE];
extern const float g_stcn_block_conv1_bias[STCN_NUM_BLOCKS][STCN_HIDDEN_DIM];
extern const float g_stcn_block_conv2_weight[STCN_NUM_BLOCKS][STCN_HIDDEN_DIM][STCN_HIDDEN_DIM][STCN_KERNEL_SIZE];
extern const float g_stcn_block_conv2_bias[STCN_NUM_BLOCKS][STCN_HIDDEN_DIM];
extern const float g_stcn_latent_weight[STCN_LATENT_DIM][STCN_HIDDEN_DIM];
extern const float g_stcn_latent_bias[STCN_LATENT_DIM];
extern const float g_stcn_gate_weight[STCN_HIDDEN_DIM][STCN_LATENT_DIM];
extern const float g_stcn_gate_bias[STCN_HIDDEN_DIM];
extern const float g_stcn_shift_weight[STCN_HIDDEN_DIM][STCN_LATENT_DIM];
extern const float g_stcn_shift_bias[STCN_HIDDEN_DIM];
extern const float g_stcn_residual_weight[STCN_OUTPUT_DIM][STCN_HIDDEN_DIM];
extern const float g_stcn_residual_bias[STCN_OUTPUT_DIM];

#endif
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_source(path: Path, weights: Dict[str, np.ndarray], metadata: Dict[str, object]) -> None:
    arrays = {
        "g_stcn_input_projection_weight": weights["input_projection.weight"][:, :, 0],
        "g_stcn_input_projection_bias": weights["input_projection.bias"],
        "g_stcn_block_conv1_weight": np.stack(
            [weights[f"blocks.{index}.conv1.weight"] for index in range(4)], axis=0
        ),
        "g_stcn_block_conv1_bias": np.stack(
            [weights[f"blocks.{index}.conv1.bias"] for index in range(4)], axis=0
        ),
        "g_stcn_block_conv2_weight": np.stack(
            [weights[f"blocks.{index}.conv2.weight"] for index in range(4)], axis=0
        ),
        "g_stcn_block_conv2_bias": np.stack(
            [weights[f"blocks.{index}.conv2.bias"] for index in range(4)], axis=0
        ),
        "g_stcn_latent_weight": weights["attachment_encoder.latent_projection.weight"][:, :, 0],
        "g_stcn_latent_bias": weights["attachment_encoder.latent_projection.bias"],
        "g_stcn_gate_weight": weights["feature_gate.weight"][:, :, 0],
        "g_stcn_gate_bias": weights["feature_gate.bias"],
        "g_stcn_shift_weight": weights["feature_shift.weight"][:, :, 0],
        "g_stcn_shift_bias": weights["feature_shift.bias"],
        "g_stcn_residual_weight": weights["residual_projection.weight"][:, :, 0],
        "g_stcn_residual_bias": weights["residual_projection.bias"],
    }
    declarations = {
        "g_stcn_input_projection_weight": "[STCN_HIDDEN_DIM][STCN_INPUT_DIM]",
        "g_stcn_input_projection_bias": "[STCN_HIDDEN_DIM]",
        "g_stcn_block_conv1_weight": "[STCN_NUM_BLOCKS][STCN_HIDDEN_DIM][STCN_HIDDEN_DIM][STCN_KERNEL_SIZE]",
        "g_stcn_block_conv1_bias": "[STCN_NUM_BLOCKS][STCN_HIDDEN_DIM]",
        "g_stcn_block_conv2_weight": "[STCN_NUM_BLOCKS][STCN_HIDDEN_DIM][STCN_HIDDEN_DIM][STCN_KERNEL_SIZE]",
        "g_stcn_block_conv2_bias": "[STCN_NUM_BLOCKS][STCN_HIDDEN_DIM]",
        "g_stcn_latent_weight": "[STCN_LATENT_DIM][STCN_HIDDEN_DIM]",
        "g_stcn_latent_bias": "[STCN_LATENT_DIM]",
        "g_stcn_gate_weight": "[STCN_HIDDEN_DIM][STCN_LATENT_DIM]",
        "g_stcn_gate_bias": "[STCN_HIDDEN_DIM]",
        "g_stcn_shift_weight": "[STCN_HIDDEN_DIM][STCN_LATENT_DIM]",
        "g_stcn_shift_bias": "[STCN_HIDDEN_DIM]",
        "g_stcn_residual_weight": "[STCN_OUTPUT_DIM][STCN_HIDDEN_DIM]",
        "g_stcn_residual_bias": "[STCN_OUTPUT_DIM]",
    }
    lines = [
        '#include "streaming_tcn_weights.h"',
        "",
        "/* Generated by tools/deploy_stm32/export_streaming_tcn_c.py. */",
        f"/* Source checkpoint: {metadata['checkpoint']} */",
        f"/* PyTorch-vs-NumPy streaming max_abs_error: {metadata['max_abs_error']:.9g} */",
        "",
    ]
    for name, array in arrays.items():
        lines.append(f"const float {name}{declarations[name]} =")
        lines.append(nested_initializer(array, 0) + ";")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def validate_numpy_stream(torch, model, weights: Dict[str, np.ndarray], vectors_path: Path) -> Dict[str, object]:
    vectors = np.load(vectors_path)
    normalized_windows = np.asarray(vectors["normalized_windows"], dtype=np.float32)
    window = normalized_windows[0]
    numpy_predictions = run_numpy_stream(weights, window)
    stream_state = None
    torch_predictions = []
    with torch.no_grad():
        for sample in window:
            output = model.forward_step(torch.from_numpy(sample).reshape(1, 6), stream_state=stream_state)
            stream_state = output["stream_state"]
            torch_predictions.append(output["prediction_step"].detach().cpu().numpy()[0])
    torch_predictions = np.asarray(torch_predictions, dtype=np.float32)
    delta = numpy_predictions - torch_predictions
    return {
        "vectors_path": str(vectors_path.resolve()),
        "checked_steps": int(window.shape[0]),
        "max_abs_error": float(np.max(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(np.square(delta)))),
        "last_step_max_abs_error": float(np.max(np.abs(delta[-1]))),
    }


def main() -> None:
    repo_root = repo_root_from_file()
    parser = argparse.ArgumentParser(description="Export the causal TCN checkpoint as STM32 streaming C weights.")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--vectors",
        type=Path,
        default=repo_root / DEFAULT_OUTPUT_ROOT / "test_vectors" / "stm32_golden_vectors.npz",
    )
    parser.add_argument(
        "--firmware-lib",
        type=Path,
        default=repo_root / "firmware" / "platformio_nucleo_h723zg" / "lib" / "streaming_tcn",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=repo_root / DEFAULT_OUTPUT_ROOT / "streaming_tcn_export_report.json",
    )
    args = parser.parse_args()

    checkpoint = resolve_checkpoint(repo_root, args.checkpoint)
    torch, _, config, model = load_checkpoint_and_model(checkpoint)
    weights = tensor_dict(model)
    validation = validate_numpy_stream(torch, model, weights, args.vectors)
    metadata = {
        "checkpoint": str(checkpoint),
        "model_name": config.get("model_name"),
        "model_config": config.get("model", {}),
        **validation,
    }
    write_header(args.firmware_lib / "include" / "streaming_tcn_weights.h")
    write_source(args.firmware_lib / "src" / "streaming_tcn_weights.c", weights, metadata)
    save_json(metadata, args.report)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] Streaming TCN export failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
