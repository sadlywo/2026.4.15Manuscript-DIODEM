from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from deploy_common import (
    DEFAULT_CHECKPOINT,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PROCESSED_ROOT,
    INPUT_CHANNELS,
    PREDICTION_CHANNELS,
    denormalize_outputs,
    load_checkpoint_and_model,
    load_json,
    load_test_windows,
    normalize_inputs,
    repo_root_from_file,
)
from export_onnx import build_export_wrapper


def _write_csv(path: Path, columns: Sequence[str], values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_index", *columns])
        for index, row in enumerate(np.asarray(values, dtype=np.float32)):
            writer.writerow([index, *[f"{float(item):.9g}" for item in row]])


def prepare_vectors(
    checkpoint: Path,
    processed_root: Path,
    output_dir: Path,
    split: str = "test",
    num_windows: int = 4,
    device: str = "cpu",
) -> dict:
    torch, _, config, model = load_checkpoint_and_model(checkpoint, device_name=device)
    stats = load_json(Path(processed_root) / "normalization_stats.json")
    raw_windows = load_test_windows(processed_root, split=split, max_windows=num_windows)
    normalized_windows = normalize_inputs(raw_windows, stats)
    wrapper = build_export_wrapper(torch, model, output_mode="last_step")
    with torch.no_grad():
        normalized_reference = wrapper(torch.from_numpy(normalized_windows).to(device)).detach().cpu().numpy()
    physical_reference = denormalize_outputs(normalized_reference, stats)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_last_samples = raw_windows[:, -1, :]
    normalized_last_samples = normalized_windows[:, -1, :]
    _write_csv(output_dir / "stm32_input_last_samples.csv", config.get("input_channels", INPUT_CHANNELS), raw_last_samples)
    _write_csv(output_dir / "stm32_input_last_samples_normalized.csv", config.get("input_channels", INPUT_CHANNELS), normalized_last_samples)
    _write_csv(output_dir / "pc_fp32_reference_last_step.csv", PREDICTION_CHANNELS, physical_reference)
    np.savez_compressed(
        output_dir / "stm32_golden_vectors.npz",
        raw_windows=raw_windows.astype(np.float32),
        normalized_windows=normalized_windows.astype(np.float32),
        normalized_reference=normalized_reference.astype(np.float32),
        physical_reference=physical_reference.astype(np.float32),
    )
    return {
        "output_dir": str(output_dir.resolve()),
        "split": split,
        "num_windows": int(raw_windows.shape[0]),
        "input_shape": list(raw_windows.shape),
        "reference_shape": list(physical_reference.shape),
    }


def main() -> None:
    repo_root = repo_root_from_file()
    parser = argparse.ArgumentParser(description="Create STM32 input vectors and PC FP32 golden references.")
    parser.add_argument("--checkpoint", type=Path, default=repo_root / DEFAULT_CHECKPOINT)
    parser.add_argument("--processed-root", type=Path, default=repo_root / DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--output-dir", type=Path, default=repo_root / DEFAULT_OUTPUT_ROOT / "test_vectors")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--num-windows", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    try:
        summary = prepare_vectors(
            checkpoint=args.checkpoint,
            processed_root=args.processed_root,
            output_dir=args.output_dir,
            split=args.split,
            num_windows=args.num_windows,
            device=args.device,
        )
    except Exception as exc:
        print(f"[ERROR] STM32 vector preparation failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(summary)


if __name__ == "__main__":
    main()
