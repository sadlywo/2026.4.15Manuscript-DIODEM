from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from project.inference import StreamingCompensator
from project.models import build_model
from project.utils.torch_compat import require_torch, torch


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def _normalize_inputs(values: np.ndarray, stats: dict, normalization: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if normalization == "none" or stats.get("mode") == "none":
        return values.astype(np.float32)
    mean = np.asarray(stats["input_mean"], dtype=np.float32)
    std = np.asarray(stats["input_std"], dtype=np.float32)
    return ((values - mean) / std).astype(np.float32)


def _denormalize_targets(values: np.ndarray, stats: dict, normalization: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if normalization == "none" or stats.get("mode") == "none":
        return values.astype(np.float32)
    mean = np.asarray(stats["target_mean"], dtype=np.float32)
    std = np.asarray(stats["target_std"], dtype=np.float32)
    return (values * std + mean).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and benchmark streaming TCN inference on cached windows.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to a trained checkpoint.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device for inference, e.g. cpu or cuda.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--max-windows", type=int, default=256, help="Maximum number of cached windows to evaluate.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path. Defaults next to the checkpoint as streaming_eval_<split>.json",
    )
    args = parser.parse_args()

    require_torch()
    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    config = dict(checkpoint.get("config") or {})
    if not config:
        raise ValueError(f"Checkpoint at {checkpoint_path} does not contain its training config.")

    repo_root = Path(config["repo_root"])
    processed_root = (repo_root / config["processed_root"]).resolve()
    stats = _load_json(processed_root / "normalization_stats.json")
    bundle = _load_pickle(processed_root / f"{args.split}_samples.pkl")
    raw_inputs = np.asarray(bundle["inputs"], dtype=np.float32)
    num_windows = min(int(args.max_windows), int(raw_inputs.shape[0]))

    model_config = {
        **dict(config.get("model", {})),
        "sampling_frequency": float(config.get("sampling_frequency", 40.0)),
    }
    model = build_model(
        model_name=str(config["model_name"]),
        input_dim=len(config["input_channels"]),
        output_dim=len(config["target_channels"]),
        model_config=model_config,
    ).to(args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    streaming = StreamingCompensator.from_checkpoint(checkpoint_path, device_name=args.device)

    max_abs_diffs = []
    rmse_diffs = []
    latency_means = []
    latency_p95s = []

    with torch.no_grad():
        for window in raw_inputs[:num_windows]:
            normalized_window = _normalize_inputs(window, stats, str(config.get("normalization", "none")))
            batch_inputs = torch.from_numpy(normalized_window).unsqueeze(0).to(args.device)
            offline_outputs = model(batch_inputs)
            if isinstance(offline_outputs, dict):
                offline_predictions = offline_outputs["predictions"]
            else:
                offline_predictions = offline_outputs
            offline_predictions = offline_predictions.squeeze(0).detach().cpu().numpy()
            offline_predictions = _denormalize_targets(
                offline_predictions,
                stats,
                str(config.get("normalization", "none")),
            )

            streaming_outputs = streaming.process_sequence(window, reset=True)
            streaming_predictions = streaming_outputs["predictions"]
            delta = streaming_predictions - offline_predictions
            max_abs_diffs.append(float(np.max(np.abs(delta))))
            rmse_diffs.append(float(np.sqrt(np.mean(np.square(delta)))))
            latency_means.append(float(streaming_outputs["latency_mean_ms"]))
            latency_p95s.append(float(streaming_outputs["latency_p95_ms"]))

    result = {
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "num_windows": int(num_windows),
        "model_name": str(config["model_name"]),
        "receptive_field": int(getattr(model, "receptive_field", 0)),
        "window_size": int(config.get("window_size", 0)),
        "streaming_vs_offline_max_abs_diff_mean": float(np.mean(max_abs_diffs)) if max_abs_diffs else 0.0,
        "streaming_vs_offline_rmse_mean": float(np.mean(rmse_diffs)) if rmse_diffs else 0.0,
        "streaming_latency_mean_ms": float(np.mean(latency_means)) if latency_means else 0.0,
        "streaming_latency_p95_ms": float(np.mean(latency_p95s)) if latency_p95s else 0.0,
    }

    output_path = args.output
    if output_path is None:
        output_path = checkpoint_path.parent.parent / f"streaming_eval_{args.split}.json"
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"Saved streaming evaluation summary to {output_path}")


if __name__ == "__main__":
    main()
