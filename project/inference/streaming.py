from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

from project.models import build_model
from project.utils.torch_compat import TORCH_AVAILABLE, require_torch, torch


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


if TORCH_AVAILABLE:  # pragma: no branch

    class StreamingCompensator:
        """Chunked or step-wise causal inference wrapper around a trained checkpoint."""

        def __init__(
            self,
            model,
            normalization: str,
            normalization_stats: Dict[str, Any],
            input_dim: int,
            device,
        ) -> None:
            self.model = model.to(device).eval()
            self.normalization = str(normalization)
            self.stats = dict(normalization_stats or {"mode": "none"})
            self.input_dim = int(input_dim)
            self.device = device
            self.stream_state = self.model.init_stream_state(batch_size=1, device=device, dtype=torch.float32)

        @classmethod
        def from_checkpoint(cls, checkpoint_path: Path | str, device_name: str = "cpu") -> "StreamingCompensator":
            require_torch()
            checkpoint_path = Path(checkpoint_path).resolve()
            checkpoint = torch.load(checkpoint_path, map_location=device_name)
            config = dict(checkpoint.get("config") or {})
            if not config:
                raise ValueError(f"Checkpoint at {checkpoint_path} does not contain its training config.")

            model_config = {
                **dict(config.get("model", {})),
                "sampling_frequency": float(config.get("sampling_frequency", 40.0)),
            }
            model = build_model(
                model_name=str(config["model_name"]),
                input_dim=len(config["input_channels"]),
                output_dim=len(config["target_channels"]),
                model_config=model_config,
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()

            repo_root = Path(config["repo_root"])
            processed_root = (repo_root / config["processed_root"]).resolve()
            stats = _load_json(processed_root / "normalization_stats.json")
            return cls(
                model=model,
                normalization=str(config.get("normalization", "none")),
                normalization_stats=stats,
                input_dim=len(config["input_channels"]),
                device=torch.device(device_name),
            )

        def reset(self, batch_size: int = 1) -> None:
            self.stream_state = self.model.init_stream_state(
                batch_size=batch_size,
                device=self.device,
                dtype=torch.float32,
            )

        def _normalize_inputs(self, values: np.ndarray) -> np.ndarray:
            array = np.asarray(values, dtype=np.float32)
            if self.normalization == "none" or self.stats.get("mode") == "none":
                return array.astype(np.float32)
            mean = np.asarray(self.stats["input_mean"], dtype=np.float32)
            std = np.asarray(self.stats["input_std"], dtype=np.float32)
            return ((array - mean) / std).astype(np.float32)

        def _denormalize_outputs(self, values: np.ndarray) -> np.ndarray:
            array = np.asarray(values, dtype=np.float32)
            if self.normalization == "none" or self.stats.get("mode") == "none":
                return array.astype(np.float32)
            mean = np.asarray(self.stats["target_mean"], dtype=np.float32)
            std = np.asarray(self.stats["target_std"], dtype=np.float32)
            return (array * std + mean).astype(np.float32)

        @torch.no_grad()
        def push(self, sample: np.ndarray) -> Dict[str, Any]:
            sample_array = np.asarray(sample, dtype=np.float32)
            if sample_array.shape != (self.input_dim,):
                raise ValueError(f"Expected single sample shape ({self.input_dim},), got {sample_array.shape}")

            normalized = self._normalize_inputs(sample_array)
            inputs = torch.from_numpy(normalized).to(self.device).unsqueeze(0)
            start = time.perf_counter()
            outputs = self.model.forward_step(inputs, stream_state=self.stream_state)
            latency_ms = float((time.perf_counter() - start) * 1000.0)
            self.stream_state = outputs["stream_state"]
            prediction = outputs["prediction_step"].detach().cpu().numpy()[0]
            prediction = self._denormalize_outputs(prediction)
            return {"prediction": prediction, "latency_ms": latency_ms}

        @torch.no_grad()
        def process_sequence(self, sequence: np.ndarray, reset: bool = True) -> Dict[str, Any]:
            values = np.asarray(sequence, dtype=np.float32)
            if values.ndim != 2 or values.shape[1] != self.input_dim:
                raise ValueError(f"Expected `[T, {self.input_dim}]` sequence, got {values.shape}")
            if reset:
                self.reset(batch_size=1)
            predictions = []
            latencies = []
            for step in values:
                result = self.push(step)
                predictions.append(result["prediction"])
                latencies.append(result["latency_ms"])
            prediction_array = np.asarray(predictions, dtype=np.float32)
            latency_array = np.asarray(latencies, dtype=np.float32)
            return {
                "predictions": prediction_array,
                "latency_ms": latency_array,
                "latency_mean_ms": float(latency_array.mean()) if len(latency_array) else 0.0,
                "latency_p95_ms": float(np.percentile(latency_array, 95)) if len(latency_array) else 0.0,
            }

else:

    class StreamingCompensator:  # pragma: no cover - runtime safeguard only
        def __init__(self, *args, **kwargs):
            require_torch()
