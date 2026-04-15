from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from project.data.window_dataset import WindowedPairDataset, windowed_pair_collate
from project.evaluation.visualize_predictions import plot_prediction_bundle
from project.models import build_model
from project.training.engine import collect_predictions
from project.training.metrics import save_metric_summary, summarize_window_metrics
from project.utils.io import ensure_dir, load_json
from project.utils.logger import get_logger
from project.utils.torch_compat import TORCH_AVAILABLE, require_torch, torch


if TORCH_AVAILABLE:  # pragma: no branch

    def _resolve_device(device_name: str):
        if device_name == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")


    def evaluate_checkpoint(config: Dict[str, Any], processed_root: Path, output_root: Path, checkpoint_path: Path) -> Path:
        """Evaluate a saved checkpoint on the test split and write reports."""
        require_torch()
        logger = get_logger("diomed-eval", output_root / "eval.log")
        ensure_dir(output_root)
        ensure_dir(output_root / "figures")
        ensure_dir(output_root / "metrics")

        stats = load_json(processed_root / "normalization_stats.json")
        dataset = WindowedPairDataset(
            processed_root / "test_samples.pkl",
            normalization=config.get("normalization", "none"),
            normalization_stats=stats,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=int(config["batch_size"]),
            shuffle=False,
            num_workers=int(config.get("num_workers", 0)),
            collate_fn=windowed_pair_collate,
        )
        device = _resolve_device(str(config["device"]).lower())
        model = build_model(
            model_name=config["model_name"],
            input_dim=len(config["input_channels"]),
            output_dim=len(config["target_channels"]),
            model_config=dict(config.get("model", {})),
        ).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        gathered = collect_predictions(model, dataloader, device)
        predictions = gathered["predictions"].numpy()
        targets = gathered["targets"].numpy()
        inputs = gathered["inputs"].numpy()
        metadata = gathered["metadata"]

        summary = summarize_window_metrics(
            predictions=predictions,
            targets=targets,
            inputs=inputs,
            metadata=metadata,
            channels=list(config["target_channels"]),
            sampling_frequency=float(config["sampling_frequency"]),
        )
        save_metric_summary(summary, output_root / "metrics")
        _visualize_examples(config, output_root / "figures", predictions, targets, inputs, metadata)
        logger.info("Evaluation finished for %d windows.", len(metadata))
        return output_root / "metrics" / "overall_metrics.json"


    def _visualize_examples(
        config: Dict[str, Any],
        figure_dir: Path,
        predictions: np.ndarray,
        targets: np.ndarray,
        inputs: np.ndarray,
        metadata: List[Dict[str, Any]],
    ) -> None:
        max_visualizations = int(config["evaluation"]["max_visualizations"])
        ranked_indices = list(range(len(metadata)))
        ranked_indices.sort(
            key=lambda idx: (not bool(metadata[idx].get("is_anomaly_case", False)), metadata[idx]["motion_name"])
        )
        for idx in ranked_indices[:max_visualizations]:
            meta = metadata[idx]
            stem = (
                f"kc_{meta['kc_type']}_{meta['experiment_id']}_{meta['motion_index']}_"
                f"{meta['segment_id']}_{meta['start_idx']:05d}"
            )
            plot_prediction_bundle(
                sample={
                    "inputs": inputs[idx],
                    "targets": targets[idx],
                    "predictions": predictions[idx],
                },
                channels=list(config["target_channels"]),
                sampling_frequency=float(config["sampling_frequency"]),
                output_dir=figure_dir,
                stem=stem,
            )

else:

    def evaluate_checkpoint(*args, **kwargs):  # pragma: no cover - runtime safeguard only
        require_torch()

