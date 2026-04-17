from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from project.data.window_dataset import WindowedPairDataset, windowed_pair_collate
from project.evaluation.visualize_predictions import plot_prediction_bundle
from project.models import build_model
from project.training.engine import collect_predictions
from project.training.metrics import save_metric_summary, summarize_window_metrics
from project.utils.io import ensure_dir, load_json, resolve_path, save_json
from project.utils.logger import get_logger
from project.utils.torch_compat import TORCH_AVAILABLE, require_torch, torch


DEFAULT_BASELINE_MODELS = ("identity", "lowpass")
DEFAULT_DELTA_METRICS = (
    "rmse_mean",
    "pearson_mean",
    "psd_distance_mean",
    "acc_norm_rmse",
    "gyr_norm_rmse",
    "hf_ratio_improvement_mean",
)


def _get_baseline_model_names(config: Dict[str, Any]) -> List[str]:
    evaluation_config = dict(config.get("evaluation", {}))
    requested = evaluation_config.get("baseline_models", list(DEFAULT_BASELINE_MODELS))
    primary_model_name = str(config["model_name"]).lower()

    names: List[str] = []
    seen = {primary_model_name}
    for model_name in requested:
        normalized_name = str(model_name).lower()
        if normalized_name in seen:
            continue
        names.append(normalized_name)
        seen.add(normalized_name)
    return names


def _build_comparison_frame(
    overall_by_model: Dict[str, Dict[str, float]],
    primary_model_name: str,
    model_roles: Dict[str, str] | None = None,
) -> pd.DataFrame:
    rows = []
    model_roles = model_roles or {}
    for model_name, metrics in overall_by_model.items():
        row = {
            "model_name": model_name,
            "model_role": model_roles.get(
                model_name,
                "trained" if model_name == primary_model_name else "baseline",
            ),
        }
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def _save_comparison_summary(
    comparison_frame: pd.DataFrame,
    metrics_root: Path,
) -> Path:
    comparison_csv_path = metrics_root / "model_comparison.csv"
    comparison_json_path = metrics_root / "model_comparison.json"
    comparison_frame.to_csv(comparison_csv_path, index=False)
    save_json({"models": comparison_frame.to_dict(orient="records")}, comparison_json_path)
    return comparison_csv_path


def _load_trained_comparison_specs(config: Dict[str, Any]) -> List[Dict[str, str]]:
    evaluation_config = dict(config.get("evaluation", {}))
    raw_specs = evaluation_config.get("trained_model_checkpoints", [])
    config_dir = Path(config["config_dir"])
    specs: List[Dict[str, str]] = []
    seen_labels = {str(config["model_name"]).lower()}
    for index, spec in enumerate(raw_specs):
        if isinstance(spec, str):
            label = Path(spec).stem.lower()
            checkpoint_path = resolve_path(spec, config_dir)
        elif isinstance(spec, dict):
            checkpoint_value = spec.get("checkpoint")
            if checkpoint_value is None:
                raise ValueError("Each `trained_model_checkpoints` item must define `checkpoint`.")
            label = str(spec.get("label") or Path(str(checkpoint_value)).stem).lower()
            checkpoint_path = resolve_path(str(checkpoint_value), config_dir)
        else:
            raise ValueError(
                f"Unsupported trained model checkpoint spec at index {index}: {type(spec).__name__}"
            )
        if label in seen_labels:
            raise ValueError(f"Duplicate trained model comparison label: {label}")
        seen_labels.add(label)
        specs.append({"label": label, "checkpoint_path": str(checkpoint_path)})
    return specs


def _build_group_delta_frame(
    candidate_frame: pd.DataFrame,
    reference_frame: pd.DataFrame,
    group_column: str,
    candidate_name: str,
    reference_name: str,
) -> pd.DataFrame:
    metric_columns = [
        metric_name
        for metric_name in DEFAULT_DELTA_METRICS
        if metric_name in candidate_frame.columns and metric_name in reference_frame.columns
    ]
    merged = candidate_frame[[group_column] + metric_columns].merge(
        reference_frame[[group_column] + metric_columns],
        on=group_column,
        how="inner",
        suffixes=(f"_{candidate_name}", f"_{reference_name}"),
    )
    for metric_name in metric_columns:
        merged[f"{metric_name}_delta_{candidate_name}_minus_{reference_name}"] = (
            merged[f"{metric_name}_{candidate_name}"] - merged[f"{metric_name}_{reference_name}"]
        )
    return merged.sort_values(group_column).reset_index(drop=True)


def _save_motion_delta_summary(
    summary_by_model: Dict[str, Dict[str, Any]],
    metrics_root: Path,
    primary_model_name: str,
    reference_model_name: str,
) -> Path | None:
    if primary_model_name not in summary_by_model or reference_model_name not in summary_by_model:
        return None
    delta_frame = _build_group_delta_frame(
        candidate_frame=summary_by_model[primary_model_name]["per_motion"],
        reference_frame=summary_by_model[reference_model_name]["per_motion"],
        group_column="motion_name",
        candidate_name=primary_model_name,
        reference_name=reference_model_name,
    )
    path = metrics_root / f"per_motion_delta_{primary_model_name}_vs_{reference_model_name}.csv"
    delta_frame.to_csv(path, index=False)
    return path


if TORCH_AVAILABLE:  # pragma: no branch

    def _resolve_device(device_name: str):
        if device_name == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")


    def _evaluate_model_outputs(
        config: Dict[str, Any],
        output_root: Path,
        model_name: str,
        predictions: np.ndarray,
        targets: np.ndarray,
        inputs: np.ndarray,
        metadata: List[Dict[str, Any]],
        write_figures: bool,
    ) -> Dict[str, Any]:
        summary = summarize_window_metrics(
            predictions=predictions,
            targets=targets,
            inputs=inputs,
            metadata=metadata,
            channels=list(config["target_channels"]),
            sampling_frequency=float(config["sampling_frequency"]),
        )
        save_metric_summary(summary, output_root / "metrics" / model_name)
        if write_figures:
            _visualize_examples(config, output_root / "figures", predictions, targets, inputs, metadata)
        return summary


    def _collect_model_outputs(model, dataloader, device) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        gathered = collect_predictions(model, dataloader, device)
        return (
            gathered["predictions"].numpy(),
            gathered["targets"].numpy(),
            gathered["inputs"].numpy(),
            gathered["metadata"],
        )


    def _evaluate_trained_checkpoint(
        checkpoint_path: Path,
        evaluation_config: Dict[str, Any],
        output_root: Path,
        model_label: str,
        dataloader,
        device,
        write_figures: bool,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model_config = dict(checkpoint.get("config") or {})
        if not model_config:
            raise ValueError(f"Checkpoint at {checkpoint_path} does not contain its training config.")
        model = build_model(
            model_name=model_config["model_name"],
            input_dim=len(model_config["input_channels"]),
            output_dim=len(model_config["target_channels"]),
            model_config=dict(model_config.get("model", {})),
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        predictions, targets, inputs, metadata = _collect_model_outputs(model, dataloader, device)
        summary = _evaluate_model_outputs(
            config=evaluation_config,
            output_root=output_root,
            model_name=model_label,
            predictions=predictions,
            targets=targets,
            inputs=inputs,
            metadata=metadata,
            write_figures=write_figures,
        )
        return summary, metadata


    def evaluate_checkpoint(config: Dict[str, Any], processed_root: Path, output_root: Path, checkpoint_path: Path) -> Path:
        """Evaluate the trained model, baselines, and optional trained comparison checkpoints."""
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
        primary_model_name = str(config["model_name"]).lower()
        input_dim = len(config["input_channels"])
        output_dim = len(config["target_channels"])

        overall_by_model: Dict[str, Dict[str, float]] = OrderedDict()
        summary_by_model: Dict[str, Dict[str, Any]] = OrderedDict()
        model_roles: Dict[str, str] = {primary_model_name: "trained"}

        primary_summary, metadata = _evaluate_trained_checkpoint(
            checkpoint_path=checkpoint_path,
            evaluation_config=config,
            output_root=output_root,
            model_label=primary_model_name,
            dataloader=dataloader,
            device=device,
            write_figures=True,
        )
        overall_by_model[primary_model_name] = primary_summary["overall"]
        summary_by_model[primary_model_name] = primary_summary

        for trained_spec in _load_trained_comparison_specs(config):
            comparison_label = trained_spec["label"]
            comparison_path = Path(trained_spec["checkpoint_path"])
            comparison_summary, _ = _evaluate_trained_checkpoint(
                checkpoint_path=comparison_path,
                evaluation_config=config,
                output_root=output_root,
                model_label=comparison_label,
                dataloader=dataloader,
                device=device,
                write_figures=False,
            )
            overall_by_model[comparison_label] = comparison_summary["overall"]
            summary_by_model[comparison_label] = comparison_summary
            model_roles[comparison_label] = "trained_comparison"

        for baseline_model_name in _get_baseline_model_names(config):
            if baseline_model_name == "linear":
                logger.warning(
                    "Skipping baseline '%s' because it is untrained at evaluation time and would use random weights.",
                    baseline_model_name,
                )
                continue
            if baseline_model_name in {"identity", "lowpass"} and input_dim != output_dim:
                logger.warning(
                    "Skipping baseline '%s' because it requires identical input and target channel dimensions.",
                    baseline_model_name,
                )
                continue

            baseline_model = build_model(
                model_name=baseline_model_name,
                input_dim=input_dim,
                output_dim=output_dim,
                model_config=dict(config.get("model", {})),
            ).to(device)
            predictions, targets, inputs, baseline_metadata = _collect_model_outputs(baseline_model, dataloader, device)
            baseline_summary = _evaluate_model_outputs(
                config=config,
                output_root=output_root,
                model_name=baseline_model_name,
                predictions=predictions,
                targets=targets,
                inputs=inputs,
                metadata=baseline_metadata,
                write_figures=False,
            )
            overall_by_model[baseline_model_name] = baseline_summary["overall"]
            summary_by_model[baseline_model_name] = baseline_summary
            model_roles[baseline_model_name] = "baseline"

        comparison_frame = _build_comparison_frame(
            overall_by_model,
            primary_model_name,
            model_roles=model_roles,
        )
        comparison_path = _save_comparison_summary(comparison_frame, output_root / "metrics")
        delta_path = _save_motion_delta_summary(
            summary_by_model=summary_by_model,
            metrics_root=output_root / "metrics",
            primary_model_name=primary_model_name,
            reference_model_name="lowpass",
        )
        logger.info(
            "Evaluation finished for %d windows. Compared %d models.",
            len(metadata),
            len(comparison_frame),
        )
        if delta_path is not None:
            logger.info("Saved per-motion delta table to %s.", delta_path)
        return comparison_path


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
