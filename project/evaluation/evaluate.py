from __future__ import annotations

import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from project.data.window_dataset import WindowedPairDataset, windowed_pair_collate
from project.evaluation.visualize_predictions import plot_prediction_bundle
from project.models import build_model
from project.training.engine import collect_predictions
from project.training.metrics import save_metric_summary, summarize_window_metrics
from project.utils.io import ensure_dir, inverse_zscore, load_csv_table, load_json, resolve_path, save_json
from project.utils.logger import get_logger
from project.utils.torch_compat import TORCH_AVAILABLE, require_torch, torch


DEFAULT_BASELINE_MODELS = ("identity", "lowpass", "butterworth", "savgol", "wiener")
DEFAULT_DELTA_METRICS = (
    "rmse_mean",
    "acc_channel_rmse_mean",
    "gyr_channel_rmse_mean",
    "pearson_mean",
    "psd_distance_mean",
    "acc_norm_rmse",
    "gyr_norm_rmse",
    "hf_ratio_improvement_mean",
)
LOWER_IS_BETTER_METRICS = {
    "rmse_mean",
    "acc_channel_rmse_mean",
    "gyr_channel_rmse_mean",
    "psd_distance_mean",
    "acc_norm_rmse",
    "gyr_norm_rmse",
}
HIGHER_IS_BETTER_METRICS = {
    "pearson_mean",
    "hf_ratio_improvement_mean",
}


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


    def _build_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
        return {
            **dict(config.get("model", {})),
            "sampling_frequency": float(config["sampling_frequency"]),
        }


    def _denormalize_bundle(
        predictions: np.ndarray,
        targets: np.ndarray,
        inputs: np.ndarray,
        stats: Dict[str, Any],
        normalization: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if normalization == "none" or stats.get("mode") == "none":
            return (
                predictions.astype(np.float32),
                targets.astype(np.float32),
                inputs.astype(np.float32),
            )

        input_mean = np.asarray(stats["input_mean"], dtype=np.float32)
        input_std = np.asarray(stats["input_std"], dtype=np.float32)
        target_mean = np.asarray(stats["target_mean"], dtype=np.float32)
        target_std = np.asarray(stats["target_std"], dtype=np.float32)
        return (
            inverse_zscore(np.asarray(predictions, dtype=np.float32), target_mean, target_std).astype(np.float32),
            inverse_zscore(np.asarray(targets, dtype=np.float32), target_mean, target_std).astype(np.float32),
            inverse_zscore(np.asarray(inputs, dtype=np.float32), input_mean, input_std).astype(np.float32),
        )


    def _evaluate_model_outputs(
        config: Dict[str, Any],
        output_root: Path,
        model_name: str,
        predictions: np.ndarray,
        targets: np.ndarray,
        inputs: np.ndarray,
        metadata: List[Dict[str, Any]],
        normalization_stats: Dict[str, Any],
        write_figures: bool,
    ) -> Dict[str, Any]:
        denorm_predictions, denorm_targets, denorm_inputs = _denormalize_bundle(
            predictions=predictions,
            targets=targets,
            inputs=inputs,
            stats=normalization_stats,
            normalization=str(config.get("normalization", "none")),
        )
        summary = summarize_window_metrics(
            predictions=denorm_predictions,
            targets=denorm_targets,
            inputs=denorm_inputs,
            metadata=metadata,
            channels=list(config["target_channels"]),
            sampling_frequency=float(config["sampling_frequency"]),
        )
        save_metric_summary(summary, output_root / "metrics" / model_name)
        if write_figures:
            _visualize_examples(config, output_root / "figures", denorm_predictions, denorm_targets, denorm_inputs, metadata)
        return summary


    def _collect_model_outputs(model, dataloader, device) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        gathered = collect_predictions(model, dataloader, device)
        return (
            gathered["predictions"].numpy(),
            gathered["targets"].numpy(),
            gathered["inputs"].numpy(),
            gathered["metadata"],
        )


    def _count_parameters(model) -> Tuple[int, int]:
        total_params = sum(parameter.numel() for parameter in model.parameters())
        trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        return int(total_params), int(trainable_params)


    @torch.no_grad()
    def _measure_forward_latency_ms(model, window_size: int, input_dim: int, warmup: int = 20, repeats: int = 100) -> float:
        sample = torch.randn(1, int(window_size), int(input_dim), dtype=torch.float32)
        model = model.cpu().eval()
        for _ in range(int(warmup)):
            model(sample)
        start = time.perf_counter()
        for _ in range(int(repeats)):
            model(sample)
        elapsed = time.perf_counter() - start
        return float(elapsed * 1000.0 / max(int(repeats), 1))


    def _deployment_verdict(parameter_count: int, cpu_forward_ms: float) -> str:
        if parameter_count == 0 and cpu_forward_ms <= 10.0:
            return "yes_classical_filter"
        if parameter_count <= 100_000 and cpu_forward_ms <= 10.0:
            return "yes_embedded_friendly"
        if parameter_count <= 500_000 and cpu_forward_ms <= 30.0:
            return "possible_on_embedded_linux"
        if parameter_count <= 2_000_000 and cpu_forward_ms <= 60.0:
            return "edge_only_or_accelerated"
        return "unlikely_for_resource_constrained_embedded"


    def _profile_model(
        label: str,
        role: str,
        model_name: str,
        input_dim: int,
        output_dim: int,
        model_config: Dict[str, Any],
        window_size: int,
        state_dict: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        profile_model = build_model(
            model_name=model_name,
            input_dim=input_dim,
            output_dim=output_dim,
            model_config=model_config,
        ).cpu()
        if state_dict is not None:
            profile_model.load_state_dict(state_dict)
        total_params, trainable_params = _count_parameters(profile_model)
        cpu_forward_ms = _measure_forward_latency_ms(
            profile_model,
            window_size=window_size,
            input_dim=input_dim,
        )
        return {
            "model_name": label,
            "model_role": role,
            "architecture_name": model_name,
            "parameter_count": total_params,
            "trainable_parameter_count": trainable_params,
            "parameter_size_mb_fp32": float(total_params * 4 / (1024**2)),
            "cpu_forward_ms_per_window": cpu_forward_ms,
            "embedded_deployment_verdict": _deployment_verdict(total_params, cpu_forward_ms),
        }


    def _save_deployment_summary(summary_frame: pd.DataFrame, metrics_root: Path) -> Path:
        csv_path = metrics_root / "model_deployment_summary.csv"
        json_path = metrics_root / "model_deployment_summary.json"
        summary_frame.to_csv(csv_path, index=False)
        save_json({"models": summary_frame.to_dict(orient="records")}, json_path)
        return csv_path


    def _evaluate_trained_checkpoint(
        checkpoint_path: Path,
        evaluation_config: Dict[str, Any],
        output_root: Path,
        model_label: str,
        model_role: str,
        dataloader,
        device,
        normalization_stats: Dict[str, Any],
        write_figures: bool,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model_config = dict(checkpoint.get("config") or {})
        if not model_config:
            raise ValueError(f"Checkpoint at {checkpoint_path} does not contain its training config.")

        runtime_model_config = {
            **dict(model_config.get("model", {})),
            "sampling_frequency": float(model_config.get("sampling_frequency", evaluation_config["sampling_frequency"])),
        }
        model = build_model(
            model_name=model_config["model_name"],
            input_dim=len(model_config["input_channels"]),
            output_dim=len(model_config["target_channels"]),
            model_config=runtime_model_config,
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
            normalization_stats=normalization_stats,
            write_figures=write_figures,
        )
        deployment_row = _profile_model(
            label=model_label,
            role=model_role,
            model_name=str(model_config["model_name"]).lower(),
            input_dim=len(model_config["input_channels"]),
            output_dim=len(model_config["target_channels"]),
            model_config=runtime_model_config,
            window_size=int(model_config.get("window_size", evaluation_config["window_size"])),
            state_dict=checkpoint["model_state_dict"],
        )
        return summary, metadata, deployment_row


    def evaluate_checkpoint(config: Dict[str, Any], processed_root: Path, output_root: Path, checkpoint_path: Path) -> Path:
        """Evaluate the trained model, baselines, and optional trained comparison checkpoints."""
        require_torch()
        logger = get_logger("diomed-eval", output_root / "eval.log")
        ensure_dir(output_root)
        ensure_dir(output_root / "figures")
        metrics_root = ensure_dir(output_root / "metrics")

        normalization_stats = load_json(processed_root / "normalization_stats.json")
        dataset = WindowedPairDataset(
            processed_root / "test_samples.pkl",
            normalization=config.get("normalization", "none"),
            normalization_stats=normalization_stats,
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
        runtime_model_config = _build_model_config(config)

        overall_by_model: Dict[str, Dict[str, float]] = OrderedDict()
        summary_by_model: Dict[str, Dict[str, Any]] = OrderedDict()
        model_roles: Dict[str, str] = {primary_model_name: "trained"}
        deployment_rows: List[Dict[str, Any]] = []

        primary_summary, metadata, primary_deployment = _evaluate_trained_checkpoint(
            checkpoint_path=checkpoint_path,
            evaluation_config=config,
            output_root=output_root,
            model_label=primary_model_name,
            model_role="trained",
            dataloader=dataloader,
            device=device,
            normalization_stats=normalization_stats,
            write_figures=True,
        )
        overall_by_model[primary_model_name] = primary_summary["overall"]
        summary_by_model[primary_model_name] = primary_summary
        deployment_rows.append(primary_deployment)

        for trained_spec in _load_trained_comparison_specs(config):
            comparison_label = trained_spec["label"]
            comparison_path = Path(trained_spec["checkpoint_path"])
            if not comparison_path.exists():
                logger.warning("Skipping comparison checkpoint '%s' because it was not found at %s.", comparison_label, comparison_path)
                continue
            comparison_summary, _, comparison_deployment = _evaluate_trained_checkpoint(
                checkpoint_path=comparison_path,
                evaluation_config=config,
                output_root=output_root,
                model_label=comparison_label,
                model_role="trained_comparison",
                dataloader=dataloader,
                device=device,
                normalization_stats=normalization_stats,
                write_figures=False,
            )
            overall_by_model[comparison_label] = comparison_summary["overall"]
            summary_by_model[comparison_label] = comparison_summary
            model_roles[comparison_label] = "trained_comparison"
            deployment_rows.append(comparison_deployment)

        for baseline_model_name in _get_baseline_model_names(config):
            if baseline_model_name == "linear":
                logger.warning(
                    "Skipping baseline '%s' because it is untrained at evaluation time and would use random weights.",
                    baseline_model_name,
                )
                continue
            if baseline_model_name in {"identity", "lowpass", "butterworth", "savgol", "wiener"} and input_dim != output_dim:
                logger.warning(
                    "Skipping baseline '%s' because it requires identical input and target channel dimensions.",
                    baseline_model_name,
                )
                continue

            baseline_model = build_model(
                model_name=baseline_model_name,
                input_dim=input_dim,
                output_dim=output_dim,
                model_config=runtime_model_config,
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
                normalization_stats=normalization_stats,
                write_figures=False,
            )
            overall_by_model[baseline_model_name] = baseline_summary["overall"]
            summary_by_model[baseline_model_name] = baseline_summary
            model_roles[baseline_model_name] = "baseline"
            deployment_rows.append(
                _profile_model(
                    label=baseline_model_name,
                    role="baseline",
                    model_name=baseline_model_name,
                    input_dim=input_dim,
                    output_dim=output_dim,
                    model_config=runtime_model_config,
                    window_size=int(config["window_size"]),
                )
            )

        comparison_frame = _build_comparison_frame(
            overall_by_model,
            primary_model_name,
            model_roles=model_roles,
        )
        comparison_path = _save_comparison_summary(comparison_frame, metrics_root)
        deployment_frame = pd.DataFrame(deployment_rows)
        if not deployment_frame.empty:
            _save_deployment_summary(deployment_frame, metrics_root)
        delta_path = _save_motion_delta_summary(
            summary_by_model=summary_by_model,
            metrics_root=metrics_root,
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


    def aggregate_multiseed_evaluations(
        config: Dict[str, Any],
        seed_output_roots: List[Path],
        aggregate_output_root: Path,
    ) -> Path | None:
        metrics_root = ensure_dir(aggregate_output_root / "metrics")
        comparison_frames = []
        deployment_frames = []
        per_motion_frames: Dict[str, List[pd.DataFrame]] = {}

        for seed_root in seed_output_roots:
            comparison_path = seed_root / "metrics" / "model_comparison.csv"
            if not comparison_path.exists():
                continue
            comparison_frame = load_csv_table(comparison_path)
            comparison_frame["seed"] = seed_root.parent.name
            comparison_frames.append(comparison_frame)

            deployment_path = seed_root / "metrics" / "model_deployment_summary.csv"
            if deployment_path.exists():
                deployment_frame = load_csv_table(deployment_path)
                deployment_frame["seed"] = seed_root.parent.name
                deployment_frames.append(deployment_frame)

            for model_name in comparison_frame["model_name"].unique():
                per_motion_path = seed_root / "metrics" / str(model_name) / "per_motion_metrics.csv"
                if not per_motion_path.exists():
                    continue
                frame = load_csv_table(per_motion_path)
                frame["seed"] = seed_root.parent.name
                per_motion_frames.setdefault(str(model_name), []).append(frame)

        if not comparison_frames:
            return None

        comparison_all = pd.concat(comparison_frames, ignore_index=True)
        aggregated_rows = []
        numeric_columns = [col for col in comparison_all.columns if col not in {"model_name", "model_role", "seed"}]
        for (model_name, model_role), frame in comparison_all.groupby(["model_name", "model_role"], sort=False):
            row = {"model_name": model_name, "model_role": model_role, "num_seeds": int(frame["seed"].nunique())}
            for column in numeric_columns:
                row[f"{column}_mean"] = float(frame[column].mean())
                row[f"{column}_std"] = float(frame[column].std(ddof=0))
            aggregated_rows.append(row)
        aggregated_comparison = pd.DataFrame(aggregated_rows)
        aggregated_csv_path = metrics_root / "multiseed_model_comparison.csv"
        aggregated_comparison.to_csv(aggregated_csv_path, index=False)
        save_json({"models": aggregated_comparison.to_dict(orient="records")}, metrics_root / "multiseed_model_comparison.json")

        if deployment_frames:
            deployment_all = pd.concat(deployment_frames, ignore_index=True)
            deployment_rows = []
            deployment_numeric = [
                col for col in deployment_all.columns if col not in {"model_name", "model_role", "architecture_name", "embedded_deployment_verdict", "seed"}
            ]
            for (model_name, model_role, architecture_name), frame in deployment_all.groupby(
                ["model_name", "model_role", "architecture_name"],
                sort=False,
            ):
                row = {
                    "model_name": model_name,
                    "model_role": model_role,
                    "architecture_name": architecture_name,
                    "num_seeds": int(frame["seed"].nunique()),
                }
                for column in deployment_numeric:
                    row[f"{column}_mean"] = float(frame[column].mean())
                    row[f"{column}_std"] = float(frame[column].std(ddof=0))
                verdicts = frame["embedded_deployment_verdict"].value_counts()
                row["embedded_deployment_verdict"] = str(verdicts.index[0])
                deployment_rows.append(row)
            deployment_summary = pd.DataFrame(deployment_rows)
            deployment_summary.to_csv(metrics_root / "multiseed_model_deployment_summary.csv", index=False)
            save_json(
                {"models": deployment_summary.to_dict(orient="records")},
                metrics_root / "multiseed_model_deployment_summary.json",
            )

        primary_model_name = str(config["model_name"]).lower()
        statistical_rows = []
        primary_frames = per_motion_frames.get(primary_model_name, [])
        if primary_frames:
            primary_mean = pd.concat(primary_frames, ignore_index=True).groupby("motion_name", as_index=False).mean(numeric_only=True)
            for model_name, frame_list in per_motion_frames.items():
                if model_name == primary_model_name:
                    continue
                candidate_mean = pd.concat(frame_list, ignore_index=True).groupby("motion_name", as_index=False).mean(numeric_only=True)
                merged = primary_mean.merge(candidate_mean, on="motion_name", how="inner", suffixes=(f"_{primary_model_name}", f"_{model_name}"))
                for metric_name in DEFAULT_DELTA_METRICS:
                    primary_column = f"{metric_name}_{primary_model_name}"
                    candidate_column = f"{metric_name}_{model_name}"
                    if primary_column not in merged.columns or candidate_column not in merged.columns:
                        continue
                    deltas = merged[primary_column] - merged[candidate_column]
                    if len(deltas) == 0:
                        continue
                    try:
                        statistic, p_value = wilcoxon(deltas)
                    except ValueError:
                        statistic, p_value = np.nan, 1.0
                    mean_delta = float(deltas.mean())
                    if metric_name in LOWER_IS_BETTER_METRICS:
                        better_model = primary_model_name if mean_delta < 0 else model_name
                    elif metric_name in HIGHER_IS_BETTER_METRICS:
                        better_model = primary_model_name if mean_delta > 0 else model_name
                    else:
                        better_model = "undetermined"
                    statistical_rows.append(
                        {
                            "candidate_model": primary_model_name,
                            "reference_model": model_name,
                            "group_column": "motion_name",
                            "num_groups": int(len(deltas)),
                            "metric_name": metric_name,
                            "mean_delta_candidate_minus_reference": mean_delta,
                            "wilcoxon_statistic": float(statistic) if not pd.isna(statistic) else np.nan,
                            "p_value": float(p_value),
                            "better_model_by_mean_delta": better_model,
                        }
                    )
        statistical_frame = pd.DataFrame(statistical_rows)
        if not statistical_frame.empty:
            statistical_frame.to_csv(metrics_root / "multiseed_statistical_tests.csv", index=False)
            save_json(
                {"tests": statistical_frame.to_dict(orient="records")},
                metrics_root / "multiseed_statistical_tests.json",
            )
        return aggregated_csv_path


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


    def aggregate_multiseed_evaluations(*args, **kwargs):  # pragma: no cover - runtime safeguard only
        require_torch()
