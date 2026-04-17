from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

from project.data.dataset_builder import build_processed_splits
from project.evaluation.evaluate import evaluate_checkpoint
from project.training.train import train_model
from project.utils.io import ensure_dir, load_json, save_json
from project.utils.torch_compat import require_torch


DEFAULT_ABLATION_VARIANTS: List[Dict[str, Any]] = [
    {
        "name": "full_model",
        "description": "Attachment latent code + composite loss.",
        "overrides": {},
    },
    {
        "name": "no_attachment_latent",
        "description": "Disable attachment latent code and its regularization terms.",
        "overrides": {
            "model": {"attach_latent_dim": 0},
            "loss_weights": {"attach_l2": 0.0, "attach_temporal": 0.0},
        },
    },
    {
        "name": "mse_only",
        "description": "Use only MSE reconstruction loss.",
        "overrides": {
            "loss_weights": {
                "time_l1": 0.0,
                "mse": 1.0,
                "derivative": 0.0,
                "spectral": 0.0,
                "smoothness": 0.0,
                "attach_l2": 0.0,
                "attach_temporal": 0.0,
            }
        },
    },
    {
        "name": "no_derivative_loss",
        "description": "Remove derivative consistency loss only.",
        "overrides": {"loss_weights": {"derivative": 0.0}},
    },
    {
        "name": "no_spectral_loss",
        "description": "Remove spectral consistency loss only.",
        "overrides": {"loss_weights": {"spectral": 0.0}},
    },
    {
        "name": "no_attachment_regularization",
        "description": "Keep attachment latent code but remove its explicit regularization terms.",
        "overrides": {
            "loss_weights": {"attach_l2": 0.0, "attach_temporal": 0.0},
        },
    },
]


def _deep_update(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _resolve_ablation_variants(config: Dict[str, Any], requested_names: Iterable[str] | None = None) -> List[Dict[str, Any]]:
    configured = config.get("ablation_variants")
    variants = copy.deepcopy(configured if configured is not None else DEFAULT_ABLATION_VARIANTS)
    if requested_names is None:
        return variants
    requested_set = {str(name).strip() for name in requested_names if str(name).strip()}
    selected = [variant for variant in variants if variant["name"] in requested_set]
    missing = sorted(requested_set - {variant["name"] for variant in selected})
    if missing:
        raise ValueError(f"Unknown ablation variants requested: {missing}")
    return selected


def build_ablation_config(base_config: Dict[str, Any], variant: Dict[str, Any], outputs_root: str) -> Dict[str, Any]:
    config = copy.deepcopy(base_config)
    overrides = dict(variant.get("overrides", {}))
    _deep_update(config, overrides)
    config["outputs_root"] = outputs_root
    config["ablation_variant"] = variant["name"]
    config["ablation_variant_description"] = variant.get("description", "")

    evaluation_config = dict(config.get("evaluation", {}))
    evaluation_config["trained_model_checkpoints"] = []
    evaluation_config["baseline_models"] = []
    config["evaluation"] = evaluation_config
    return config


def run_ablation_suite(
    base_config: Dict[str, Any],
    variant_names: Iterable[str] | None = None,
    train: bool = True,
    evaluate: bool = True,
) -> Path:
    require_torch()
    repo_root = Path(base_config["repo_root"])
    base_outputs_root = Path(str(base_config["outputs_root"]))
    ablation_root = ensure_dir((repo_root / base_outputs_root.parent / f"{base_outputs_root.name}_ablations").resolve())
    processed_root = (repo_root / base_config["processed_root"]).resolve()
    build_processed_splits(base_config)

    summary_rows: List[Dict[str, Any]] = []
    for variant in _resolve_ablation_variants(base_config, variant_names):
        variant_name = str(variant["name"])
        variant_outputs_root = str((ablation_root / variant_name).relative_to(repo_root))
        config = build_ablation_config(base_config, variant, outputs_root=variant_outputs_root)
        training_root = ensure_dir((repo_root / config["outputs_root"] / "training").resolve())
        evaluation_root = ensure_dir((repo_root / config["outputs_root"] / "evaluation").resolve())

        checkpoint_path = training_root / "checkpoints" / str(config["evaluation"]["checkpoint_name"])
        if train:
            checkpoint_path = train_model(config, processed_root=processed_root, output_root=training_root)
        if evaluate:
            evaluate_checkpoint(
                config,
                processed_root=processed_root,
                output_root=evaluation_root,
                checkpoint_path=checkpoint_path,
            )

        metrics_path = evaluation_root / "metrics" / str(config["model_name"]).lower() / "overall_metrics.json"
        metrics = load_json(metrics_path)
        summary_rows.append(
            {
                "variant_name": variant_name,
                "description": variant.get("description", ""),
                "outputs_root": config["outputs_root"],
                "attach_latent_dim": int(config.get("model", {}).get("attach_latent_dim", 0)),
                "time_l1": float(config["loss_weights"].get("time_l1", config["loss_weights"].get("l1", 0.0))),
                "mse": float(config["loss_weights"].get("mse", 0.0)),
                "derivative": float(config["loss_weights"].get("derivative", 0.0)),
                "spectral": float(config["loss_weights"].get("spectral", 0.0)),
                "smoothness": float(config["loss_weights"].get("smoothness", 0.0)),
                "attach_l2": float(config["loss_weights"].get("attach_l2", 0.0)),
                "attach_temporal": float(config["loss_weights"].get("attach_temporal", 0.0)),
                **metrics,
            }
        )

        variant_config_path = ablation_root / variant_name / "resolved_config.json"
        save_json(config, variant_config_path)

    summary_frame = pd.DataFrame(summary_rows).sort_values("rmse_mean").reset_index(drop=True)
    summary_csv_path = ablation_root / "ablation_summary.csv"
    summary_json_path = ablation_root / "ablation_summary.json"
    summary_frame.to_csv(summary_csv_path, index=False)
    save_json({"variants": summary_frame.to_dict(orient="records")}, summary_json_path)
    return summary_csv_path
