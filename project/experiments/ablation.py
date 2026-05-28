from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

from project.data.dataset_builder import build_processed_splits
from project.evaluation.evaluate import evaluate_checkpoint
from project.experiments.runtime import build_seed_run_config, resolve_experiment_seeds
from project.training.train import train_model
from project.utils.io import ensure_dir, load_json, save_json
from project.utils.torch_compat import require_torch


DEFAULT_ABLATION_VARIANTS: List[Dict[str, Any]] = [
    {
        "name": "full_model",
        "description": "Attachment latent code + L1/MSE/spectral composite loss.",
        "overrides": {},
    },
    {
        "name": "mse_only",
        "description": "Use only MSE reconstruction loss.",
        "overrides": {
            "loss_weights": {
                "time_l1": 0.0,
                "mse": 1.0,
                "spectral": 0.0,
            }
        },
    },
    {
        "name": "no_spectral_loss",
        "description": "Remove spectral consistency loss only.",
        "overrides": {"loss_weights": {"spectral": 0.0}},
    },
    {
        "name": "no_attachment_latent",
        "description": "Disable attachment latent code while retaining the final composite loss.",
        "overrides": {"model": {"attach_latent_dim": 0}},
    },
]

_SUMMARY_ID_COLUMNS = {"variant_name", "description", "outputs_root", "seed", "seed_label"}
_SUMMARY_CONFIG_COLUMNS = {
    "attach_latent_dim",
    "time_l1",
    "mse",
    "spectral",
}


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


def _strip_seed_run_suffix(path_text: str) -> str:
    path = Path(str(path_text))
    parts = path.parts
    if len(parts) >= 2 and parts[-2] == "seed_runs" and parts[-1].startswith("seed_"):
        return str(Path(*parts[:-2]))
    return str(path)


def _std_column_name(metric_column: str) -> str:
    if metric_column.endswith("_mean"):
        return f"{metric_column[:-5]}_std"
    return f"{metric_column}_std"


def aggregate_ablation_seed_rows(seed_rows: List[Dict[str, Any]]) -> pd.DataFrame:
    if not seed_rows:
        return pd.DataFrame()

    seed_frame = pd.DataFrame(seed_rows)
    aggregated_rows: List[Dict[str, Any]] = []
    metric_columns = [
        column
        for column in seed_frame.columns
        if column not in _SUMMARY_ID_COLUMNS
        and column not in _SUMMARY_CONFIG_COLUMNS
        and pd.api.types.is_numeric_dtype(pd.to_numeric(seed_frame[column], errors="coerce"))
    ]

    for variant_name, frame in seed_frame.groupby("variant_name", sort=False):
        first = frame.iloc[0]
        seeds = sorted(int(seed) for seed in frame["seed"].dropna().unique()) if "seed" in frame else []
        row: Dict[str, Any] = {
            "variant_name": variant_name,
            "description": first.get("description", ""),
            "outputs_root": _strip_seed_run_suffix(str(first.get("outputs_root", ""))),
            "num_seeds": int(len(seeds) if seeds else len(frame)),
            "seed_list": ",".join(str(seed) for seed in seeds),
        }
        for column in _SUMMARY_CONFIG_COLUMNS:
            if column in frame.columns:
                row[column] = first[column]
        for column in metric_columns:
            values = pd.to_numeric(frame[column], errors="coerce").dropna()
            if values.empty:
                continue
            row[column] = float(values.mean())
            row[_std_column_name(column)] = float(values.std(ddof=0))
        aggregated_rows.append(row)

    summary_frame = pd.DataFrame(aggregated_rows)
    if "rmse_mean" in summary_frame.columns:
        summary_frame = summary_frame.sort_values("rmse_mean")
    return summary_frame.reset_index(drop=True)


def _build_summary_row(config: Dict[str, Any], variant: Dict[str, Any], metrics: Dict[str, Any], seed: int | None = None) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "variant_name": str(variant["name"]),
        "description": variant.get("description", ""),
        "outputs_root": config["outputs_root"],
        "attach_latent_dim": int(config.get("model", {}).get("attach_latent_dim", 0)),
        "time_l1": float(config["loss_weights"].get("time_l1", config["loss_weights"].get("l1", 0.0))),
        "mse": float(config["loss_weights"].get("mse", 0.0)),
        "spectral": float(config["loss_weights"].get("spectral", 0.0)),
        **metrics,
    }
    if seed is not None:
        row["seed"] = int(seed)
        row["seed_label"] = f"seed_{int(seed):04d}"
    return row


def run_ablation_suite(
    base_config: Dict[str, Any],
    variant_names: Iterable[str] | None = None,
    train: bool = True,
    evaluate: bool = True,
    seeds: Iterable[int] | None = None,
) -> Path:
    require_torch()
    repo_root = Path(base_config["repo_root"])
    base_outputs_root = Path(str(base_config["outputs_root"]))
    ablation_root_name = base_outputs_root.name if base_outputs_root.name.endswith("_ablations") else f"{base_outputs_root.name}_ablations"
    ablation_root = ensure_dir((repo_root / base_outputs_root.parent / ablation_root_name).resolve())
    processed_root = (repo_root / base_config["processed_root"]).resolve()
    build_processed_splits(base_config)
    resolved_seeds = resolve_experiment_seeds(base_config, explicit_seeds=seeds)
    multi_seed = len(resolved_seeds) > 1

    seed_summary_rows: List[Dict[str, Any]] = []
    for variant in _resolve_ablation_variants(base_config, variant_names):
        variant_name = str(variant["name"])
        variant_outputs_root = str((ablation_root / variant_name).relative_to(repo_root))
        variant_config = build_ablation_config(base_config, variant, outputs_root=variant_outputs_root)
        variant_root = ensure_dir(ablation_root / variant_name)
        save_json(variant_config, variant_root / "resolved_config.json")

        for seed in resolved_seeds:
            config = build_seed_run_config(variant_config, seed=seed, multi_seed=multi_seed)
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
            seed_summary_rows.append(_build_summary_row(config, variant, metrics, seed=seed))

            if multi_seed:
                save_json(config, Path(training_root).parent / "resolved_config.json")

    seed_summary_frame = pd.DataFrame(seed_summary_rows)
    summary_frame = aggregate_ablation_seed_rows(seed_summary_rows) if multi_seed else seed_summary_frame.sort_values("rmse_mean").reset_index(drop=True)
    summary_csv_path = ablation_root / "ablation_summary.csv"
    summary_json_path = ablation_root / "ablation_summary.json"
    summary_frame.to_csv(summary_csv_path, index=False)
    save_json({"variants": summary_frame.to_dict(orient="records")}, summary_json_path)
    if multi_seed:
        seed_summary_csv_path = ablation_root / "ablation_summary_by_seed.csv"
        seed_summary_json_path = ablation_root / "ablation_summary_by_seed.json"
        seed_summary_frame.sort_values(["variant_name", "seed"]).to_csv(seed_summary_csv_path, index=False)
        save_json({"variants": seed_summary_frame.to_dict(orient="records")}, seed_summary_json_path)
    return summary_csv_path
