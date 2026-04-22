from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Iterable, List


def resolve_experiment_seeds(config: Dict[str, Any], explicit_seeds: Iterable[int] | None = None) -> List[int]:
    if explicit_seeds:
        seeds = [int(seed) for seed in explicit_seeds]
    else:
        configured = config.get("experiment_seeds")
        if configured:
            seeds = [int(seed) for seed in configured]
        else:
            seeds = [int(config["seed"])]

    ordered: List[int] = []
    seen = set()
    for seed in seeds:
        if seed in seen:
            continue
        seen.add(seed)
        ordered.append(seed)
    return ordered


def apply_runtime_overrides(
    base_config: Dict[str, Any],
    split_strategy: str | None = None,
    anomaly_mode: str | None = None,
    run_name: str | None = None,
) -> Dict[str, Any]:
    config = copy.deepcopy(base_config)

    if split_strategy is not None:
        config["split_strategy"] = str(split_strategy)
    if anomaly_mode is not None:
        anomaly = dict(config.get("anomaly", {}))
        anomaly["mode"] = str(anomaly_mode)
        config["anomaly"] = anomaly

    suffix = _derive_run_suffix(base_config, config, run_name)
    config["run_suffix"] = suffix
    if suffix:
        config["processed_root"] = _suffix_path(config["processed_root"], suffix)
        config["outputs_root"] = _suffix_path(config["outputs_root"], suffix)
        _rewrite_trained_checkpoint_specs(config, suffix=suffix, seed=None, multi_seed=False)
    return config


def build_seed_run_config(config: Dict[str, Any], seed: int, multi_seed: bool) -> Dict[str, Any]:
    run_config = copy.deepcopy(config)
    run_config["seed"] = int(seed)
    if multi_seed:
        run_config["outputs_root"] = str(Path(run_config["outputs_root"]) / "seed_runs" / f"seed_{int(seed):04d}")
    _rewrite_trained_checkpoint_specs(
        run_config,
        suffix=None,
        seed=int(seed),
        multi_seed=bool(multi_seed),
    )
    return run_config


def _derive_run_suffix(base_config: Dict[str, Any], config: Dict[str, Any], run_name: str | None) -> str | None:
    if run_name:
        return str(run_name).strip()

    parts: List[str] = []
    if str(config.get("split_strategy")) != str(base_config.get("split_strategy")):
        parts.append(str(config["split_strategy"]))
    base_anomaly = dict(base_config.get("anomaly", {})).get("mode")
    current_anomaly = dict(config.get("anomaly", {})).get("mode")
    if str(current_anomaly) != str(base_anomaly):
        parts.append(f"anomaly_{current_anomaly}")
    return "_".join(parts) if parts else None


def _suffix_path(path_text: str, suffix: str) -> str:
    path = Path(str(path_text))
    sanitized_suffix = str(suffix).replace("\\", "_").replace("/", "_").strip("_")
    if not sanitized_suffix:
        return str(path)
    return str(path.parent / f"{path.name}_{sanitized_suffix}")


def _rewrite_trained_checkpoint_specs(
    config: Dict[str, Any],
    suffix: str | None,
    seed: int | None,
    multi_seed: bool,
) -> None:
    evaluation = dict(config.get("evaluation", {}))
    raw_specs = list(evaluation.get("trained_model_checkpoints", []))
    if not raw_specs:
        return

    rewritten_specs: List[Dict[str, Any]] = []
    for spec in raw_specs:
        if isinstance(spec, str):
            rewritten_specs.append(_rewrite_checkpoint_spec({"checkpoint": spec}, suffix=suffix, seed=seed, multi_seed=multi_seed))
            continue
        if isinstance(spec, dict):
            rewritten_specs.append(_rewrite_checkpoint_spec(spec, suffix=suffix, seed=seed, multi_seed=multi_seed))
            continue
        rewritten_specs.append(spec)
    evaluation["trained_model_checkpoints"] = rewritten_specs
    config["evaluation"] = evaluation


def _rewrite_checkpoint_spec(spec: Dict[str, Any], suffix: str | None, seed: int | None, multi_seed: bool) -> Dict[str, Any]:
    rewritten = copy.deepcopy(spec)
    checkpoint_value = rewritten.get("checkpoint")
    if checkpoint_value is None:
        return rewritten

    checkpoint_path = Path(str(checkpoint_value))
    if checkpoint_path.name.lower() != "best.pt" or checkpoint_path.parent.name != "checkpoints":
        return rewritten
    if checkpoint_path.parent.parent.name != "training":
        return rewritten

    model_root = checkpoint_path.parent.parent.parent
    trailing_path = Path("training") / "checkpoints" / checkpoint_path.name

    if suffix:
        model_root = Path(_suffix_path(str(model_root), suffix))
    if multi_seed and seed is not None:
        model_root = model_root / "seed_runs" / f"seed_{int(seed):04d}"

    rewritten["checkpoint"] = str(model_root / trailing_path)
    return rewritten
