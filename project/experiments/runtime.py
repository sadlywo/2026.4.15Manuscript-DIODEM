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
    if suffix:
        config["processed_root"] = _suffix_path(config["processed_root"], suffix)
        config["outputs_root"] = _suffix_path(config["outputs_root"], suffix)
    return config


def build_seed_run_config(config: Dict[str, Any], seed: int, multi_seed: bool) -> Dict[str, Any]:
    run_config = copy.deepcopy(config)
    run_config["seed"] = int(seed)
    if multi_seed:
        run_config["outputs_root"] = str(Path(run_config["outputs_root"]) / "seed_runs" / f"seed_{int(seed):04d}")
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
