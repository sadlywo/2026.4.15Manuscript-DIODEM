from __future__ import annotations

from typing import Dict, Iterable, List

import pandas as pd


def _validate_disjoint(groups: Dict[str, Iterable[str]]) -> None:
    seen: Dict[str, str] = {}
    for split_name, values in groups.items():
        for value in values:
            if value in seen:
                raise ValueError(
                    f"Value {value!r} is assigned to both {seen[value]!r} and {split_name!r}."
                )
            seen[value] = split_name


def _assign_from_mapping(frame: pd.DataFrame, column: str, mapping: Dict[str, List[str]]) -> pd.Series:
    _validate_disjoint(mapping)
    split_series = pd.Series("exclude", index=frame.index, dtype=object)
    for split_name, values in mapping.items():
        if not values:
            continue
        split_series.loc[frame[column].isin(values)] = split_name
    return split_series


def apply_anomaly_policy(frame: pd.DataFrame, config: Dict[str, object]) -> pd.DataFrame:
    """Apply anomaly handling after the primary split is assigned."""
    anomaly_config = dict(config.get("anomaly", {}))
    mode = anomaly_config.get("mode", "include_all")
    result = frame.copy()
    if "is_anomaly_case" not in result.columns:
        result["is_anomaly_case"] = False

    if mode == "include_all":
        return result
    if mode == "exclude_all":
        return result.loc[~result["is_anomaly_case"]].reset_index(drop=True)
    if mode == "exclude_from_train":
        mask = result["is_anomaly_case"] & (result["split"] == "train")
        result.loc[mask, "split"] = "exclude"
        return result
    if mode == "test_only":
        result.loc[result["is_anomaly_case"], "split"] = "test"
        return result
    raise ValueError(f"Unsupported anomaly mode: {mode}")


def assign_split_labels(pairs_df: pd.DataFrame, split_config: Dict[str, object]) -> pd.DataFrame:
    """Assign `train` / `val` / `test` labels without window-level leakage."""
    strategy = split_config.get("strategy", "by_experiment")
    result = pairs_df.copy()

    if strategy == "by_experiment":
        strategy_config = dict(split_config.get("by_experiment", {}))
        result["split"] = _assign_from_mapping(result, "experiment_id", strategy_config)
    elif strategy == "by_motion_type":
        strategy_config = dict(split_config.get("by_motion_type", {}))
        result["split"] = _assign_from_mapping(result, "motion_name", strategy_config)
    elif strategy == "by_chain":
        strategy_config = dict(split_config.get("by_chain", {}))
        result["split"] = _assign_from_mapping(result, "kc_type", strategy_config)
    else:
        raise ValueError(f"Unsupported split strategy: {strategy}")

    result = apply_anomaly_policy(result, split_config)
    valid_mask = result["split"].isin(["train", "val", "test"])
    return result.loc[valid_mask].reset_index(drop=True)
