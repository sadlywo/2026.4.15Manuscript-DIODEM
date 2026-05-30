from __future__ import annotations

import argparse
import copy
import csv
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


DEFAULT_WEIGHT_SENSITIVITY_SEEDS = [42, 43, 44, 45, 46]
FINAL_LOSS_WEIGHTS = {"time_l1": 1.0, "mse": 0.5, "spectral": 0.2}
LOCAL_WEIGHT_SWEEP_VALUES = {
    "time_l1": [0.25, 0.5, 1.0, 1.5, 2.0],
    "mse": [0.125, 0.25, 0.5, 0.75, 1.0],
    "spectral": [0.05, 0.1, 0.2, 0.3, 0.4],
}

METRIC_SPECS = [
    {"source": "rmse_mean", "prefix": "rmse", "label": "RMSE", "precision": 4, "direction": "lower"},
    {"source": "pearson_mean", "prefix": "pearson", "label": "Pearson", "precision": 4, "direction": "higher"},
    {"source": "psd_distance_mean", "prefix": "psd_distance", "label": "PSD Dist.", "precision": 5, "direction": "lower"},
    {
        "source": "hf_ratio_improvement_mean",
        "prefix": "hf_ratio_improvement",
        "label": "HF Improve.",
        "precision": 3,
        "direction": "higher",
    },
    {"source": "acc_norm_rmse", "prefix": "acc_norm_rmse", "label": "Acc Norm RMSE", "precision": 4, "direction": "lower"},
    {"source": "gyr_norm_rmse", "prefix": "gyr_norm_rmse", "label": "Gyr Norm RMSE", "precision": 4, "direction": "lower"},
]


def _load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config at {config_path} did not produce a mapping.")
    config["config_path"] = str(config_path)
    config["config_dir"] = str(config_path.parent)
    config["repo_root"] = str(config_path.parent.parent.parent)
    return config


def _deep_update(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _make_variant(
    name: str,
    label: str,
    purpose: str,
    changed_term: str,
    overrides: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "name": name,
        "label": label,
        "purpose": purpose,
        "changed_term": changed_term,
        "overrides": overrides,
    }


def _weight_token(value: float) -> str:
    numeric = float(value)
    text = f"{numeric:.1f}" if numeric.is_integer() else f"{numeric:.4g}"
    text = text.replace(".", "p").replace("-", "m")
    return text


def _term_variant_prefix(term: str) -> str:
    return {"time_l1": "l1", "mse": "mse", "spectral": "spec"}[term]


def _term_label(term: str) -> str:
    return {"time_l1": "L1", "mse": "MSE", "spectral": "Spectral"}[term]


def _term_purpose(term: str, value: float, baseline_value: float) -> str:
    term_name = _term_label(term)
    direction = "Increase" if value > baseline_value else "Reduce"
    if term == "time_l1":
        role = "the L1 reconstruction weight to probe point-wise robustness."
    elif term == "mse":
        role = "the MSE contribution to probe the quadratic reconstruction penalty."
    else:
        role = "spectral consistency to probe frequency-domain alignment."
    return f"{direction} {role}"


def _build_variants(preset: str) -> List[Dict[str, Any]]:
    normalized = str(preset).strip().lower()
    if normalized not in {"local", "complete", "fast"}:
        raise ValueError(f"Unknown weight sensitivity preset: {preset}")

    local_variants: List[Dict[str, Any]] = [
        _make_variant("baseline", "Baseline", "Final loss setting used in the main paper.", "none", {}),
    ]
    for term, values in LOCAL_WEIGHT_SWEEP_VALUES.items():
        baseline_value = FINAL_LOSS_WEIGHTS[term]
        for value in values:
            if float(value) == baseline_value:
                continue
            prefix = _term_variant_prefix(term)
            local_variants.append(
                _make_variant(
                    f"{prefix}_{_weight_token(value)}",
                    f"{_term_label(term)} = {_format_weight(value)}",
                    _term_purpose(term, value, baseline_value),
                    term,
                    {"loss_weights": {term: float(value)}},
                )
            )
    if normalized == "fast":
        keep = {"baseline"}
        for term, values in LOCAL_WEIGHT_SWEEP_VALUES.items():
            prefix = _term_variant_prefix(term)
            keep.add(f"{prefix}_{_weight_token(values[0])}")
            keep.add(f"{prefix}_{_weight_token(values[-1])}")
        return [variant for variant in local_variants if variant["name"] in keep]
    return local_variants


def _final_loss_weights(config: Dict[str, Any]) -> Dict[str, float]:
    weights = dict(config.get("loss_weights", {}))
    return {
        "time_l1": float(weights.get("time_l1", weights.get("l1", 0.0))),
        "mse": float(weights.get("mse", 0.0)),
        "spectral": float(weights.get("spectral", 0.0)),
    }


def _build_variant_config(base_config: Dict[str, Any], variant: Dict[str, Any], outputs_root: str, seed: int) -> Dict[str, Any]:
    config = copy.deepcopy(base_config)
    _deep_update(config, variant.get("overrides", {}))
    config["seed"] = int(seed)
    config["outputs_root"] = outputs_root
    config["weight_sensitivity_variant"] = variant["name"]
    config["weight_sensitivity_label"] = variant["label"]
    config["weight_sensitivity_purpose"] = variant["purpose"]
    config["weight_sensitivity_changed_term"] = variant["changed_term"]

    evaluation = dict(config.get("evaluation", {}))
    evaluation["trained_model_checkpoints"] = []
    evaluation["baseline_models"] = []
    evaluation["max_visualizations"] = 0
    config["evaluation"] = evaluation
    return config


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_seeds(base_config: Dict[str, Any], cli_seeds: List[int] | None, cli_seed: int | None) -> List[int]:
    if cli_seeds:
        seeds = [int(seed) for seed in cli_seeds]
    elif cli_seed is not None:
        seeds = [int(cli_seed)]
    elif base_config.get("experiment_seeds"):
        seeds = [int(seed) for seed in base_config["experiment_seeds"]]
    else:
        seeds = list(DEFAULT_WEIGHT_SENSITIVITY_SEEDS)

    ordered: List[int] = []
    seen = set()
    for seed in seeds:
        if seed in seen:
            continue
        seen.add(seed)
        ordered.append(seed)
    return ordered


def _format_weight(value: float) -> str:
    value = float(value)
    if value == 0.0:
        return "0"
    if 0.01 <= abs(value) < 100:
        return f"{value:.4g}"
    return f"{value:.0e}"


def _format_float(value: float, precision: int) -> str:
    return f"{float(value):.{precision}f}"


def _format_mean_std(mean: float, std: float, precision: int) -> str:
    return f"{_format_float(mean, precision)} +/- {_format_float(std, precision)}"


def _format_percent(value: float | str) -> str:
    if value == "":
        return ""
    return f"{float(value):+.2f}%"


def _plan_rows(base_config: Dict[str, Any], variants: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for variant in variants:
        config = copy.deepcopy(base_config)
        _deep_update(config, variant.get("overrides", {}))
        weights = _final_loss_weights(config)
        rows.append(
            {
                "Variant": variant["name"],
                "Label": variant["label"],
                "Changed term": str(variant["changed_term"]),
                "Purpose": str(variant["purpose"]),
                "time_l1": _format_weight(weights["time_l1"]),
                "mse": _format_weight(weights["mse"]),
                "spectral": _format_weight(weights["spectral"]),
            }
        )
    return rows


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows available for CSV output: {path}")
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown_table(path: Path, title: str, intro: str, rows: List[Dict[str, Any]], notes: Iterable[str] = ()) -> None:
    if not rows:
        raise ValueError(f"No rows available for Markdown output: {path}")
    _ensure_dir(path.parent)
    headers = list(rows[0].keys())
    lines = [title, "", intro, ""]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    note_list = list(notes)
    if note_list:
        lines.append("")
        lines.append("Notes:")
        for note in note_list:
            lines.append(f"- {note}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_plan_markdown(path: Path, rows: List[Dict[str, str]], preset: str, seeds: List[int]) -> None:
    _write_markdown_table(
        path=path,
        title="**Supplementary Table Sx. Loss-weight sensitivity study design.**",
        intro=(
            f"This table defines the `{preset}` sensitivity preset used to assess whether the final composite-loss "
            f"coefficients produce a robust empirical trade-off. Unless otherwise stated, only the listed loss "
            f"coefficients are changed. Seeds: {', '.join(str(seed) for seed in seeds)}."
        ),
        rows=rows,
        notes=[
            "`baseline` matches the final loss used in the main experiments.",
            "`time_l1`, `mse`, and `spectral` are the only active loss weights in the current objective.",
        ],
    )


def _read_single_row_csv(path: Path, key: str, value: str) -> Dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if str(row.get(key, "")).lower() == str(value).lower():
                return {str(k): str(v) for k, v in row.items()}
    raise ValueError(f"Could not find row with {key}={value} in {path}")


def _is_blank(value: Any) -> bool:
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none", "null"}


def _get_metric_stat(row: Dict[str, Any], metric_name: str, stat: str) -> float:
    if stat == "mean":
        candidates = [metric_name, f"{metric_name}_mean"]
    elif stat == "std":
        candidates = [f"{metric_name}_std"]
        if metric_name.endswith("_mean"):
            candidates.append(f"{metric_name[:-5]}_std")
    else:
        raise ValueError(f"Unknown metric stat: {stat}")

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate in row and not _is_blank(row[candidate]):
            return float(row[candidate])

    if stat == "std":
        return 0.0
    available = ", ".join(sorted(row.keys()))
    raise KeyError(f"Could not find metric '{metric_name}' in row. Available columns: {available}")


def _extract_metric_stats(metrics_row: Dict[str, Any]) -> Dict[str, float]:
    output: Dict[str, float] = {}
    for spec in METRIC_SPECS:
        source = str(spec["source"])
        prefix = str(spec["prefix"])
        output[f"{prefix}_mean"] = _get_metric_stat(metrics_row, source, "mean")
        output[f"{prefix}_std"] = _get_metric_stat(metrics_row, source, "std")
    return output


def _get_num_seeds(metrics_row: Dict[str, Any], fallback: int) -> int:
    if "num_seeds" in metrics_row and not _is_blank(metrics_row["num_seeds"]):
        return int(float(metrics_row["num_seeds"]))
    return int(fallback)


def _make_summary_row(
    variant_config: Dict[str, Any],
    variant: Dict[str, Any],
    metrics_row: Dict[str, Any],
    seed_list: List[int],
) -> Dict[str, Any]:
    weights = _final_loss_weights(variant_config)
    row: Dict[str, Any] = {
        "variant": str(variant["name"]),
        "variant_label": str(variant["label"]),
        "changed_term": str(variant["changed_term"]),
        "purpose": str(variant["purpose"]),
        "time_l1": weights["time_l1"],
        "mse": weights["mse"],
        "spectral": weights["spectral"],
        "num_seeds": _get_num_seeds(metrics_row, fallback=max(1, len(seed_list))),
        "seed_list": ",".join(str(seed) for seed in seed_list),
    }
    row.update(_extract_metric_stats(metrics_row))
    return row


def _make_per_seed_row(
    variant_config: Dict[str, Any],
    variant: Dict[str, Any],
    seed: int,
    metrics_row: Dict[str, Any],
) -> Dict[str, Any]:
    weights = _final_loss_weights(variant_config)
    row: Dict[str, Any] = {
        "variant": str(variant["name"]),
        "variant_label": str(variant["label"]),
        "changed_term": str(variant["changed_term"]),
        "seed": int(seed),
        "time_l1": weights["time_l1"],
        "mse": weights["mse"],
        "spectral": weights["spectral"],
    }
    for spec in METRIC_SPECS:
        row[str(spec["prefix"])] = _get_metric_stat(metrics_row, str(spec["source"]), "mean")
    return row


def _build_numeric_summary_rows(summary_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    baseline = next((row for row in summary_rows if row["variant"] == "baseline"), None)
    if baseline is None:
        raise ValueError("Weight sensitivity summary requires a baseline row.")

    output: List[Dict[str, Any]] = []
    for row in summary_rows:
        numeric_row: Dict[str, Any] = {
            "variant": row["variant"],
            "variant_label": row["variant_label"],
            "changed_term": row["changed_term"],
            "purpose": row["purpose"],
            "time_l1": row["time_l1"],
            "mse": row["mse"],
            "spectral": row["spectral"],
            "num_seeds": row["num_seeds"],
            "seed_list": row["seed_list"],
        }
        for spec in METRIC_SPECS:
            prefix = str(spec["prefix"])
            mean = float(row[f"{prefix}_mean"])
            std = float(row[f"{prefix}_std"])
            baseline_mean = float(baseline[f"{prefix}_mean"])
            delta = mean - baseline_mean
            delta_percent: float | str = "" if baseline_mean == 0 else delta / baseline_mean * 100.0
            numeric_row[f"{prefix}_mean"] = mean
            numeric_row[f"{prefix}_std"] = std
            numeric_row[f"{prefix}_delta_vs_baseline"] = delta
            numeric_row[f"{prefix}_delta_percent_vs_baseline"] = delta_percent
        output.append(numeric_row)
    return output


def _build_manuscript_rows(numeric_rows: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for row in numeric_rows:
        manuscript_row: Dict[str, str] = {
            "Variant": str(row["variant_label"]),
            "Changed term": str(row["changed_term"]),
            "L1": _format_weight(float(row["time_l1"])),
            "MSE": _format_weight(float(row["mse"])),
            "Spectral": _format_weight(float(row["spectral"])),
            "Seeds": str(row["num_seeds"]),
        }
        for spec in METRIC_SPECS:
            prefix = str(spec["prefix"])
            manuscript_row[str(spec["label"])] = _format_mean_std(
                float(row[f"{prefix}_mean"]),
                float(row[f"{prefix}_std"]),
                int(spec["precision"]),
            )
        manuscript_row["Delta RMSE vs baseline"] = _format_percent(row["rmse_delta_percent_vs_baseline"])
        manuscript_row["Delta PSD vs baseline"] = _format_percent(row["psd_distance_delta_percent_vs_baseline"])
        manuscript_row["Delta HF vs baseline"] = _format_percent(row["hf_ratio_improvement_delta_percent_vs_baseline"])
        rows.append(manuscript_row)
    return rows


def _changed_weight_value(row: Dict[str, Any]) -> str:
    term = str(row["changed_term"])
    if term == "none":
        return "baseline"
    if term in {"time_l1", "mse", "spectral"}:
        return _format_weight(float(row[term]))
    return ""


def _build_plot_source_rows(numeric_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in numeric_rows:
        for spec in METRIC_SPECS:
            prefix = str(spec["prefix"])
            rows.append(
                {
                    "variant": row["variant"],
                    "variant_label": row["variant_label"],
                    "changed_term": row["changed_term"],
                    "changed_weight_value": _changed_weight_value(row),
                    "time_l1": row["time_l1"],
                    "mse": row["mse"],
                    "spectral": row["spectral"],
                    "metric": prefix,
                    "metric_label": spec["label"],
                    "metric_direction": spec["direction"],
                    "mean": row[f"{prefix}_mean"],
                    "std": row[f"{prefix}_std"],
                    "delta_vs_baseline": row[f"{prefix}_delta_vs_baseline"],
                    "delta_percent_vs_baseline": row[f"{prefix}_delta_percent_vs_baseline"],
                    "num_seeds": row["num_seeds"],
                    "seed_list": row["seed_list"],
                }
            )
    return rows


def _write_results_markdown(path: Path, rows: List[Dict[str, str]], preset: str) -> None:
    _write_markdown_table(
        path=path,
        title="**Supplementary Table Sy. Loss-weight sensitivity results.**",
        intro=(
            f"Rows report multi-seed results for the `{preset}` loss-weight sensitivity preset. Lower RMSE and PSD "
            "distance are better, whereas higher Pearson correlation and HF improvement are better."
        ),
        rows=rows,
        notes=[
            "`Baseline` matches the final coefficients used in the main experiments.",
            "Deltas are computed relative to the baseline mean.",
            "The table is intended to support local robustness of the selected coefficients, not exhaustive global optimization.",
        ],
    )


def _write_paper_outputs(
    repo_root: Path,
    sweep_root: Path,
    numeric_rows: List[Dict[str, Any]],
    manuscript_rows: List[Dict[str, str]],
    plot_rows: List[Dict[str, Any]],
    per_seed_rows: List[Dict[str, Any]],
    preset: str,
) -> None:
    paper_root = _ensure_dir(repo_root / "outputs" / "paper_tables")

    outputs = [
        (sweep_root / "weight_sensitivity_per_seed.csv", per_seed_rows),
        (sweep_root / "weight_sensitivity_summary_numeric.csv", numeric_rows),
        (sweep_root / "weight_sensitivity_summary.csv", manuscript_rows),
        (sweep_root / "weight_sensitivity_plot_source.csv", plot_rows),
        (paper_root / "supplementary_loss_weight_sensitivity_per_seed.csv", per_seed_rows),
        (paper_root / "supplementary_loss_weight_sensitivity_numeric.csv", numeric_rows),
        (paper_root / "supplementary_loss_weight_sensitivity_table.csv", manuscript_rows),
        (paper_root / "supplementary_loss_weight_sensitivity_plot_source.csv", plot_rows),
    ]
    for path, rows in outputs:
        _write_csv(path, rows)

    _write_results_markdown(sweep_root / "weight_sensitivity_summary.md", manuscript_rows, preset=preset)
    _write_results_markdown(
        paper_root / "supplementary_loss_weight_sensitivity_table.md",
        manuscript_rows,
        preset=preset,
    )


def _run_sweep(
    base_config: Dict[str, Any],
    variants: List[Dict[str, Any]],
    seeds: List[int],
    study_name: str,
    train: bool,
    evaluate: bool,
    summarize_only: bool,
) -> Path:
    from project.data.dataset_builder import build_processed_splits
    from project.evaluation.evaluate import aggregate_multiseed_evaluations, evaluate_checkpoint
    from project.experiments import build_seed_run_config
    from project.training.train import train_model
    from project.utils.io import ensure_dir, save_json
    from project.utils.torch_compat import require_torch

    require_torch()

    repo_root = Path(base_config["repo_root"])
    base_outputs_root = Path(str(base_config["outputs_root"]))
    sweep_root = ensure_dir((repo_root / base_outputs_root.parent / f"{base_outputs_root.name}_weight_sensitivity_{study_name}").resolve())
    processed_root = (repo_root / base_config["processed_root"]).resolve()
    if not summarize_only:
        build_processed_splits(base_config)

    summary_rows: List[Dict[str, Any]] = []
    per_seed_rows: List[Dict[str, Any]] = []

    for variant in variants:
        variant_name = str(variant["name"])
        variant_root_rel = str((sweep_root / variant_name).relative_to(repo_root))
        variant_base_config = _build_variant_config(base_config, variant, outputs_root=variant_root_rel, seed=seeds[0])

        seed_eval_roots: List[Path] = []
        available_seeds: List[int] = []
        for seed in seeds:
            seed_config = build_seed_run_config(variant_base_config, seed=int(seed), multi_seed=len(seeds) > 1)
            training_root = ensure_dir((repo_root / seed_config["outputs_root"] / "training").resolve())
            evaluation_root = ensure_dir((repo_root / seed_config["outputs_root"] / "evaluation").resolve())
            checkpoint_path = training_root / "checkpoints" / str(seed_config["evaluation"]["checkpoint_name"])

            if not summarize_only:
                if train:
                    checkpoint_path = train_model(seed_config, processed_root=processed_root, output_root=training_root)
                if evaluate:
                    evaluate_checkpoint(
                        seed_config,
                        processed_root=processed_root,
                        output_root=evaluation_root,
                        checkpoint_path=checkpoint_path,
                    )
                save_json(seed_config, training_root / "resolved_config.json")

            seed_eval_roots.append(evaluation_root)
            seed_metrics_path = evaluation_root / "metrics" / "model_comparison.csv"
            if seed_metrics_path.exists():
                seed_metrics_row = _read_single_row_csv(seed_metrics_path, "model_name", str(variant_base_config["model_name"]).lower())
                per_seed_rows.append(_make_per_seed_row(variant_base_config, variant, seed=int(seed), metrics_row=seed_metrics_row))
                available_seeds.append(int(seed))

        aggregate_output_root = ensure_dir((repo_root / variant_base_config["outputs_root"] / "evaluation").resolve())
        metrics_root = ensure_dir((aggregate_output_root / "metrics").resolve())
        aggregate_path = metrics_root / "multiseed_model_comparison.csv"
        if len(seeds) > 1 and (not aggregate_path.exists() or not summarize_only):
            aggregate_multiseed_evaluations(
                config=variant_base_config,
                seed_output_roots=seed_eval_roots,
                aggregate_output_root=aggregate_output_root,
            )

        if len(seeds) > 1:
            result_path = aggregate_path
        else:
            result_path = seed_eval_roots[0] / "metrics" / "model_comparison.csv"

        metrics_row = _read_single_row_csv(result_path, "model_name", str(variant_base_config["model_name"]).lower())
        summary_seed_list = available_seeds if available_seeds else seeds
        summary_rows.append(_make_summary_row(variant_base_config, variant, metrics_row, seed_list=summary_seed_list))

    numeric_rows = _build_numeric_summary_rows(summary_rows)
    manuscript_rows = _build_manuscript_rows(numeric_rows)
    plot_rows = _build_plot_source_rows(numeric_rows)
    _write_paper_outputs(
        repo_root=repo_root,
        sweep_root=sweep_root,
        numeric_rows=numeric_rows,
        manuscript_rows=manuscript_rows,
        plot_rows=plot_rows,
        per_seed_rows=per_seed_rows,
        preset=study_name,
    )
    return sweep_root / "weight_sensitivity_summary_numeric.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and summarize a local loss-weight sensitivity study.")
    parser.add_argument("--config", type=Path, default=Path("project/configs/tcn_causal.yaml"))
    parser.add_argument("--preset", choices=["local", "complete", "fast"], default="local")
    parser.add_argument("--seed", type=int, default=None, help="Optional single-seed override.")
    parser.add_argument("--seeds", type=int, nargs="*", default=None, help="Optional multi-seed sweep.")
    parser.add_argument("--plan-only", action="store_true", help="Only export the planned weight-sensitivity table.")
    parser.add_argument("--summarize-only", action="store_true", help="Skip training/evaluation and summarize existing outputs.")
    parser.add_argument("--skip-train", action="store_true", help="Skip training and use existing checkpoints.")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation and summarize existing metrics.")
    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Optional custom suffix for the sweep output directory.",
    )
    args = parser.parse_args()

    base_config = _load_config(args.config.resolve())
    seeds = _resolve_seeds(base_config, cli_seeds=args.seeds, cli_seed=args.seed)
    variants = _build_variants(args.preset)
    study_name = str(args.study_name).strip() if args.study_name else args.preset

    repo_root = Path(base_config["repo_root"])
    plan_rows = _plan_rows(base_config, variants)
    plan_csv = repo_root / "outputs" / "paper_tables" / f"supplementary_loss_weight_sensitivity_plan_{study_name}.csv"
    plan_md = repo_root / "outputs" / "paper_tables" / f"supplementary_loss_weight_sensitivity_plan_{study_name}.md"
    _write_csv(plan_csv, plan_rows)
    _write_plan_markdown(plan_md, plan_rows, preset=study_name, seeds=seeds)
    print(f"Wrote sensitivity plan CSV to {plan_csv}")
    print(f"Wrote sensitivity plan Markdown to {plan_md}")

    if args.plan_only:
        return

    summary_csv = _run_sweep(
        base_config=base_config,
        variants=variants,
        seeds=seeds,
        study_name=study_name,
        train=not args.skip_train and not args.summarize_only,
        evaluate=not args.skip_eval and not args.summarize_only,
        summarize_only=bool(args.summarize_only),
    )
    print(f"Wrote sensitivity summary CSV to {summary_csv}")
    print(f"Wrote paper numeric table to {repo_root / 'outputs' / 'paper_tables' / 'supplementary_loss_weight_sensitivity_numeric.csv'}")
    print(f"Wrote paper manuscript table to {repo_root / 'outputs' / 'paper_tables' / 'supplementary_loss_weight_sensitivity_table.csv'}")
    print(f"Wrote plot source table to {repo_root / 'outputs' / 'paper_tables' / 'supplementary_loss_weight_sensitivity_plot_source.csv'}")


if __name__ == "__main__":
    main()
