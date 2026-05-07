from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


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
    purpose: str,
    changed_term: str,
    overrides: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "name": name,
        "purpose": purpose,
        "changed_term": changed_term,
        "overrides": overrides,
    }


def _build_variants(preset: str) -> List[Dict[str, Any]]:
    common: List[Dict[str, Any]] = [
        _make_variant("baseline", "Reference setting used in the main paper.", "none", {}),
        _make_variant(
            "time_l1_0p5",
            "Reduce the weight of L1 reconstruction to test weaker point-wise robustness.",
            "time_l1",
            {"loss_weights": {"time_l1": 0.5}},
        ),
        _make_variant(
            "time_l1_2p0",
            "Increase the weight of L1 reconstruction to test stronger point-wise robustness.",
            "time_l1",
            {"loss_weights": {"time_l1": 2.0}},
        ),
        _make_variant(
            "mse_0p25",
            "Reduce the MSE contribution to test a lighter quadratic penalty.",
            "mse",
            {"loss_weights": {"mse": 0.25}},
        ),
        _make_variant(
            "mse_1p0",
            "Increase the MSE contribution to test a stronger quadratic penalty.",
            "mse",
            {"loss_weights": {"mse": 1.0}},
        ),
        _make_variant(
            "derivative_0p1",
            "Reduce derivative consistency to test weaker temporal dynamic preservation.",
            "derivative",
            {"loss_weights": {"derivative": 0.1}},
        ),
        _make_variant(
            "derivative_0p5",
            "Increase derivative consistency to test stronger temporal dynamic preservation.",
            "derivative",
            {"loss_weights": {"derivative": 0.5}},
        ),
        _make_variant(
            "spectral_0p1",
            "Reduce spectral consistency to test weaker frequency-domain alignment.",
            "spectral",
            {"loss_weights": {"spectral": 0.1}},
        ),
        _make_variant(
            "spectral_0p4",
            "Increase spectral consistency to test stronger frequency-domain alignment.",
            "spectral",
            {"loss_weights": {"spectral": 0.4}},
        ),
        _make_variant(
            "attach_l2_0",
            "Disable latent magnitude regularization.",
            "attach_l2",
            {"loss_weights": {"attach_l2": 0.0}},
        ),
        _make_variant(
            "attach_l2_1e4",
            "Weaken latent magnitude regularization by one order of magnitude.",
            "attach_l2",
            {"loss_weights": {"attach_l2": 1e-4}},
        ),
        _make_variant(
            "attach_l2_1e2",
            "Strengthen latent magnitude regularization by one order of magnitude.",
            "attach_l2",
            {"loss_weights": {"attach_l2": 1e-2}},
        ),
        _make_variant(
            "attach_temporal_0",
            "Disable latent temporal smoothness regularization.",
            "attach_temporal",
            {"loss_weights": {"attach_temporal": 0.0}},
        ),
        _make_variant(
            "attach_temporal_1e4",
            "Weaken latent temporal smoothness regularization by one order of magnitude.",
            "attach_temporal",
            {"loss_weights": {"attach_temporal": 1e-4}},
        ),
        _make_variant(
            "attach_temporal_1e2",
            "Strengthen latent temporal smoothness regularization by one order of magnitude.",
            "attach_temporal",
            {"loss_weights": {"attach_temporal": 1e-2}},
        ),
    ]
    if preset == "fast":
        keep = {
            "baseline",
            "derivative_0p1",
            "derivative_0p5",
            "spectral_0p1",
            "spectral_0p4",
            "attach_l2_0",
            "attach_l2_1e2",
            "attach_temporal_0",
            "attach_temporal_1e2",
        }
        return [variant for variant in common if variant["name"] in keep]
    return common


def _build_variant_config(base_config: Dict[str, Any], variant: Dict[str, Any], outputs_root: str, seed: int) -> Dict[str, Any]:
    config = copy.deepcopy(base_config)
    _deep_update(config, variant.get("overrides", {}))
    config["seed"] = int(seed)
    config["outputs_root"] = outputs_root
    config["weight_sensitivity_variant"] = variant["name"]
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
        seeds = [int(base_config["seed"])]
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
    if abs(value) >= 0.01 and abs(value) < 100:
        return f"{value:.4g}"
    return f"{value:.0e}"


def _plan_rows(base_config: Dict[str, Any], variants: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    base_weights = dict(base_config.get("loss_weights", {}))
    rows: List[Dict[str, str]] = []
    for variant in variants:
        config = copy.deepcopy(base_config)
        _deep_update(config, variant.get("overrides", {}))
        weights = dict(config.get("loss_weights", {}))
        rows.append(
            {
                "Variant": variant["name"],
                "Changed term": str(variant["changed_term"]),
                "Purpose": str(variant["purpose"]),
                "time_l1": _format_weight(weights.get("time_l1", weights.get("l1", 0.0))),
                "mse": _format_weight(weights.get("mse", 0.0)),
                "derivative": _format_weight(weights.get("derivative", 0.0)),
                "spectral": _format_weight(weights.get("spectral", 0.0)),
                "attach_l2": _format_weight(weights.get("attach_l2", 0.0)),
                "attach_temporal": _format_weight(weights.get("attach_temporal", 0.0)),
            }
        )
    return rows


def _write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_plan_markdown(path: Path, rows: List[Dict[str, str]], preset: str, seeds: List[int]) -> None:
    lines = []
    lines.append("**Supplementary Table Sx. Recommended loss-weight sensitivity study design.**")
    lines.append("")
    lines.append(
        f"This table defines the `{preset}` sensitivity preset used to assess whether the composite-loss coefficients "
        f"produce a robust empirical trade-off. Unless otherwise stated, all settings inherit the same model backbone, "
        f"data split, and preprocessing pipeline; only the listed loss coefficients are changed. Recommended seeds: {', '.join(str(seed) for seed in seeds)}."
    )
    lines.append("")
    headers = list(rows[0].keys())
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row[h] for h in headers) + " |")
    lines.append("")
    lines.append("Suggested interpretation:")
    lines.append("- `time_l1` and `mse` probe the balance between absolute and quadratic reconstruction penalties.")
    lines.append("- `derivative` tests the sensitivity of temporal dynamic preservation.")
    lines.append("- `spectral` tests the sensitivity of frequency-domain consistency.")
    lines.append("- `attach_l2` and `attach_temporal` test whether the attachment latent is under- or over-regularized.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_single_row_csv(path: Path, key: str, value: str) -> Dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get(key) == value:
                return {str(k): str(v) for k, v in row.items()}
    raise ValueError(f"Could not find row with {key}={value} in {path}")


def _format_result_rows(summary_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    baseline = next(row for row in summary_rows if row["Variant"] == "baseline")
    baseline_rmse = float(baseline["RMSE"])
    baseline_psd = float(baseline["PSD Dist."])
    baseline_hf = float(baseline["HF Improve."])
    output: List[Dict[str, str]] = []
    for row in summary_rows:
        rmse = float(row["RMSE"])
        psd = float(row["PSD Dist."])
        hf = float(row["HF Improve."])
        output.append(
            {
                **row,
                "Delta RMSE vs baseline": f"{(rmse - baseline_rmse) / baseline_rmse * 100.0:+.2f}%",
                "Delta PSD vs baseline": f"{(psd - baseline_psd) / baseline_psd * 100.0:+.2f}%",
                "Delta HF vs baseline": f"{(hf - baseline_hf) / baseline_hf * 100.0:+.2f}%",
            }
        )
    return output


def _write_results_markdown(path: Path, rows: List[Dict[str, str]], preset: str, multi_seed: bool) -> None:
    title = "**Supplementary Table Sy. Loss-weight sensitivity results.**"
    lines = [title, ""]
    if multi_seed:
        lines.append(
            f"Rows report aggregated multi-seed results for the `{preset}` sensitivity preset. Lower RMSE and PSD distance are better, whereas higher HF improvement is better."
        )
    else:
        lines.append(
            f"Rows report single-seed results for the `{preset}` sensitivity preset. Lower RMSE and PSD distance are better, whereas higher HF improvement is better."
        )
    lines.append("")
    headers = list(rows[0].keys())
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row[h] for h in headers) + " |")
    lines.append("")
    lines.append("Notes:")
    lines.append("- Variants should be interpreted relative to the `baseline` row, which matches the main-paper loss coefficients.")
    lines.append("- When a larger weight improves one metric while degrading another, the final choice should be justified as an empirical trade-off rather than as a globally optimal setting.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    build_processed_splits(base_config)

    result_rows: List[Dict[str, str]] = []
    for variant in variants:
        variant_name = str(variant["name"])
        variant_root_rel = str((sweep_root / variant_name).relative_to(repo_root))
        variant_base_config = _build_variant_config(base_config, variant, outputs_root=variant_root_rel, seed=seeds[0])

        seed_eval_roots: List[Path] = []
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
            seed_eval_roots.append(evaluation_root)
            save_json(seed_config, training_root / "resolved_config.json")

        metrics_root = ensure_dir((repo_root / variant_base_config["outputs_root"] / "evaluation" / "metrics").resolve())
        if len(seeds) > 1 and not summarize_only:
            aggregate_output_root = ensure_dir((repo_root / variant_base_config["outputs_root"] / "evaluation").resolve())
            aggregate_multiseed_evaluations(
                config=variant_base_config,
                seed_output_roots=seed_eval_roots,
                aggregate_output_root=aggregate_output_root,
            )

        if len(seeds) > 1:
            result_path = metrics_root / "multiseed_model_comparison.csv"
        else:
            result_path = seed_eval_roots[0] / "metrics" / "model_comparison.csv"

        metrics_row = _read_single_row_csv(result_path, "model_name", str(variant_base_config["model_name"]).lower())
        weights = dict(variant_base_config.get("loss_weights", {}))
        result_rows.append(
            {
                "Variant": variant_name,
                "Changed term": str(variant["changed_term"]),
                "time_l1": _format_weight(weights.get("time_l1", weights.get("l1", 0.0))),
                "mse": _format_weight(weights.get("mse", 0.0)),
                "derivative": _format_weight(weights.get("derivative", 0.0)),
                "spectral": _format_weight(weights.get("spectral", 0.0)),
                "attach_l2": _format_weight(weights.get("attach_l2", 0.0)),
                "attach_temporal": _format_weight(weights.get("attach_temporal", 0.0)),
                "RMSE": f"{float(metrics_row['rmse_mean']):.4f}",
                "Pearson": f"{float(metrics_row['pearson_mean']):.4f}",
                "PSD Dist.": f"{float(metrics_row['psd_distance_mean']):.5f}",
                "HF Improve.": f"{float(metrics_row['hf_ratio_improvement_mean']):.3f}",
                "Acc Norm RMSE": f"{float(metrics_row['acc_norm_rmse']):.4f}",
                "Gyr Norm RMSE": f"{float(metrics_row['gyr_norm_rmse']):.4f}",
            }
        )

    formatted_rows = _format_result_rows(result_rows)
    summary_csv = sweep_root / "weight_sensitivity_summary.csv"
    _write_csv(summary_csv, formatted_rows)
    _write_results_markdown(
        sweep_root / "weight_sensitivity_summary.md",
        formatted_rows,
        preset=study_name,
        multi_seed=len(seeds) > 1,
    )
    return summary_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and summarize a loss-weight sensitivity study.")
    parser.add_argument("--config", type=Path, default=Path("project/configs/tcn_causal.yaml"))
    parser.add_argument("--preset", choices=["fast", "complete"], default="complete")
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
    plan_csv = repo_root / "outputs" / "paper_tables" / f"supplementary_weight_sensitivity_plan_{study_name}.csv"
    plan_md = repo_root / "outputs" / "paper_tables" / f"supplementary_weight_sensitivity_plan_{study_name}.md"
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


if __name__ == "__main__":
    main()
