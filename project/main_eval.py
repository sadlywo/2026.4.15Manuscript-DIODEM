from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from project.evaluation.evaluate import aggregate_multiseed_evaluations, evaluate_checkpoint
from project.experiments import apply_runtime_overrides, build_seed_run_config, resolve_experiment_seeds
from project.utils.io import ensure_dir, load_yaml_config
from project.utils.torch_compat import require_torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained DIODEM artifact-compensation baseline.")
    parser.add_argument("--config", type=Path, default=Path("project/configs/default.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None, help="Optional single-seed override.")
    parser.add_argument("--seeds", type=int, nargs="*", default=None, help="Optional multi-seed sweep.")
    parser.add_argument(
        "--split-strategy",
        choices=["by_experiment", "by_motion_type", "by_chain"],
        default=None,
        help="Override the configured split strategy.",
    )
    parser.add_argument(
        "--anomaly-mode",
        choices=["include_all", "exclude_all", "exclude_from_train", "test_only"],
        default=None,
        help="Override the configured anomaly policy.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional suffix used to derive dedicated processed/output directories for this run.",
    )
    args = parser.parse_args()

    config = apply_runtime_overrides(
        load_yaml_config(args.config.resolve()),
        split_strategy=args.split_strategy,
        anomaly_mode=args.anomaly_mode,
        run_name=args.run_name,
    )
    require_torch()

    repo_root = Path(config["repo_root"])
    processed_root = (repo_root / config["processed_root"]).resolve()
    explicit_seeds = args.seeds if args.seeds else ([args.seed] if args.seed is not None else None)
    seeds = resolve_experiment_seeds(config, explicit_seeds=explicit_seeds)

    metrics_paths = []
    seed_output_roots = []
    for seed in seeds:
        seed_config = build_seed_run_config(config, seed=seed, multi_seed=len(seeds) > 1)
        output_root = ensure_dir((repo_root / seed_config["outputs_root"] / "evaluation").resolve())
        checkpoint_path = args.checkpoint
        if checkpoint_path is None:
            checkpoint_path = (
                repo_root
                / seed_config["outputs_root"]
                / "training"
                / "checkpoints"
                / seed_config["evaluation"]["checkpoint_name"]
            )
        checkpoint_path = Path(checkpoint_path).resolve()
        metrics_path = evaluate_checkpoint(
            seed_config,
            processed_root=processed_root,
            output_root=output_root,
            checkpoint_path=checkpoint_path,
        )
        metrics_paths.append((seed, metrics_path))
        seed_output_roots.append(output_root)

    if len(seed_output_roots) > 1:
        aggregate_output_root = ensure_dir((repo_root / config["outputs_root"] / "evaluation").resolve())
        aggregate_path = aggregate_multiseed_evaluations(
            config=config,
            seed_output_roots=seed_output_roots,
            aggregate_output_root=aggregate_output_root,
        )
        if aggregate_path is not None:
            print(f"Multi-seed evaluation summary saved to {aggregate_path}")
        else:
            print("Multi-seed evaluation finished, but no aggregate summary could be generated.")
    elif metrics_paths:
        print(f"Evaluation metrics saved to {metrics_paths[0][1]}")


if __name__ == "__main__":
    main()
