from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from project.data.dataset_builder import build_processed_splits
from project.experiments import apply_runtime_overrides, build_seed_run_config, resolve_experiment_seeds
from project.training.train import train_model
from project.utils.io import ensure_dir, load_yaml_config
from project.utils.torch_compat import require_torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a supervised DIODEM artifact-compensation baseline.")
    parser.add_argument("--config", type=Path, default=Path("project/configs/default.yaml"))
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

    explicit_seeds = args.seeds if args.seeds else ([args.seed] if args.seed is not None else None)
    seeds = resolve_experiment_seeds(config, explicit_seeds=explicit_seeds)

    repo_root = Path(config["repo_root"])
    processed_root = (repo_root / config["processed_root"]).resolve()
    build_processed_splits(config)

    checkpoint_paths = []
    for seed in seeds:
        seed_config = build_seed_run_config(config, seed=seed, multi_seed=len(seeds) > 1)
        output_root = ensure_dir((repo_root / seed_config["outputs_root"] / "training").resolve())
        best_checkpoint = train_model(seed_config, processed_root=processed_root, output_root=output_root)
        checkpoint_paths.append((seed, best_checkpoint))

    if len(checkpoint_paths) == 1:
        print(f"Best checkpoint saved to {checkpoint_paths[0][1]}")
    else:
        for seed, checkpoint_path in checkpoint_paths:
            print(f"Seed {seed}: best checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
