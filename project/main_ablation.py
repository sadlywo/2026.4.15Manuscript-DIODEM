from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from project.experiments import run_ablation_suite
from project.utils.io import load_yaml_config
from project.utils.torch_compat import require_torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ablation experiments for the DIODEM artifact-compensation model.")
    parser.add_argument("--config", type=Path, default=Path("project/configs/default.yaml"))
    parser.add_argument(
        "--variants",
        nargs="*",
        default=None,
        help="Optional ablation variant names. Defaults to the full built-in ablation suite.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training and reuse saved checkpoints from each ablation output directory.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip evaluation and only train checkpoints for each ablation variant.",
    )
    args = parser.parse_args()

    config = load_yaml_config(args.config.resolve())
    require_torch()
    summary_path = run_ablation_suite(
        config,
        variant_names=args.variants,
        train=not args.skip_train,
        evaluate=not args.skip_eval,
    )
    print(f"Ablation summary saved to {summary_path}")


if __name__ == "__main__":
    main()
