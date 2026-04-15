from __future__ import annotations

import argparse
from pathlib import Path

from project.evaluation.evaluate import evaluate_checkpoint
from project.utils.io import ensure_dir, load_yaml_config
from project.utils.torch_compat import require_torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained DIODEM artifact-compensation baseline.")
    parser.add_argument("--config", type=Path, default=Path("project/configs/default.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()

    config = load_yaml_config(args.config.resolve())
    require_torch()

    repo_root = Path(config["repo_root"])
    processed_root = (repo_root / config["processed_root"]).resolve()
    output_root = ensure_dir((repo_root / config["outputs_root"] / "evaluation").resolve())
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = repo_root / config["outputs_root"] / "training" / "checkpoints" / config["evaluation"]["checkpoint_name"]
    checkpoint_path = checkpoint_path.resolve()
    metrics_path = evaluate_checkpoint(config, processed_root=processed_root, output_root=output_root, checkpoint_path=checkpoint_path)
    print(f"Evaluation metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
