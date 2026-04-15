from __future__ import annotations

import argparse
from pathlib import Path

from project.data.dataset_builder import build_processed_splits
from project.training.train import train_model
from project.utils.io import ensure_dir, load_yaml_config
from project.utils.torch_compat import require_torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a supervised DIODEM artifact-compensation baseline.")
    parser.add_argument("--config", type=Path, default=Path("project/configs/default.yaml"))
    args = parser.parse_args()

    config = load_yaml_config(args.config.resolve())
    require_torch()

    processed_root = (Path(config["repo_root"]) / config["processed_root"]).resolve()
    output_root = ensure_dir((Path(config["repo_root"]) / config["outputs_root"] / "training").resolve())
    build_processed_splits(config)
    best_checkpoint = train_model(config, processed_root=processed_root, output_root=output_root)
    print(f"Best checkpoint saved to {best_checkpoint}")


if __name__ == "__main__":
    main()
