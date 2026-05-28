from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from project.experiments import run_ablation_suite
from project.main_make_loss_ablation_table import create_loss_ablation_table
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
    parser.add_argument("--seed", type=int, default=None, help="Optional single-seed override.")
    parser.add_argument("--seeds", type=int, nargs="*", default=None, help="Optional multi-seed ablation sweep.")
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
    parser.add_argument(
        "--skip-table",
        action="store_true",
        help="Skip automatic supplementary_loss_ablation_table generation.",
    )
    parser.add_argument(
        "--table-output-csv",
        type=Path,
        default=Path("outputs/paper_tables/supplementary_loss_ablation_table.csv"),
        help="Output CSV path for the paper-ready loss ablation table.",
    )
    parser.add_argument(
        "--table-output-md",
        type=Path,
        default=Path("outputs/paper_tables/supplementary_loss_ablation_table.md"),
        help="Output Markdown path for the paper-ready loss ablation table.",
    )
    args = parser.parse_args()

    config = load_yaml_config(args.config.resolve())
    require_torch()
    explicit_seeds = args.seeds if args.seeds else ([args.seed] if args.seed is not None else None)
    summary_path = run_ablation_suite(
        config,
        variant_names=args.variants,
        train=not args.skip_train,
        evaluate=not args.skip_eval,
        seeds=explicit_seeds,
    )
    print(f"Ablation summary saved to {summary_path}")
    if not args.skip_table and not args.skip_eval:
        create_loss_ablation_table(
            input_csv=summary_path,
            output_csv=args.table_output_csv,
            output_md=args.table_output_md,
        )
        print(f"Supplementary loss ablation table saved to {args.table_output_csv.resolve()}")


if __name__ == "__main__":
    main()
