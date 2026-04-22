from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from project.utils.io import ensure_dir, load_csv_table, save_json


MAIN_TABLE_METRICS = (
    ("rmse_mean", 4),
    ("acc_channel_rmse_mean", 4),
    ("gyr_channel_rmse_mean", 4),
    ("pearson_mean", 4),
    ("hf_ratio_improvement_mean", 4),
    ("psd_distance_mean", 4),
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paper-ready results tables from experiment runs.")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run specification in the form `label=outputs/run_dir` where the directory contains `evaluation/metrics` or is itself the evaluation root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/paper_tables"),
        help="Directory where the generated CSV/Markdown tables will be saved.",
    )
    args = parser.parse_args()

    run_specs = [_parse_run_spec(item) for item in args.run]
    output_dir = ensure_dir(args.output_dir.resolve())

    main_table = build_main_results_table(run_specs)
    main_csv_path = output_dir / "paper_main_results.csv"
    main_md_path = output_dir / "paper_main_results.md"
    main_json_path = output_dir / "paper_main_results.json"
    main_table.to_csv(main_csv_path, index=False)
    main_md_path.write_text(main_table.to_markdown(index=False), encoding="utf-8")
    save_json({"rows": main_table.to_dict(orient="records")}, main_json_path)
    print(f"Paper main table saved to {main_csv_path}")
    print(f"Paper markdown table saved to {main_md_path}")


def build_main_results_table(run_specs: List[Tuple[str, Path]]) -> pd.DataFrame:
    rows = []
    for setting_label, run_root in run_specs:
        metrics_root = _resolve_metrics_root(run_root)
        comparison_frame, multi_seed = _load_comparison_frame(metrics_root)
        deployment_frame = _load_deployment_frame(metrics_root)
        deployment_lookup = {
            str(row["model_name"]): row
            for row in deployment_frame.to_dict(orient="records")
        }

        for row in comparison_frame.to_dict(orient="records"):
            model_name = str(row["model_name"])
            deployment_row = deployment_lookup.get(model_name, {})
            output_row = {
                "setting": setting_label,
                "num_seeds": int(row.get("num_seeds", 1 if multi_seed else 1)),
                "model_name": model_name,
                "model_role": row.get("model_role", "unknown"),
            }
            for metric_name, decimals in MAIN_TABLE_METRICS:
                output_row[f"{metric_name}_text"] = _format_metric_text(row, metric_name, decimals=decimals)
                output_row[f"{metric_name}_mean"] = _extract_mean_value(row, metric_name)
                output_row[f"{metric_name}_std"] = _extract_std_value(row, metric_name)

            output_row["parameter_count_text"] = _format_integer_metric(
                deployment_row,
                "parameter_count",
                prefer_mean=multi_seed,
            )
            output_row["cpu_forward_ms_text"] = _format_metric_text(
                deployment_row,
                "cpu_forward_ms_per_window",
                decimals=3,
                prefer_mean=multi_seed,
            )
            output_row["embedded_deployment_verdict"] = deployment_row.get("embedded_deployment_verdict", "")
            rows.append(output_row)

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    preferred_order = {
        "trained": 0,
        "trained_comparison": 1,
        "baseline": 2,
    }
    frame["_role_order"] = frame["model_role"].map(lambda value: preferred_order.get(str(value), 9))
    frame = frame.sort_values(["setting", "_role_order", "model_name"]).drop(columns="_role_order").reset_index(drop=True)
    return frame


def _parse_run_spec(raw_text: str) -> Tuple[str, Path]:
    if "=" not in raw_text:
        raise ValueError(f"Run spec must look like `label=path`, got: {raw_text}")
    label, path_text = raw_text.split("=", 1)
    return label.strip(), Path(path_text.strip()).resolve()


def _resolve_metrics_root(run_root: Path) -> Path:
    candidates = [
        run_root / "evaluation" / "metrics",
        run_root / "metrics",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find metrics directory under {run_root}")


def _load_comparison_frame(metrics_root: Path) -> Tuple[pd.DataFrame, bool]:
    multiseed_path = metrics_root / "multiseed_model_comparison.csv"
    if multiseed_path.exists():
        return load_csv_table(multiseed_path), True
    return load_csv_table(metrics_root / "model_comparison.csv"), False


def _load_deployment_frame(metrics_root: Path) -> pd.DataFrame:
    multiseed_path = metrics_root / "multiseed_model_deployment_summary.csv"
    if multiseed_path.exists():
        return load_csv_table(multiseed_path)
    single_path = metrics_root / "model_deployment_summary.csv"
    if single_path.exists():
        return load_csv_table(single_path)
    return pd.DataFrame()


def _extract_mean_value(row: Dict[str, object], metric_name: str) -> float | None:
    if f"{metric_name}_mean" in row:
        return _to_float(row.get(f"{metric_name}_mean"))
    if metric_name in row:
        return _to_float(row.get(metric_name))
    return None


def _extract_std_value(row: Dict[str, object], metric_name: str) -> float | None:
    return _to_float(row.get(f"{metric_name}_std"))


def _format_metric_text(
    row: Dict[str, object],
    metric_name: str,
    decimals: int,
    prefer_mean: bool = True,
) -> str:
    if prefer_mean and f"{metric_name}_mean" in row:
        mean_value = _to_float(row.get(f"{metric_name}_mean"))
        std_value = _to_float(row.get(f"{metric_name}_std"))
    else:
        mean_value = _to_float(row.get(metric_name))
        std_value = _to_float(row.get(f"{metric_name}_std"))

    if mean_value is None:
        return ""
    if std_value is None:
        return f"{mean_value:.{decimals}f}"
    return f"{mean_value:.{decimals}f} +- {std_value:.{decimals}f}"


def _format_integer_metric(row: Dict[str, object], metric_name: str, prefer_mean: bool = True) -> str:
    if prefer_mean and f"{metric_name}_mean" in row:
        value = _to_float(row.get(f"{metric_name}_mean"))
        std_value = _to_float(row.get(f"{metric_name}_std"))
    else:
        value = _to_float(row.get(metric_name))
        std_value = _to_float(row.get(f"{metric_name}_std"))
    if value is None:
        return ""
    if std_value is None or abs(std_value) < 1e-12:
        return f"{int(round(value)):,}"
    return f"{int(round(value)):,} +- {int(round(std_value)):,}"


def _to_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    main()
