from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from project.data.dataset_builder import build_processed_splits
from project.evaluation.evaluate import aggregate_multiseed_evaluations, evaluate_checkpoint
from project.experiments import apply_runtime_overrides, build_seed_run_config, resolve_experiment_seeds
from project.models import build_model
from project.training.train import train_model
from project.utils.io import ensure_dir, load_csv_table, load_yaml_config
from project.utils.torch_compat import TORCH_AVAILABLE, require_torch, torch


CAUSAL_MODEL_ORDER = (
    "tcn_causal",
    "transformer_causal",
    "gru_causal",
    "lstm_causal",
    "mlp_causal",
)

MODEL_LABELS = {
    "tcn_causal": "TCN-causal",
    "transformer_causal": "Transformer-causal",
    "gru_causal": "GRU-causal",
    "lstm_causal": "LSTM-causal",
    "mlp_causal": "MLP-causal",
}

CAUSAL_TYPES = {
    "tcn_causal": "dilated convolution",
    "transformer_causal": "causal self-attention mask",
    "gru_causal": "recurrent hidden state",
    "lstm_causal": "recurrent hidden/cell state",
    "mlp_causal": "per-timestep zero-history",
}

CAUSAL_TABLE_COLUMNS = [
    "Setting",
    "Model",
    "Causal type",
    "RMSE",
    "Pearson",
    "PSD Dist.",
    "HF Improve.",
    "Parameters",
    "FP32 size (MB)",
    "CPU forward ms/window",
    "Streaming ms/step",
    "Deployment verdict",
    "Loss profile",
]

LOSS_PROFILES = {
    "best_sensitivity": {
        "description": "L1=1.0, MSE=0.25, Derivative=0.3, Spectral=0.4, Attachment=1e-3/1e-3",
        "weights": {
            "time_l1": 1.0,
            "mse": 0.25,
            "derivative": 0.3,
            "spectral": 0.4,
            "attach_l2": 0.001,
            "attach_temporal": 0.001,
        },
    },
}

SETTING_SPECS = {
    "by_experiment": {
        "split_strategy": "by_experiment",
        "anomaly_mode": "exclude_all",
        "run_name": "by_experiment",
    },
    "by_motion_type": {
        "split_strategy": "by_motion_type",
        "anomaly_mode": "exclude_all",
        "run_name": "by_motion_type",
    },
    "anomaly_test_only": {
        "split_strategy": "by_experiment",
        "anomaly_mode": "test_only",
        "run_name": "anomaly_test_only",
    },
}


def _apply_loss_profile(config: Dict[str, Any], loss_profile: str) -> Dict[str, Any]:
    if loss_profile not in LOSS_PROFILES:
        raise ValueError(f"Unsupported loss profile: {loss_profile}")
    updated = copy.deepcopy(config)
    updated["loss_profile"] = loss_profile
    updated["loss_weights"] = dict(LOSS_PROFILES[loss_profile]["weights"])
    return updated


def _load_setting_config(
    model_name: str,
    config_dir: Path,
    output_root: Path,
    setting: str,
    loss_profile: str,
) -> Dict[str, Any]:
    if setting not in SETTING_SPECS:
        raise ValueError(f"Unsupported setting: {setting}")
    config_path = config_dir / f"{model_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing causal model config: {config_path}")

    config = _apply_loss_profile(load_yaml_config(config_path.resolve()), loss_profile)
    config["repo_root"] = str(Path(__file__).resolve().parent.parent)
    config["outputs_root"] = str(output_root / model_name)
    spec = SETTING_SPECS[setting]
    return apply_runtime_overrides(
        config,
        split_strategy=spec["split_strategy"],
        anomaly_mode=spec["anomaly_mode"],
        run_name=spec["run_name"],
    )


def _resolve_models(models: Iterable[str]) -> List[str]:
    ordered: List[str] = []
    seen = set()
    for model_name in models:
        normalized = str(model_name).lower()
        if normalized not in CAUSAL_MODEL_ORDER:
            raise ValueError(
                f"Unsupported causal model '{model_name}'. "
                f"Expected one of: {', '.join(CAUSAL_MODEL_ORDER)}"
            )
        if normalized in seen:
            continue
        ordered.append(normalized)
        seen.add(normalized)
    return ordered


def _train_and_evaluate_model(
    config: Dict[str, Any],
    seeds: List[int],
    skip_train: bool,
    skip_eval: bool,
) -> None:
    require_torch()
    repo_root = Path(config["repo_root"])
    processed_root = (repo_root / config["processed_root"]).resolve()
    build_processed_splits(config)

    seed_output_roots: List[Path] = []
    for seed in seeds:
        seed_config = build_seed_run_config(config, seed=seed, multi_seed=len(seeds) > 1)
        if not skip_train:
            train_root = ensure_dir((repo_root / seed_config["outputs_root"] / "training").resolve())
            train_model(seed_config, processed_root=processed_root, output_root=train_root)

        if not skip_eval:
            eval_root = ensure_dir((repo_root / seed_config["outputs_root"] / "evaluation").resolve())
            checkpoint_path = (
                repo_root
                / seed_config["outputs_root"]
                / "training"
                / "checkpoints"
                / seed_config["evaluation"]["checkpoint_name"]
            ).resolve()
            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"Checkpoint not found for {config['model_name']} seed {seed}: {checkpoint_path}"
                )
            evaluate_checkpoint(
                seed_config,
                processed_root=processed_root,
                output_root=eval_root,
                checkpoint_path=checkpoint_path,
            )
            seed_output_roots.append(eval_root)

    if len(seed_output_roots) > 1:
        aggregate_root = ensure_dir((repo_root / config["outputs_root"] / "evaluation").resolve())
        aggregate_multiseed_evaluations(
            config=config,
            seed_output_roots=seed_output_roots,
            aggregate_output_root=aggregate_root,
        )


def _metric_value(row: pd.Series, base_name: str) -> tuple[float | None, float | None]:
    mean_name = f"{base_name}_mean"
    std_name = f"{base_name}_std"
    if mean_name in row.index:
        mean_value = row.get(mean_name)
        std_value = row.get(std_name, 0.0)
    elif base_name in row.index:
        mean_value = row.get(base_name)
        std_value = 0.0
    else:
        return None, None
    if pd.isna(mean_value):
        return None, None
    return float(mean_value), 0.0 if pd.isna(std_value) else float(std_value)


def _format_mean_std(row: pd.Series, base_name: str, digits: int = 4) -> str:
    mean_value, std_value = _metric_value(row, base_name)
    if mean_value is None:
        return "NA"
    if std_value is None or abs(std_value) < 1e-12:
        return f"{mean_value:.{digits}f}"
    return f"{mean_value:.{digits}f} +/- {std_value:.{digits}f}"


def _format_integer_mean(row: pd.Series, base_name: str) -> str:
    mean_value, _ = _metric_value(row, base_name)
    if mean_value is None:
        return "NA"
    return f"{int(round(mean_value))}"


def _load_primary_row(csv_path: Path, model_name: str) -> pd.Series | None:
    if not csv_path.exists():
        return None
    frame = load_csv_table(csv_path)
    if "model_name" not in frame.columns:
        return None
    matched = frame.loc[frame["model_name"].astype(str).str.lower() == model_name]
    if matched.empty:
        return None
    return matched.iloc[0]


def _metrics_root_for_config(config: Dict[str, Any]) -> Path:
    return Path(config["repo_root"]) / config["outputs_root"] / "evaluation" / "metrics"


def _checkpoint_paths_for_config(config: Dict[str, Any], seeds: List[int]) -> List[Path]:
    repo_root = Path(config["repo_root"])
    paths = []
    for seed in seeds:
        seed_config = build_seed_run_config(config, seed=seed, multi_seed=len(seeds) > 1)
        paths.append(
            (
                repo_root
                / seed_config["outputs_root"]
                / "training"
                / "checkpoints"
                / seed_config["evaluation"]["checkpoint_name"]
            ).resolve()
        )
    return paths


@torch.no_grad() if TORCH_AVAILABLE else (lambda fn: fn)
def _measure_checkpoint_streaming_latency_ms(
    checkpoint_path: Path,
    warmup: int = 20,
    repeats: int = 100,
) -> float | None:
    if not TORCH_AVAILABLE or not checkpoint_path.exists():
        return None
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = dict(checkpoint.get("config") or {})
    if not config:
        return None

    model_config = {
        **dict(config.get("model", {})),
        "sampling_frequency": float(config.get("sampling_frequency", 40.0)),
    }
    model = build_model(
        model_name=str(config["model_name"]),
        input_dim=len(config["input_channels"]),
        output_dim=len(config["target_channels"]),
        model_config=model_config,
    ).cpu()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    if not hasattr(model, "forward_step"):
        return None

    sample = torch.randn(1, len(config["input_channels"]), dtype=torch.float32)
    stream_state = model.init_stream_state(batch_size=1, device=sample.device, dtype=sample.dtype)
    for _ in range(int(warmup)):
        step_outputs = model.forward_step(sample, stream_state=stream_state)
        stream_state = step_outputs.get("stream_state", stream_state)

    start = time.perf_counter()
    for _ in range(int(repeats)):
        step_outputs = model.forward_step(sample, stream_state=stream_state)
        stream_state = step_outputs.get("stream_state", stream_state)
    elapsed = time.perf_counter() - start
    return float(elapsed * 1000.0 / max(int(repeats), 1))


def _summarize_streaming_latency(
    checkpoint_paths: List[Path],
    warmup: int,
    repeats: int,
) -> tuple[float | None, float | None]:
    values = [
        value
        for value in (
            _measure_checkpoint_streaming_latency_ms(path, warmup=warmup, repeats=repeats)
            for path in checkpoint_paths
        )
        if value is not None
    ]
    if not values:
        return None, None
    series = pd.Series(values, dtype=float)
    return float(series.mean()), float(series.std(ddof=0))


def _format_latency(mean_value: float | None, std_value: float | None, digits: int = 4) -> str:
    if mean_value is None:
        return "NA"
    if std_value is None or abs(std_value) < 1e-12:
        return f"{mean_value:.{digits}f}"
    return f"{mean_value:.{digits}f} +/- {std_value:.{digits}f}"


def _loss_profile_label(loss_profile: str) -> str:
    profile = LOSS_PROFILES[loss_profile]
    return f"{loss_profile} ({profile['description']})"


def _build_summary_rows(
    settings: List[str],
    models: List[str],
    config_dir: Path,
    output_root: Path,
    seeds: List[int],
    loss_profile: str,
    streaming_warmup: int,
    streaming_repeats: int,
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for setting in settings:
        for model_name in models:
            config = _load_setting_config(
                model_name=model_name,
                config_dir=config_dir,
                output_root=output_root,
                setting=setting,
                loss_profile=loss_profile,
            )
            metrics_root = _metrics_root_for_config(config)
            comparison_path = metrics_root / "multiseed_model_comparison.csv"
            deployment_path = metrics_root / "multiseed_model_deployment_summary.csv"
            if not comparison_path.exists():
                comparison_path = metrics_root / "model_comparison.csv"
            if not deployment_path.exists():
                deployment_path = metrics_root / "model_deployment_summary.csv"

            metric_row = _load_primary_row(comparison_path, model_name)
            deployment_row = _load_primary_row(deployment_path, model_name)
            if metric_row is None:
                continue
            checkpoint_paths = _checkpoint_paths_for_config(config, seeds)
            streaming_mean, streaming_std = _summarize_streaming_latency(
                checkpoint_paths,
                warmup=streaming_warmup,
                repeats=streaming_repeats,
            )

            row = {
                "Setting": setting,
                "Model": MODEL_LABELS[model_name],
                "Causal type": CAUSAL_TYPES[model_name],
                "RMSE": _format_mean_std(metric_row, "rmse_mean"),
                "Pearson": _format_mean_std(metric_row, "pearson_mean"),
                "PSD Dist.": _format_mean_std(metric_row, "psd_distance_mean"),
                "HF Improve.": _format_mean_std(metric_row, "hf_ratio_improvement_mean"),
                "Parameters": "NA",
                "FP32 size (MB)": "NA",
                "CPU forward ms/window": "NA",
                "Streaming ms/step": _format_latency(streaming_mean, streaming_std),
                "Deployment verdict": "NA",
                "Loss profile": _loss_profile_label(loss_profile),
            }
            if deployment_row is not None:
                row.update(
                    {
                        "Parameters": _format_integer_mean(deployment_row, "parameter_count"),
                        "FP32 size (MB)": _format_mean_std(deployment_row, "parameter_size_mb_fp32"),
                        "CPU forward ms/window": _format_mean_std(deployment_row, "cpu_forward_ms_per_window"),
                        "Deployment verdict": str(deployment_row.get("embedded_deployment_verdict", "NA")),
                    }
                )
            rows.append(row)
    return rows


def _write_comparison_table(rows: List[Dict[str, Any]], output_root: Path) -> Dict[str, Path]:
    table_root = ensure_dir(output_root / "tables")
    frame = pd.DataFrame(rows)
    for column in CAUSAL_TABLE_COLUMNS:
        if column not in frame.columns:
            frame[column] = "NA"
    frame = frame[CAUSAL_TABLE_COLUMNS]

    csv_path = table_root / "causal_model_comparison_table.csv"
    markdown_path = table_root / "causal_model_comparison_table.md"
    json_path = table_root / "causal_model_comparison_table.json"
    frame.to_csv(csv_path, index=False)
    try:
        markdown_text = frame.to_markdown(index=False)
    except ImportError:
        markdown_text = _frame_to_markdown(frame)
    markdown_path.write_text(markdown_text, encoding="utf-8")
    json_path.write_text(
        json.dumps({"rows": frame.to_dict(orient="records")}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {"csv": csv_path, "markdown": markdown_path, "json": json_path}


def _frame_to_markdown(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    rows = [[str(value) for value in row] for row in frame.to_numpy()]
    widths = [
        max([len(headers[index])] + [len(row[index]) for row in rows])
        for index in range(len(headers))
    ]
    header_line = "| " + " | ".join(header.ljust(widths[index]) for index, header in enumerate(headers)) + " |"
    divider_line = "| " + " | ".join("-" * widths[index] for index in range(len(headers))) + " |"
    body_lines = [
        "| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(row)) + " |"
        for row in rows
    ]
    return "\n".join([header_line, divider_line, *body_lines])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train, evaluate, and summarize causal deep IMU compensation models.")
    parser.add_argument("--setting", choices=sorted(SETTING_SPECS), default="by_experiment")
    parser.add_argument("--settings", choices=sorted(SETTING_SPECS), nargs="*", default=None)
    parser.add_argument("--seeds", type=int, nargs="*", default=[42, 43, 44])
    parser.add_argument("--models", nargs="*", default=list(CAUSAL_MODEL_ORDER))
    parser.add_argument("--loss-profile", choices=sorted(LOSS_PROFILES), default="best_sensitivity")
    parser.add_argument("--config-dir", type=Path, default=Path("project/configs/causal_models"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/causal_model_comparison"))
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--streaming-warmup", type=int, default=20)
    parser.add_argument("--streaming-repeats", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    models = _resolve_models(args.models)
    settings = args.settings if args.settings else [args.setting]
    seeds = resolve_experiment_seeds({"seed": 42}, explicit_seeds=args.seeds)
    config_dir = args.config_dir.resolve()
    output_root = args.output_root

    if not args.summarize_only:
        for setting in settings:
            for model_name in models:
                config = _load_setting_config(
                    model_name=model_name,
                    config_dir=config_dir,
                    output_root=output_root,
                    setting=setting,
                    loss_profile=args.loss_profile,
                )
                print(
                    f"[{setting}] {model_name}: seeds={seeds}, "
                    f"outputs={config['outputs_root']}, processed={config['processed_root']}"
                )
                _train_and_evaluate_model(
                    config=config,
                    seeds=seeds,
                    skip_train=bool(args.skip_train),
                    skip_eval=bool(args.skip_eval),
                )

    if not args.skip_eval:
        rows = _build_summary_rows(
            settings=settings,
            models=models,
            config_dir=config_dir,
            output_root=output_root,
            seeds=seeds,
            loss_profile=args.loss_profile,
            streaming_warmup=int(args.streaming_warmup),
            streaming_repeats=int(args.streaming_repeats),
        )
        paths = _write_comparison_table(rows, Path(output_root))
        print(f"Causal model comparison table saved to {paths['csv']}")


if __name__ == "__main__":
    main()
