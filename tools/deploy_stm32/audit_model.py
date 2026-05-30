from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from deploy_common import (
    DEFAULT_CHECKPOINT,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PROCESSED_ROOT,
    import_torch,
    iter_checkpoint_candidates,
    load_json,
    repo_root_from_file,
    save_json,
)


def _file_entry(path: Path) -> Dict[str, Any]:
    path = Path(path)
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": int(path.stat().st_size) if path.exists() else None,
    }


def _safe_json(path: Path) -> Dict[str, Any] | None:
    try:
        return load_json(path)
    except Exception:
        return None


def _summarize_stats(stats: Dict[str, Any] | None, path: Path) -> Dict[str, Any]:
    if not stats:
        return {"path": str(path), "exists": path.exists(), "mode": None}
    return {
        "path": str(path),
        "exists": True,
        "mode": stats.get("mode"),
        "input_mean": stats.get("input_mean"),
        "input_std": stats.get("input_std"),
        "target_mean": stats.get("target_mean"),
        "target_std": stats.get("target_std"),
    }


def _deployment_summary(repo_root: Path) -> Dict[str, Any] | None:
    path = (
        repo_root
        / "outputs"
        / "supervised_tcn_causal_by_experiment"
        / "seed_runs"
        / "seed_0042"
        / "evaluation"
        / "metrics"
        / "model_deployment_summary.json"
    )
    return _safe_json(path)


def build_audit_report(
    repo_root: Path | None = None,
    checkpoint: Path | None = None,
    processed_root: Path | None = None,
    try_torch: bool = True,
) -> Dict[str, Any]:
    repo_root = Path(repo_root or repo_root_from_file()).resolve()
    checkpoint = Path(checkpoint or (repo_root / DEFAULT_CHECKPOINT)).resolve()
    processed_root = Path(processed_root or (repo_root / DEFAULT_PROCESSED_ROOT)).resolve()
    stats_path = processed_root / "normalization_stats.json"
    stats = _safe_json(stats_path)

    report: Dict[str, Any] = {
        "repo_root": str(repo_root),
        "checkpoint": _file_entry(checkpoint),
        "processed_root": _file_entry(processed_root),
        "normalization": _summarize_stats(stats, stats_path),
        "configs": {
            "primary": _file_entry(repo_root / "project" / "configs" / "tcn_causal.yaml"),
            "causal_model": _file_entry(repo_root / "project" / "configs" / "causal_models" / "tcn_causal.yaml"),
        },
        "data": {
            "train_samples": _file_entry(processed_root / "train_samples.pkl"),
            "val_samples": _file_entry(processed_root / "val_samples.pkl"),
            "test_samples": _file_entry(processed_root / "test_samples.pkl"),
            "pair_table": _file_entry(processed_root / "pair_table.csv"),
        },
        "discovered_checkpoints": [str(path) for path in list(iter_checkpoint_candidates(repo_root))[:50]],
        "torch": {"attempted": bool(try_torch), "available": None, "error": None},
        "model": {},
        "deployment_summary": _deployment_summary(repo_root),
    }

    if not try_torch:
        return report

    torch, torch_error = import_torch()
    report["torch"]["available"] = torch is not None
    report["torch"]["error"] = torch_error
    if torch is None:
        return report

    try:
        from deploy_common import load_checkpoint_and_model

        _, checkpoint_data, config, model = load_checkpoint_and_model(checkpoint, device_name="cpu")
        param_count = int(sum(parameter.numel() for parameter in model.parameters()))
        param_bytes = int(sum(parameter.numel() * parameter.element_size() for parameter in model.parameters()))
        report["model"] = {
            "checkpoint_epoch": checkpoint_data.get("epoch"),
            "model_name": config.get("model_name"),
            "window_size": config.get("window_size"),
            "stride": config.get("stride"),
            "sampling_frequency": config.get("sampling_frequency"),
            "input_channels": config.get("input_channels"),
            "target_channels": config.get("target_channels"),
            "normalization": config.get("normalization"),
            "model_config": config.get("model"),
            "parameter_count": param_count,
            "parameter_size_mb_fp32": param_bytes / (1024.0 * 1024.0),
            "receptive_field": int(getattr(model, "receptive_field", 0)),
        }
    except Exception as exc:  # pragma: no cover - depends on local runtime/checkpoint
        report["torch"]["checkpoint_load_error"] = f"{type(exc).__name__}: {exc}"
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the causal TCN checkpoint and deployment inputs.")
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_file())
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--processed-root", type=Path, default=None)
    parser.add_argument("--skip-torch", action="store_true", help="Do not import torch or load the checkpoint.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "model_audit_report.json",
        help="Path for the JSON audit report.",
    )
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    checkpoint = args.checkpoint or (repo_root / DEFAULT_CHECKPOINT)
    processed_root = args.processed_root or (repo_root / DEFAULT_PROCESSED_ROOT)
    report = build_audit_report(
        repo_root=repo_root,
        checkpoint=checkpoint,
        processed_root=processed_root,
        try_torch=not args.skip_torch,
    )
    output = args.output if args.output.is_absolute() else repo_root / args.output
    save_json(report, output)
    print(f"Saved audit report to {output}")


if __name__ == "__main__":
    main()
