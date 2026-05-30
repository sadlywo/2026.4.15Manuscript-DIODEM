from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np


DEFAULT_CHECKPOINT = Path(
    "outputs/supervised_tcn_causal_by_experiment/seed_runs/seed_0042/training/checkpoints/best.pt"
)
DEFAULT_PROCESSED_ROOT = Path("processed_by_experiment")
DEFAULT_OUTPUT_ROOT = Path("deploy_stm32")
INPUT_CHANNELS = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]
PREDICTION_CHANNELS = [
    "pred_acc_x",
    "pred_acc_y",
    "pred_acc_z",
    "pred_gyr_x",
    "pred_gyr_y",
    "pred_gyr_z",
]


def repo_root_from_file(path: Path | None = None) -> Path:
    current = (path or Path(__file__)).resolve()
    for parent in [current, *current.parents]:
        if (parent / "project").is_dir() and (parent / "README.md").exists():
            return parent
    return Path.cwd().resolve()


def ensure_repo_on_path(repo_root: Path) -> None:
    repo_text = str(repo_root.resolve())
    if repo_text not in sys.path:
        sys.path.insert(0, repo_text)


def load_json(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(data: Dict[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def load_pickle(path: Path) -> Any:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def _stat_array(stats: Dict[str, Any], key: str) -> np.ndarray:
    return np.asarray(stats[key], dtype=np.float32)


def normalize_inputs(values: np.ndarray, stats: Dict[str, Any]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if str(stats.get("mode", "none")) == "none":
        return array.astype(np.float32)
    mean = _stat_array(stats, "input_mean")
    std = _stat_array(stats, "input_std")
    return ((array - mean) / std).astype(np.float32)


def denormalize_outputs(values: np.ndarray, stats: Dict[str, Any]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if str(stats.get("mode", "none")) == "none":
        return array.astype(np.float32)
    mean = _stat_array(stats, "target_mean")
    std = _stat_array(stats, "target_std")
    return (array * std + mean).astype(np.float32)


def import_torch() -> Tuple[Any | None, str | None]:
    try:
        import torch  # type: ignore

        return torch, None
    except Exception as exc:  # pragma: no cover - depends on local runtime
        return None, f"{type(exc).__name__}: {exc}"


def iter_checkpoint_candidates(repo_root: Path) -> Iterable[Path]:
    preferred = repo_root / DEFAULT_CHECKPOINT
    if preferred.exists():
        yield preferred
    outputs_root = repo_root / "outputs"
    if not outputs_root.exists():
        return
    yielded = {preferred.resolve()} if preferred.exists() else set()
    for path in sorted(outputs_root.rglob("*.pt")):
        resolved = path.resolve()
        if resolved in yielded:
            continue
        yielded.add(resolved)
        yield path


def load_checkpoint_and_model(checkpoint_path: Path, device_name: str = "cpu"):
    repo_root = repo_root_from_file()
    ensure_repo_on_path(repo_root)
    torch, torch_error = import_torch()
    if torch is None:
        raise RuntimeError(f"PyTorch could not be imported. {torch_error}")

    from project.models import build_model

    checkpoint_path = Path(checkpoint_path).resolve()
    checkpoint = torch.load(checkpoint_path, map_location=device_name)
    config = dict(checkpoint.get("config") or {})
    if not config:
        raise ValueError(f"Checkpoint {checkpoint_path} does not contain a training config.")
    model_config = {
        **dict(config.get("model", {})),
        "sampling_frequency": float(config.get("sampling_frequency", 40.0)),
    }
    model = build_model(
        model_name=str(config["model_name"]),
        input_dim=len(config["input_channels"]),
        output_dim=len(config["target_channels"]),
        model_config=model_config,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device_name).eval()
    return torch, checkpoint, config, model


def load_test_windows(processed_root: Path, split: str = "test", max_windows: int = 16) -> np.ndarray:
    bundle = load_pickle(Path(processed_root) / f"{split}_samples.pkl")
    inputs = np.asarray(bundle["inputs"], dtype=np.float32)
    if inputs.ndim != 3:
        raise ValueError(f"Expected cached `[N, T, C]` inputs, got {inputs.shape}")
    return inputs[: int(max_windows)].astype(np.float32)

