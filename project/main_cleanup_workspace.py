from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from project.utils.io import ensure_dir


CACHE_DIR_NAMES = {"__pycache__", ".ipynb_checkpoints", ".pytest_cache"}
TEMP_FILE_PATTERNS = ("*.tmp", "*.bak")
REPRESENTATIVE_FIGURE_FILES = (
    "figure1_representative_imu_signals.png",
    "figure1_representative_imu_signals.svg",
    "figure1_representative_imu_signals.pdf",
    "figure1_signal_error_metrics.csv",
    "plot_imu_figure.py",
    "原始绘图数据.xlsx",
)
LEGACY_OUTPUT_DIRS = (
    "outputs/supervised_multiseed_default",
    "outputs/supervised",
    "outputs/supervised_gru",
    "outputs/supervised_transformer",
    "outputs/figures",
    "outputs/tables",
)
REGENERABLE_PROCESSED_DIRS = (
    "processed_multiseed_default",
    "processed",
)


def _is_within_directory(path: Path, directory: Path) -> bool:
    resolved_path = path.resolve(strict=False)
    resolved_directory = directory.resolve(strict=False)
    try:
        resolved_path.relative_to(resolved_directory)
    except ValueError:
        return False
    return True


def _append_action(actions: List[Dict[str, Any]], action: str, source: Path, target: Path | None = None) -> None:
    record: Dict[str, Any] = {"action": action, "source": str(source)}
    if target is not None:
        record["target"] = str(target)
    actions.append(record)


def _safe_remove(path: Path, root: Path, actions: List[Dict[str, Any]], dry_run: bool) -> bool:
    if not path.exists():
        return False
    if not _is_within_directory(path, root):
        raise ValueError(f"Refusing to remove path outside repository root: {path}")
    _append_action(actions, "remove", path)
    if dry_run:
        return True
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    return True


def _move_if_exists(source: Path, target: Path, root: Path, dry_run: bool) -> bool:
    if not source.exists():
        return False
    if not _is_within_directory(source, root):
        raise ValueError(f"Refusing to move source outside repository root: {source}")
    if not _is_within_directory(target, root):
        raise ValueError(f"Refusing to move target outside repository root: {target}")
    if dry_run:
        return True
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target = target.with_name(f"{target.stem}_{timestamp}{target.suffix}")
    shutil.move(str(source), str(target))
    return True


def _archive_directory(source: Path, archive_root: Path, root: Path, actions: List[Dict[str, Any]], dry_run: bool) -> bool:
    if not source.exists():
        return False
    target = archive_root / source.name
    if target.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target = archive_root / f"{source.name}_{timestamp}"
    _append_action(actions, "archive", source, target)
    return _move_if_exists(source, target, root=root, dry_run=dry_run)


def _move_representative_files(root: Path, actions: List[Dict[str, Any]], dry_run: bool) -> None:
    source_dir = root / "mnt" / "data"
    target_dir = root / "docs" / "figures" / "source" / "representative_imu_signals"
    for file_name in REPRESENTATIVE_FIGURE_FILES:
        source = source_dir / file_name
        target = target_dir / file_name
        if source.exists():
            _append_action(actions, "move", source, target)
            _move_if_exists(source, target, root=root, dry_run=dry_run)

    if not dry_run:
        for candidate in (source_dir, root / "mnt"):
            if candidate.exists() and candidate.is_dir() and not any(candidate.iterdir()):
                _safe_remove(candidate, root=root, actions=actions, dry_run=False)


def cleanup_conservative(root: Path, dry_run: bool = False) -> List[Dict[str, Any]]:
    root = root.resolve(strict=False)
    actions: List[Dict[str, Any]] = []

    for directory in sorted(root.rglob("*")):
        if directory.is_dir() and directory.name in CACHE_DIR_NAMES:
            _safe_remove(directory, root=root, actions=actions, dry_run=dry_run)

    for pattern in TEMP_FILE_PATTERNS:
        for file_path in sorted(root.rglob(pattern)):
            if file_path.is_file():
                _safe_remove(file_path, root=root, actions=actions, dry_run=dry_run)

    _move_representative_files(root=root, actions=actions, dry_run=dry_run)

    archive_root = root / "outputs" / "archive_legacy"
    if any((root / path).exists() for path in LEGACY_OUTPUT_DIRS):
        _append_action(actions, "ensure_dir", archive_root)
        if not dry_run:
            ensure_dir(archive_root)
    for relative_path in LEGACY_OUTPUT_DIRS:
        _archive_directory(root / relative_path, archive_root=archive_root, root=root, actions=actions, dry_run=dry_run)

    for relative_path in REGENERABLE_PROCESSED_DIRS:
        _safe_remove(root / relative_path, root=root, actions=actions, dry_run=dry_run)

    return actions


def _write_report(root: Path, actions: List[Dict[str, Any]], dry_run: bool) -> Path:
    report_root = ensure_dir(root / "outputs")
    report_path = report_root / "cleanup_report.json"
    report = {
        "dry_run": bool(dry_run),
        "num_actions": len(actions),
        "actions": actions,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Conservatively clean derived files and archive legacy outputs.")
    parser.add_argument("--mode", choices=["conservative"], default="conservative")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    root = args.root.resolve(strict=False)
    if args.mode != "conservative":
        raise ValueError(f"Unsupported cleanup mode: {args.mode}")
    actions = cleanup_conservative(root=root, dry_run=bool(args.dry_run))
    report_path = _write_report(root=root, actions=actions, dry_run=bool(args.dry_run))
    print(f"Cleanup mode: {args.mode}")
    print(f"Dry run: {bool(args.dry_run)}")
    print(f"Planned/performed actions: {len(actions)}")
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
