from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt

from project.data.window_dataset import WindowedPairDataset, windowed_pair_collate
from project.models import build_model
from project.training.engine import run_epoch
from project.training.losses import CompositeLoss
from project.utils.io import ensure_dir, load_json
from project.utils.logger import get_logger
from project.utils.seed import seed_everything
from project.utils.torch_compat import TORCH_AVAILABLE, require_torch, torch


if TORCH_AVAILABLE:  # pragma: no branch

    def _resolve_device(device_name: str):
        if device_name == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")


    def _build_dataloader(dataset, batch_size: int, shuffle: bool, num_workers: int):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=windowed_pair_collate,
        )


    def train_model(config: Dict[str, Any], processed_root: Path, output_root: Path) -> Path:
        """Train one baseline model and save checkpoints plus curves."""
        require_torch()
        ensure_dir(output_root)
        ensure_dir(output_root / "checkpoints")
        logger = get_logger("diomed-train", output_root / "train.log")

        seed_everything(int(config["seed"]))
        device = _resolve_device(str(config["device"]).lower())
        stats = load_json(processed_root / "normalization_stats.json")
        normalization = config.get("normalization", "none")

        train_dataset = WindowedPairDataset(
            processed_root / "train_samples.pkl",
            normalization=normalization,
            normalization_stats=stats,
        )
        val_dataset = WindowedPairDataset(
            processed_root / "val_samples.pkl",
            normalization=normalization,
            normalization_stats=stats,
        )

        model = build_model(
            model_name=config["model_name"],
            input_dim=len(config["input_channels"]),
            output_dim=len(config["target_channels"]),
            model_config=dict(config.get("model", {})),
        ).to(device)
        criterion = CompositeLoss(dict(config["loss_weights"]))
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(config["learning_rate"]),
            weight_decay=float(config["weight_decay"]),
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2,
        )

        train_loader = _build_dataloader(
            train_dataset,
            batch_size=int(config["batch_size"]),
            shuffle=True,
            num_workers=int(config.get("num_workers", 0)),
        )
        val_loader = _build_dataloader(
            val_dataset,
            batch_size=int(config["batch_size"]),
            shuffle=False,
            num_workers=int(config.get("num_workers", 0)),
        )

        history = {"train_total": [], "val_total": []}
        best_val = float("inf")
        patience_counter = 0
        best_path = output_root / "checkpoints" / str(config["evaluation"]["checkpoint_name"])

        for epoch in range(int(config["num_epochs"])):
            train_losses = run_epoch(model, train_loader, optimizer, criterion, device, training=True)
            val_losses = run_epoch(model, val_loader, optimizer, criterion, device, training=False)
            scheduler.step(val_losses["total"])

            history["train_total"].append(train_losses["total"])
            history["val_total"].append(val_losses["total"])
            logger.info(
                "Epoch %d | train_total=%.6f | val_total=%.6f",
                epoch + 1,
                train_losses["total"],
                val_losses["total"],
            )

            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
                "history": history,
            }
            torch.save(checkpoint, output_root / "checkpoints" / "last.pt")

            if val_losses["total"] < best_val:
                best_val = val_losses["total"]
                patience_counter = 0
                torch.save(checkpoint, best_path)
            else:
                patience_counter += 1
                if patience_counter >= int(config["patience"]):
                    logger.info("Early stopping triggered at epoch %d.", epoch + 1)
                    break

        _save_loss_curve(history, output_root / "train_val_loss_curve.png")
        return best_path


    def _save_loss_curve(history: Dict[str, list], path: Path) -> None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(history["train_total"], label="train_total")
        ax.plot(history["val_total"], label="val_total")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training and validation loss")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)

else:

    def train_model(*args, **kwargs):  # pragma: no cover - runtime safeguard only
        require_torch()

