from __future__ import annotations

from typing import Dict, List

from project.utils.torch_compat import TORCH_AVAILABLE, require_torch, torch


if TORCH_AVAILABLE:  # pragma: no branch

    def run_epoch(model, dataloader, optimizer, criterion, device, training: bool) -> Dict[str, float]:
        """Run one train or validation epoch and average all loss terms."""
        model.train(training)
        totals: Dict[str, float] = {}
        num_batches = 0

        for batch in dataloader:
            inputs = batch["inputs"].to(device)
            targets = batch["targets"].to(device)
            predictions = model(inputs)
            loss_terms = criterion(predictions, targets)
            loss = loss_terms["total"]

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            for name, value in loss_terms.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach().cpu())
            num_batches += 1

        if num_batches == 0:
            return {"total": 0.0}
        return {name: value / num_batches for name, value in totals.items()}


    @torch.no_grad()
    def collect_predictions(model, dataloader, device):
        """Collect denormalized predictions for evaluation."""
        model.eval()
        predictions: List[torch.Tensor] = []
        targets: List[torch.Tensor] = []
        inputs: List[torch.Tensor] = []
        metadata = []
        for batch in dataloader:
            batch_inputs = batch["inputs"].to(device)
            batch_targets = batch["targets"].to(device)
            batch_predictions = model(batch_inputs)
            predictions.append(batch_predictions.cpu())
            targets.append(batch_targets.cpu())
            inputs.append(batch_inputs.cpu())
            metadata.extend(batch["metadata"])
        if not predictions:
            return {
                "predictions": torch.empty(0),
                "targets": torch.empty(0),
                "inputs": torch.empty(0),
                "metadata": metadata,
            }
        return {
            "predictions": torch.cat(predictions, dim=0),
            "targets": torch.cat(targets, dim=0),
            "inputs": torch.cat(inputs, dim=0),
            "metadata": metadata,
        }

else:

    def run_epoch(*args, **kwargs):  # pragma: no cover - runtime safeguard only
        require_torch()


    def collect_predictions(*args, **kwargs):  # pragma: no cover - runtime safeguard only
        require_torch()

