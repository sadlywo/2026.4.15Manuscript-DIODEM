from __future__ import annotations

from typing import Dict, List

from project.utils.torch_compat import TORCH_AVAILABLE, require_torch, torch


if TORCH_AVAILABLE:  # pragma: no branch

    def _unwrap_model_outputs(model_outputs):
        if isinstance(model_outputs, dict):
            predictions = model_outputs.get("predictions")
            if predictions is None:
                raise ValueError("Model output dict must contain a `predictions` tensor.")
            return predictions, model_outputs
        return model_outputs, {"predictions": model_outputs}


    def run_epoch(model, dataloader, optimizer, criterion, device, training: bool) -> Dict[str, float]:
        """Run one train or validation epoch and average all loss terms."""
        model.train(training)
        totals: Dict[str, float] = {}
        num_batches = 0

        for batch in dataloader:
            inputs = batch["inputs"].to(device)
            targets = batch["targets"].to(device)
            model_outputs = model(inputs)
            predictions, aux_outputs = _unwrap_model_outputs(model_outputs)
            loss_terms = criterion(predictions, targets, aux_outputs=aux_outputs)
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
        attach_states: List[torch.Tensor] = []
        attach_sequences: List[torch.Tensor] = []
        metadata = []
        for batch in dataloader:
            batch_inputs = batch["inputs"].to(device)
            batch_targets = batch["targets"].to(device)
            model_outputs = model(batch_inputs)
            batch_predictions, aux_outputs = _unwrap_model_outputs(model_outputs)
            predictions.append(batch_predictions.cpu())
            targets.append(batch_targets.cpu())
            inputs.append(batch_inputs.cpu())
            if "z_attach" in aux_outputs:
                attach_states.append(aux_outputs["z_attach"].detach().cpu())
            if "z_attach_sequence" in aux_outputs:
                attach_sequences.append(aux_outputs["z_attach_sequence"].detach().cpu())
            metadata.extend(batch["metadata"])
        if not predictions:
            return {
                "predictions": torch.empty(0),
                "targets": torch.empty(0),
                "inputs": torch.empty(0),
                "z_attach": torch.empty(0),
                "z_attach_sequence": torch.empty(0),
                "metadata": metadata,
            }
        return {
            "predictions": torch.cat(predictions, dim=0),
            "targets": torch.cat(targets, dim=0),
            "inputs": torch.cat(inputs, dim=0),
            "z_attach": torch.cat(attach_states, dim=0) if attach_states else torch.empty(0),
            "z_attach_sequence": torch.cat(attach_sequences, dim=0) if attach_sequences else torch.empty(0),
            "metadata": metadata,
        }

else:

    def run_epoch(*args, **kwargs):  # pragma: no cover - runtime safeguard only
        require_torch()


    def collect_predictions(*args, **kwargs):  # pragma: no cover - runtime safeguard only
        require_torch()
