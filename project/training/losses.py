from __future__ import annotations

from typing import Dict

from project.utils.torch_compat import TORCH_AVAILABLE, nn, require_torch, torch


if TORCH_AVAILABLE:  # pragma: no branch

    def spectral_loss(predictions, targets):
        pred_fft = torch.fft.rfft(predictions, dim=1)
        target_fft = torch.fft.rfft(targets, dim=1)
        return torch.mean(torch.abs(torch.abs(pred_fft) - torch.abs(target_fft)))


    class CompositeLoss(nn.Module):
        """Weighted combination of time-domain and spectral reconstruction losses."""

        def __init__(self, weights: Dict[str, float]):
            super().__init__()
            self.weights = {key: float(value) for key, value in weights.items()}
            self.l1 = nn.L1Loss()
            self.mse = nn.MSELoss()

        def forward(self, predictions, targets, aux_outputs=None):
            total = predictions.new_tensor(0.0)
            terms = {}

            time_l1_weight = self.weights.get("time_l1", self.weights.get("l1", 0.0))
            if time_l1_weight > 0:
                terms["l1"] = self.l1(predictions, targets)
                total = total + time_l1_weight * terms["l1"]
            if self.weights.get("mse", 0.0) > 0:
                terms["mse"] = self.mse(predictions, targets)
                total = total + self.weights["mse"] * terms["mse"]
            if self.weights.get("spectral", 0.0) > 0:
                terms["spectral"] = spectral_loss(predictions, targets)
                total = total + self.weights["spectral"] * terms["spectral"]

            terms["total"] = total
            return terms

else:

    class CompositeLoss:  # pragma: no cover - runtime safeguard only
        def __init__(self, *args, **kwargs):
            require_torch()
