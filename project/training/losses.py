from __future__ import annotations

from typing import Dict

from project.utils.torch_compat import TORCH_AVAILABLE, nn, require_torch, torch


if TORCH_AVAILABLE:  # pragma: no branch

    def derivative_loss(predictions, targets):
        pred_diff = predictions[:, 1:, :] - predictions[:, :-1, :]
        target_diff = targets[:, 1:, :] - targets[:, :-1, :]
        return torch.mean(torch.abs(pred_diff - target_diff))


    def spectral_loss(predictions, targets):
        pred_fft = torch.fft.rfft(predictions, dim=1)
        target_fft = torch.fft.rfft(targets, dim=1)
        return torch.mean(torch.abs(torch.abs(pred_fft) - torch.abs(target_fft)))


    def smoothness_regularization(predictions):
        pred_diff = predictions[:, 1:, :] - predictions[:, :-1, :]
        return torch.mean(pred_diff**2)


    class CompositeLoss(nn.Module):
        """Weighted combination of time, derivative, and spectral losses."""

        def __init__(self, weights: Dict[str, float]):
            super().__init__()
            self.weights = {key: float(value) for key, value in weights.items()}
            self.l1 = nn.L1Loss()
            self.mse = nn.MSELoss()

        def forward(self, predictions, targets):
            total = predictions.new_tensor(0.0)
            terms = {}

            if self.weights.get("l1", 0.0) > 0:
                terms["l1"] = self.l1(predictions, targets)
                total = total + self.weights["l1"] * terms["l1"]
            if self.weights.get("mse", 0.0) > 0:
                terms["mse"] = self.mse(predictions, targets)
                total = total + self.weights["mse"] * terms["mse"]
            if self.weights.get("derivative", 0.0) > 0:
                terms["derivative"] = derivative_loss(predictions, targets)
                total = total + self.weights["derivative"] * terms["derivative"]
            if self.weights.get("spectral", 0.0) > 0:
                terms["spectral"] = spectral_loss(predictions, targets)
                total = total + self.weights["spectral"] * terms["spectral"]
            if self.weights.get("smoothness", 0.0) > 0:
                terms["smoothness"] = smoothness_regularization(predictions)
                total = total + self.weights["smoothness"] * terms["smoothness"]

            terms["total"] = total
            return terms

else:

    class CompositeLoss:  # pragma: no cover - runtime safeguard only
        def __init__(self, *args, **kwargs):
            require_torch()

