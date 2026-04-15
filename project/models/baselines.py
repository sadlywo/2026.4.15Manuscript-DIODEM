from __future__ import annotations

from typing import Dict

from project.models.mlp_model import MLPBaseline
from project.models.tcn_model import TCNBaseline
from project.utils.torch_compat import TORCH_AVAILABLE, nn, require_torch


if TORCH_AVAILABLE:  # pragma: no branch

    class IdentityBaseline(nn.Module):
        """Return the input unchanged."""

        def forward(self, inputs):
            if inputs.ndim != 3:
                raise ValueError(f"Expected `[B, T, C]` input, got {tuple(inputs.shape)}")
            return inputs


    class LowPassBaseline(nn.Module):
        """Simple moving-average low-pass smoothing baseline."""

        def __init__(self, channels: int, kernel_size: int = 5):
            super().__init__()
            self.kernel_size = int(kernel_size)
            self.pad = self.kernel_size // 2
            self.pool = nn.AvgPool1d(kernel_size=self.kernel_size, stride=1)
            self.channels = channels

        def forward(self, inputs):
            if inputs.ndim != 3:
                raise ValueError(f"Expected `[B, T, C]` input, got {tuple(inputs.shape)}")
            outputs = inputs.transpose(1, 2)
            outputs = nn.functional.pad(outputs, (self.pad, self.pad), mode="replicate")
            outputs = self.pool(outputs)
            return outputs.transpose(1, 2)


    class LinearProjectionBaseline(nn.Module):
        """Per-timestep linear projection from input channels to target channels."""

        def __init__(self, input_dim: int, output_dim: int):
            super().__init__()
            self.projection = nn.Linear(input_dim, output_dim)

        def forward(self, inputs):
            if inputs.ndim != 3:
                raise ValueError(f"Expected `[B, T, C]` input, got {tuple(inputs.shape)}")
            return self.projection(inputs)

else:

    class IdentityBaseline:  # pragma: no cover - runtime safeguard only
        def __init__(self, *args, **kwargs):
            require_torch()


    class LowPassBaseline:  # pragma: no cover - runtime safeguard only
        def __init__(self, *args, **kwargs):
            require_torch()


    class LinearProjectionBaseline:  # pragma: no cover - runtime safeguard only
        def __init__(self, *args, **kwargs):
            require_torch()


AVAILABLE_MODELS = {
    "identity": IdentityBaseline,
    "lowpass": LowPassBaseline,
    "linear": LinearProjectionBaseline,
    "mlp": MLPBaseline,
    "tcn": TCNBaseline,
}


def build_model(model_name: str, input_dim: int, output_dim: int, model_config: Dict[str, float]):
    """Instantiate a baseline model by name."""
    name = model_name.lower()
    if name not in AVAILABLE_MODELS:
        raise ValueError(f"Unsupported model: {model_name}")

    if name == "identity":
        return IdentityBaseline()
    if name == "lowpass":
        return LowPassBaseline(channels=input_dim, kernel_size=int(model_config.get("kernel_size", 5)))
    if name == "linear":
        return LinearProjectionBaseline(input_dim=input_dim, output_dim=output_dim)
    if name == "mlp":
        return MLPBaseline(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=int(model_config.get("mlp_hidden_dim", 128)),
            dropout=float(model_config.get("dropout", 0.1)),
        )
    return TCNBaseline(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=int(model_config.get("hidden_dim", 64)),
        num_layers=int(model_config.get("num_layers", 4)),
        kernel_size=int(model_config.get("kernel_size", 3)),
        dropout=float(model_config.get("dropout", 0.1)),
    )

