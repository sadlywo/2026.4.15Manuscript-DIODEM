from __future__ import annotations

from project.utils.torch_compat import TORCH_AVAILABLE, nn, require_torch


if TORCH_AVAILABLE:  # pragma: no branch

    class MLPBaseline(nn.Module):
        """A small per-timestep MLP baseline."""

        def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )

        def forward(self, inputs):
            if inputs.ndim != 3:
                raise ValueError(f"Expected `[B, T, C]` input, got {tuple(inputs.shape)}")
            return self.network(inputs)

        def init_stream_state(self, batch_size: int = 1, device=None, dtype=None):
            return {}

        def forward_step(self, input_step, stream_state=None):
            if input_step.ndim != 2:
                raise ValueError(f"Expected `[B, C]` input step, got {tuple(input_step.shape)}")
            prediction_step = self.network(input_step)
            return {
                "prediction_step": prediction_step,
                "residual_step": prediction_step - input_step if prediction_step.shape == input_step.shape else prediction_step,
                "stream_state": {} if stream_state is None else stream_state,
            }

else:

    class MLPBaseline:  # pragma: no cover - runtime safeguard only
        def __init__(self, *args, **kwargs):
            require_torch()
