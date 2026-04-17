from __future__ import annotations

from project.utils.torch_compat import TORCH_AVAILABLE, nn, require_torch


if TORCH_AVAILABLE:  # pragma: no branch

    class GRUBaseline(nn.Module):
        """Sequence-to-sequence GRU residual baseline."""

        def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_dim: int = 128,
            num_layers: int = 2,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.input_dim = int(input_dim)
            self.output_dim = int(output_dim)
            self.hidden_dim = int(hidden_dim)
            self.num_layers = int(num_layers)
            gru_dropout = float(dropout) if self.num_layers > 1 else 0.0
            self.gru = nn.GRU(
                input_size=self.input_dim,
                hidden_size=self.hidden_dim,
                num_layers=self.num_layers,
                dropout=gru_dropout,
                batch_first=True,
            )
            self.residual_head = nn.Linear(self.hidden_dim, self.output_dim)
            self.base_projection = (
                nn.Linear(self.input_dim, self.output_dim) if self.input_dim != self.output_dim else None
            )

        def forward(self, inputs):
            if inputs.ndim != 3:
                raise ValueError(f"Expected `[B, T, C]` input, got {tuple(inputs.shape)}")
            sequence_features, _ = self.gru(inputs)
            residual = self.residual_head(sequence_features)
            base_signal = self.base_projection(inputs) if self.base_projection is not None else inputs
            predictions = base_signal + residual
            return {
                "predictions": predictions,
                "residual": residual,
            }

else:

    class GRUBaseline:  # pragma: no cover - runtime safeguard only
        def __init__(self, *args, **kwargs):
            require_torch()
