from __future__ import annotations

import math

from project.utils.torch_compat import TORCH_AVAILABLE, nn, require_torch, torch


if TORCH_AVAILABLE:  # pragma: no branch

    class SinusoidalPositionalEncoding(nn.Module):
        def __init__(self, model_dim: int, max_length: int = 4096):
            super().__init__()
            positions = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, model_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / max(model_dim, 1))
            )
            encoding = torch.zeros(max_length, model_dim, dtype=torch.float32)
            encoding[:, 0::2] = torch.sin(positions * div_term)
            encoding[:, 1::2] = torch.cos(positions * div_term[: encoding[:, 1::2].shape[1]])
            self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

        def forward(self, inputs):
            sequence_length = inputs.shape[1]
            return inputs + self.encoding[:, :sequence_length, :]


    class TransformerBaseline(nn.Module):
        """Transformer encoder residual baseline."""

        def __init__(
            self,
            input_dim: int,
            output_dim: int,
            model_dim: int = 128,
            num_layers: int = 3,
            num_heads: int = 4,
            feedforward_dim: int = 256,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.input_dim = int(input_dim)
            self.output_dim = int(output_dim)
            self.model_dim = int(model_dim)
            self.input_projection = nn.Linear(self.input_dim, self.model_dim)
            self.position_encoding = SinusoidalPositionalEncoding(self.model_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.model_dim,
                nhead=int(num_heads),
                dim_feedforward=int(feedforward_dim),
                dropout=float(dropout),
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(num_layers))
            self.residual_head = nn.Linear(self.model_dim, self.output_dim)
            self.base_projection = (
                nn.Linear(self.input_dim, self.output_dim) if self.input_dim != self.output_dim else None
            )

        def forward(self, inputs):
            if inputs.ndim != 3:
                raise ValueError(f"Expected `[B, T, C]` input, got {tuple(inputs.shape)}")
            embedded = self.input_projection(inputs)
            encoded = self.encoder(self.position_encoding(embedded))
            residual = self.residual_head(encoded)
            base_signal = self.base_projection(inputs) if self.base_projection is not None else inputs
            predictions = base_signal + residual
            return {
                "predictions": predictions,
                "residual": residual,
            }

else:

    class TransformerBaseline:  # pragma: no cover - runtime safeguard only
        def __init__(self, *args, **kwargs):
            require_torch()
