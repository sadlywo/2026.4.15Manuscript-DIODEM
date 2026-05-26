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
            causal: bool = False,
            stream_history: int = 4096,
        ):
            super().__init__()
            self.input_dim = int(input_dim)
            self.output_dim = int(output_dim)
            self.model_dim = int(model_dim)
            self.causal = bool(causal)
            self.stream_history = int(stream_history)
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
            attention_mask = self._causal_mask(inputs.shape[1], inputs.device) if self.causal else None
            encoded = self.encoder(self.position_encoding(embedded), mask=attention_mask)
            residual = self.residual_head(encoded)
            base_signal = self.base_projection(inputs) if self.base_projection is not None else inputs
            predictions = base_signal + residual
            return {
                "predictions": predictions,
                "residual": residual,
            }

        def _causal_mask(self, sequence_length: int, device):
            mask = torch.full((int(sequence_length), int(sequence_length)), float("-inf"), device=device)
            return torch.triu(mask, diagonal=1)

        def init_stream_state(self, batch_size: int = 1, device=None, dtype=torch.float32):
            if not self.causal:
                raise RuntimeError("Streaming inference is only available when TransformerBaseline is causal.")
            device = device if device is not None else next(self.parameters()).device
            return {
                "history": torch.empty(int(batch_size), 0, self.input_dim, device=device, dtype=dtype),
            }

        def forward_step(self, input_step, stream_state=None):
            if not self.causal:
                raise RuntimeError("forward_step is only available when TransformerBaseline is causal.")
            if input_step.ndim != 2:
                raise ValueError(f"Expected `[B, C]` input step, got {tuple(input_step.shape)}")
            if stream_state is None:
                stream_state = self.init_stream_state(
                    batch_size=input_step.shape[0],
                    device=input_step.device,
                    dtype=input_step.dtype,
                )
            history = torch.cat([stream_state["history"], input_step.unsqueeze(1)], dim=1)
            if self.stream_history > 0 and history.shape[1] > self.stream_history:
                history = history[:, -self.stream_history :, :]
            outputs = self.forward(history)
            prediction_step = outputs["predictions"][:, -1, :]
            residual_step = outputs["residual"][:, -1, :]
            return {
                "prediction_step": prediction_step,
                "residual_step": residual_step,
                "stream_state": {"history": history.detach()},
            }

else:

    class TransformerBaseline:  # pragma: no cover - runtime safeguard only
        def __init__(self, *args, **kwargs):
            require_torch()
