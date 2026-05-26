from __future__ import annotations

from project.utils.torch_compat import TORCH_AVAILABLE, nn, require_torch, torch


if TORCH_AVAILABLE:  # pragma: no branch

    class LSTMBaseline(nn.Module):
        """Sequence-to-sequence LSTM residual baseline with causal step inference."""

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
            lstm_dropout = float(dropout) if self.num_layers > 1 else 0.0
            self.lstm = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=self.hidden_dim,
                num_layers=self.num_layers,
                dropout=lstm_dropout,
                batch_first=True,
            )
            self.residual_head = nn.Linear(self.hidden_dim, self.output_dim)
            self.base_projection = (
                nn.Linear(self.input_dim, self.output_dim) if self.input_dim != self.output_dim else None
            )

        def forward(self, inputs):
            if inputs.ndim != 3:
                raise ValueError(f"Expected `[B, T, C]` input, got {tuple(inputs.shape)}")
            sequence_features, _ = self.lstm(inputs)
            residual = self.residual_head(sequence_features)
            base_signal = self.base_projection(inputs) if self.base_projection is not None else inputs
            predictions = base_signal + residual
            return {
                "predictions": predictions,
                "residual": residual,
            }

        def init_stream_state(self, batch_size: int = 1, device=None, dtype=torch.float32):
            device = device if device is not None else next(self.parameters()).device
            hidden = torch.zeros(
                self.num_layers,
                int(batch_size),
                self.hidden_dim,
                device=device,
                dtype=dtype,
            )
            cell = torch.zeros_like(hidden)
            return {"hidden": hidden, "cell": cell}

        def forward_step(self, input_step, stream_state=None):
            if input_step.ndim != 2:
                raise ValueError(f"Expected `[B, C]` input step, got {tuple(input_step.shape)}")
            if stream_state is None:
                stream_state = self.init_stream_state(
                    batch_size=input_step.shape[0],
                    device=input_step.device,
                    dtype=input_step.dtype,
                )
            sequence_features, (hidden, cell) = self.lstm(
                input_step.unsqueeze(1),
                (stream_state["hidden"], stream_state["cell"]),
            )
            residual_step = self.residual_head(sequence_features[:, -1, :])
            base_step = self.base_projection(input_step) if self.base_projection is not None else input_step
            prediction_step = base_step + residual_step
            return {
                "prediction_step": prediction_step,
                "residual_step": residual_step,
                "stream_state": {"hidden": hidden.detach(), "cell": cell.detach()},
            }

else:

    class LSTMBaseline:  # pragma: no cover - runtime safeguard only
        def __init__(self, *args, **kwargs):
            require_torch()
