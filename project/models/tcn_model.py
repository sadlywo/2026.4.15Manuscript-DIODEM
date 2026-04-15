from __future__ import annotations

from project.utils.torch_compat import TORCH_AVAILABLE, nn, require_torch


if TORCH_AVAILABLE:  # pragma: no branch

    class TemporalBlock(nn.Module):
        """Dilated residual block used inside the TCN baseline."""

        def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
            super().__init__()
            padding = (kernel_size - 1) * dilation
            self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
            self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
            self.activation = nn.ReLU()
            self.dropout = nn.Dropout(dropout)

        def _crop(self, values, target_length: int):
            return values[..., :target_length]

        def forward(self, inputs):
            residual = inputs
            target_length = inputs.shape[-1]
            outputs = self.conv1(inputs)
            outputs = self._crop(outputs, target_length)
            outputs = self.activation(outputs)
            outputs = self.dropout(outputs)
            outputs = self.conv2(outputs)
            outputs = self._crop(outputs, target_length)
            outputs = self.activation(outputs)
            outputs = self.dropout(outputs)
            return outputs + residual


    class TCNBaseline(nn.Module):
        """A compact 1D temporal convolution baseline."""

        def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_dim: int = 64,
            num_layers: int = 4,
            kernel_size: int = 3,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.input_projection = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
            self.blocks = nn.ModuleList(
                [
                    TemporalBlock(hidden_dim, kernel_size, dilation=2**layer_idx, dropout=dropout)
                    for layer_idx in range(num_layers)
                ]
            )
            self.output_projection = nn.Conv1d(hidden_dim, output_dim, kernel_size=1)

        def forward(self, inputs):
            if inputs.ndim != 3:
                raise ValueError(f"Expected `[B, T, C]` input, got {tuple(inputs.shape)}")
            outputs = inputs.transpose(1, 2)
            outputs = self.input_projection(outputs)
            for block in self.blocks:
                outputs = block(outputs)
            outputs = self.output_projection(outputs)
            return outputs.transpose(1, 2)

else:

    class TCNBaseline:  # pragma: no cover - runtime safeguard only
        def __init__(self, *args, **kwargs):
            require_torch()

