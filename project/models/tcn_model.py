from __future__ import annotations

from project.utils.torch_compat import TORCH_AVAILABLE, nn, require_torch, torch


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


    class AttachmentStateEncoder(nn.Module):
        """Infer a latent attachment state from temporal backbone features."""

        def __init__(self, hidden_dim: int, latent_dim: int):
            super().__init__()
            self.latent_projection = nn.Conv1d(hidden_dim, latent_dim, kernel_size=1)

        def forward(self, features):
            latent_sequence = self.latent_projection(features).transpose(1, 2)
            latent_summary = latent_sequence.mean(dim=1)
            return latent_sequence, latent_summary


    class TCNBaseline(nn.Module):
        """Attachment-aware residual TCN for artifact compensation."""

        def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_dim: int = 64,
            num_layers: int = 4,
            kernel_size: int = 3,
            dropout: float = 0.1,
            attach_latent_dim: int = 8,
        ):
            super().__init__()
            self.input_dim = int(input_dim)
            self.output_dim = int(output_dim)
            self.attach_latent_dim = int(attach_latent_dim)
            self.use_attachment_latent = self.attach_latent_dim > 0
            self.input_projection = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
            self.blocks = nn.ModuleList(
                [
                    TemporalBlock(hidden_dim, kernel_size, dilation=2**layer_idx, dropout=dropout)
                    for layer_idx in range(num_layers)
                ]
            )
            self.attachment_encoder = None
            self.feature_gate = None
            self.feature_shift = None
            if self.use_attachment_latent:
                self.attachment_encoder = AttachmentStateEncoder(hidden_dim, self.attach_latent_dim)
                self.feature_gate = nn.Conv1d(self.attach_latent_dim, hidden_dim, kernel_size=1)
                self.feature_shift = nn.Conv1d(self.attach_latent_dim, hidden_dim, kernel_size=1)
            self.residual_projection = nn.Conv1d(hidden_dim, output_dim, kernel_size=1)
            self.base_projection = None
            if self.input_dim != self.output_dim:
                self.base_projection = nn.Conv1d(input_dim, output_dim, kernel_size=1)

        def forward(self, inputs):
            if inputs.ndim != 3:
                raise ValueError(f"Expected `[B, T, C]` input, got {tuple(inputs.shape)}")
            base = inputs.transpose(1, 2)
            outputs = self.input_projection(base)
            for block in self.blocks:
                outputs = block(outputs)

            aux_outputs = {}
            conditioned = outputs
            if self.use_attachment_latent:
                latent_sequence, latent_summary = self.attachment_encoder(outputs)
                latent_features = latent_sequence.transpose(1, 2)
                gate = torch.sigmoid(self.feature_gate(latent_features))
                shift = self.feature_shift(latent_features)
                conditioned = outputs * (1.0 + gate) + shift
                aux_outputs["z_attach"] = latent_summary
                aux_outputs["z_attach_sequence"] = latent_sequence

            residual = self.residual_projection(conditioned)
            if self.base_projection is not None:
                base_signal = self.base_projection(base)
            else:
                base_signal = base
            predictions = (base_signal + residual).transpose(1, 2)
            outputs = {
                "predictions": predictions,
                "residual": residual.transpose(1, 2),
            }
            outputs.update(aux_outputs)
            return outputs

else:

    class TCNBaseline:  # pragma: no cover - runtime safeguard only
        def __init__(self, *args, **kwargs):
            require_torch()
