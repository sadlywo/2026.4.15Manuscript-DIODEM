from __future__ import annotations

from typing import Any, Dict

from project.utils.torch_compat import TORCH_AVAILABLE, nn, require_torch, torch


if TORCH_AVAILABLE:  # pragma: no branch

    class TemporalBlock(nn.Module):
        """Dilated residual block used inside the TCN baseline."""

        def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float, causal: bool = False):
            super().__init__()
            self.causal = bool(causal)
            self.kernel_size = int(kernel_size)
            self.dilation = int(dilation)
            self.padding = (self.kernel_size - 1) * self.dilation
            conv_padding = 0 if self.causal else self.padding
            self.conv1 = nn.Conv1d(channels, channels, self.kernel_size, padding=conv_padding, dilation=self.dilation)
            self.conv2 = nn.Conv1d(channels, channels, self.kernel_size, padding=conv_padding, dilation=self.dilation)
            self.activation = nn.ReLU()
            self.dropout = nn.Dropout(dropout)

        def _crop(self, values, target_length: int):
            return values[..., :target_length]

        def _apply_conv(self, convolution, values, target_length: int):
            if self.causal and self.padding > 0:
                values = nn.functional.pad(values, (self.padding, 0))
            outputs = convolution(values)
            if not self.causal:
                outputs = self._crop(outputs, target_length)
            return outputs

        def forward(self, inputs):
            residual = inputs
            target_length = inputs.shape[-1]
            outputs = self._apply_conv(self.conv1, inputs, target_length)
            outputs = self.activation(outputs)
            outputs = self.dropout(outputs)
            outputs = self._apply_conv(self.conv2, outputs, target_length)
            outputs = self.activation(outputs)
            outputs = self.dropout(outputs)
            return outputs + residual

        def init_stream_state(self, batch_size: int, channels: int, device, dtype) -> Dict[str, torch.Tensor]:
            history_length = self.padding if self.causal else 0
            return {
                "conv1_history": torch.zeros(batch_size, channels, history_length, device=device, dtype=dtype),
                "conv2_history": torch.zeros(batch_size, channels, history_length, device=device, dtype=dtype),
            }

        def _step_convolution(self, convolution, current_step, history):
            if not self.causal:
                raise RuntimeError("TemporalBlock.forward_step only supports causal blocks.")
            current_inputs = current_step.unsqueeze(-1)
            if self.padding > 0:
                full_inputs = torch.cat([history, current_inputs], dim=-1)
                next_history = full_inputs[:, :, -self.padding :].detach()
            else:
                full_inputs = current_inputs
                next_history = history[:, :, :0]
            outputs = convolution(full_inputs).squeeze(-1)
            return outputs, next_history

        def forward_step(self, inputs, stream_state: Dict[str, torch.Tensor]):
            if inputs.ndim != 2:
                raise ValueError(f"Expected `[B, C]` block input, got {tuple(inputs.shape)}")
            conv1_history = stream_state["conv1_history"]
            conv2_history = stream_state["conv2_history"]
            outputs, next_conv1_history = self._step_convolution(self.conv1, inputs, conv1_history)
            outputs = self.activation(outputs)
            outputs = self.dropout(outputs)
            outputs, next_conv2_history = self._step_convolution(self.conv2, outputs, conv2_history)
            outputs = self.activation(outputs)
            outputs = self.dropout(outputs)
            return outputs + inputs, {
                "conv1_history": next_conv1_history,
                "conv2_history": next_conv2_history,
            }


    class AttachmentStateEncoder(nn.Module):
        """Infer a latent attachment state from temporal backbone features."""

        def __init__(self, hidden_dim: int, latent_dim: int):
            super().__init__()
            self.latent_projection = nn.Conv1d(hidden_dim, latent_dim, kernel_size=1)

        def forward(self, features):
            latent_sequence = self.latent_projection(features).transpose(1, 2)
            latent_summary = latent_sequence.mean(dim=1)
            return latent_sequence, latent_summary

        def forward_step(self, feature_step):
            if feature_step.ndim != 2:
                raise ValueError(f"Expected `[B, C]` feature_step, got {tuple(feature_step.shape)}")
            return self.latent_projection(feature_step.unsqueeze(-1)).squeeze(-1)


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
            causal: bool = False,
        ):
            super().__init__()
            self.input_dim = int(input_dim)
            self.output_dim = int(output_dim)
            self.hidden_dim = int(hidden_dim)
            self.num_layers = int(num_layers)
            self.kernel_size = int(kernel_size)
            self.attach_latent_dim = int(attach_latent_dim)
            self.use_attachment_latent = self.attach_latent_dim > 0
            self.causal = bool(causal)
            self.input_projection = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
            self.blocks = nn.ModuleList(
                [
                    TemporalBlock(
                        hidden_dim,
                        self.kernel_size,
                        dilation=2**layer_idx,
                        dropout=dropout,
                        causal=self.causal,
                    )
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

        @property
        def receptive_field(self) -> int:
            dilations = [2**layer_idx for layer_idx in range(self.num_layers)]
            history = 2 * (self.kernel_size - 1) * sum(dilations)
            return int(1 + history)

        def init_stream_state(self, batch_size: int = 1, device=None, dtype=None) -> Dict[str, torch.Tensor]:
            if batch_size <= 0:
                raise ValueError("batch_size must be positive.")
            if device is None or dtype is None:
                parameter = next(self.parameters())
                if device is None:
                    device = parameter.device
                if dtype is None:
                    dtype = parameter.dtype
            state: Dict[str, Any] = {
                "block_states": [
                    block.init_stream_state(
                        batch_size=int(batch_size),
                        channels=self.hidden_dim,
                        device=device,
                        dtype=dtype,
                    )
                    for block in self.blocks
                ]
            }
            if self.use_attachment_latent:
                state["attach_sum"] = torch.zeros(
                    int(batch_size),
                    self.attach_latent_dim,
                    device=device,
                    dtype=dtype,
                )
                state["attach_count"] = torch.zeros(
                    int(batch_size),
                    1,
                    device=device,
                    dtype=dtype,
                )
            return state

        def forward_stream(self, inputs, stream_state: Dict[str, torch.Tensor] | None = None):
            if inputs.ndim != 3:
                raise ValueError(f"Expected `[B, T, C]` input, got {tuple(inputs.shape)}")
            if not self.causal:
                raise RuntimeError("forward_stream is only available when the TCN is configured as causal.")
            step_predictions = []
            step_residuals = []
            attach_sequences = []
            current_state = stream_state
            for step_index in range(inputs.shape[1]):
                step_outputs = self.forward_step(inputs[:, step_index, :], stream_state=current_state)
                current_state = step_outputs["stream_state"]
                step_predictions.append(step_outputs["predictions"])
                step_residuals.append(step_outputs["residual"])
                if "z_attach_sequence" in step_outputs:
                    attach_sequences.append(step_outputs["z_attach_sequence"])
            outputs = {
                "predictions": torch.cat(step_predictions, dim=1),
                "residual": torch.cat(step_residuals, dim=1),
                "stream_state": current_state,
            }
            if attach_sequences:
                attach_sequence = torch.cat(attach_sequences, dim=1)
                outputs["z_attach_sequence"] = attach_sequence
                outputs["z_attach"] = attach_sequence.mean(dim=1)
            return outputs

        def forward_step(self, input_step, stream_state: Dict[str, torch.Tensor] | None = None):
            if input_step.ndim == 2:
                step_inputs = input_step
            elif input_step.ndim == 3 and input_step.shape[1] == 1:
                step_inputs = input_step[:, 0, :]
            else:
                raise ValueError(
                    f"Expected `[B, C]` or `[B, 1, C]` input_step, got {tuple(input_step.shape)}"
                )
            if not self.causal:
                raise RuntimeError("forward_step is only available when the TCN is configured as causal.")

            batch_size = int(step_inputs.shape[0])
            if stream_state is None:
                stream_state = self.init_stream_state(
                    batch_size=batch_size,
                    device=step_inputs.device,
                    dtype=step_inputs.dtype,
                )
            block_states = list(stream_state.get("block_states", []))
            if len(block_states) != len(self.blocks):
                raise ValueError(
                    f"Expected {len(self.blocks)} block states, got {len(block_states)}"
                )

            base = step_inputs
            outputs = self.input_projection(base.unsqueeze(-1)).squeeze(-1)
            next_block_states = []
            for block, block_state in zip(self.blocks, block_states):
                outputs, next_block_state = block.forward_step(outputs, block_state)
                next_block_states.append(next_block_state)

            aux_outputs = {}
            conditioned = outputs
            next_stream_state: Dict[str, Any] = {"block_states": next_block_states}
            if self.use_attachment_latent:
                latent_step = self.attachment_encoder.forward_step(outputs)
                gate = torch.sigmoid(self.feature_gate(latent_step.unsqueeze(-1)).squeeze(-1))
                shift = self.feature_shift(latent_step.unsqueeze(-1)).squeeze(-1)
                conditioned = outputs * (1.0 + gate) + shift

                attach_sum = stream_state["attach_sum"] + latent_step.detach()
                attach_count = stream_state["attach_count"] + 1.0
                next_stream_state["attach_sum"] = attach_sum
                next_stream_state["attach_count"] = attach_count
                aux_outputs["z_attach"] = attach_sum / attach_count
                aux_outputs["z_attach_sequence"] = latent_step.unsqueeze(1)

            residual = self.residual_projection(conditioned.unsqueeze(-1)).squeeze(-1)
            if self.base_projection is not None:
                base_signal = self.base_projection(base.unsqueeze(-1)).squeeze(-1)
            else:
                base_signal = base
            prediction_step = base_signal + residual
            outputs = {
                "predictions": prediction_step.unsqueeze(1),
                "prediction_step": prediction_step,
                "residual": residual.unsqueeze(1),
                "stream_state": next_stream_state,
            }
            outputs.update(aux_outputs)
            return outputs

else:

    class TCNBaseline:  # pragma: no cover - runtime safeguard only
        def __init__(self, *args, **kwargs):
            require_torch()
