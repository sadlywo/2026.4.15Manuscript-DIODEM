from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.signal import butter, savgol_filter, sosfiltfilt, wiener

from project.models.gru_model import GRUBaseline
from project.models.mlp_model import MLPBaseline
from project.models.tcn_model import TCNBaseline
from project.models.transformer_model import TransformerBaseline
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


    class _ScipyFilterBaseline(nn.Module):
        """Base class for CPU-side classic signal filters used only at inference time."""

        def _filter_numpy(self, values: np.ndarray) -> np.ndarray:
            raise NotImplementedError

        def forward(self, inputs):
            if inputs.ndim != 3:
                raise ValueError(f"Expected `[B, T, C]` input, got {tuple(inputs.shape)}")
            filtered = self._filter_numpy(inputs.detach().cpu().numpy())
            return inputs.new_tensor(filtered)


    class ButterworthLowPassBaseline(_ScipyFilterBaseline):
        """Zero-phase Butterworth low-pass filter baseline."""

        def __init__(self, sampling_frequency: float, cutoff_hz: float = 5.0, order: int = 4):
            super().__init__()
            nyquist = max(float(sampling_frequency) / 2.0, 1e-6)
            normalized_cutoff = min(max(float(cutoff_hz) / nyquist, 1e-4), 0.999)
            self.sos = butter(int(order), normalized_cutoff, btype="low", output="sos")

        def _filter_numpy(self, values: np.ndarray) -> np.ndarray:
            filtered = np.empty_like(values, dtype=np.float32)
            for batch_index in range(values.shape[0]):
                for channel_index in range(values.shape[2]):
                    signal = values[batch_index, :, channel_index]
                    filtered[batch_index, :, channel_index] = sosfiltfilt(self.sos, signal).astype(np.float32)
            return filtered


    class SavitzkyGolayBaseline(_ScipyFilterBaseline):
        """Savitzky-Golay smoothing baseline."""

        def __init__(self, window_length: int = 9, polyorder: int = 3):
            super().__init__()
            self.window_length = int(window_length)
            self.polyorder = int(polyorder)

        def _resolve_window(self, signal_length: int) -> int:
            min_valid_window = self.polyorder + 1
            if min_valid_window % 2 == 0:
                min_valid_window += 1
            max_valid_window = signal_length if signal_length % 2 == 1 else signal_length - 1
            window = min(self.window_length, max_valid_window)
            window = max(window, min_valid_window)
            if window > signal_length:
                window = signal_length if signal_length % 2 == 1 else signal_length - 1
            return int(window)

        def _filter_numpy(self, values: np.ndarray) -> np.ndarray:
            filtered = np.empty_like(values, dtype=np.float32)
            signal_length = values.shape[1]
            window = self._resolve_window(signal_length)
            if window <= self.polyorder:
                return values.astype(np.float32)
            for batch_index in range(values.shape[0]):
                for channel_index in range(values.shape[2]):
                    signal = values[batch_index, :, channel_index]
                    filtered[batch_index, :, channel_index] = savgol_filter(
                        signal,
                        window_length=window,
                        polyorder=min(self.polyorder, window - 1),
                        mode="interp",
                    ).astype(np.float32)
            return filtered


    class WienerBaseline(_ScipyFilterBaseline):
        """Channel-wise Wiener filtering baseline."""

        def __init__(self, window_size: int = 7):
            super().__init__()
            self.window_size = int(window_size)

        def _filter_numpy(self, values: np.ndarray) -> np.ndarray:
            filtered = np.empty_like(values, dtype=np.float32)
            mysize = max(3, self.window_size)
            if mysize % 2 == 0:
                mysize += 1
            for batch_index in range(values.shape[0]):
                for channel_index in range(values.shape[2]):
                    signal = values[batch_index, :, channel_index]
                    filtered[batch_index, :, channel_index] = np.asarray(
                        wiener(signal, mysize=mysize),
                        dtype=np.float32,
                    )
            return filtered

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


    class ButterworthLowPassBaseline:  # pragma: no cover - runtime safeguard only
        def __init__(self, *args, **kwargs):
            require_torch()


    class SavitzkyGolayBaseline:  # pragma: no cover - runtime safeguard only
        def __init__(self, *args, **kwargs):
            require_torch()


    class WienerBaseline:  # pragma: no cover - runtime safeguard only
        def __init__(self, *args, **kwargs):
            require_torch()


AVAILABLE_MODELS = {
    "identity": IdentityBaseline,
    "lowpass": LowPassBaseline,
    "butterworth": ButterworthLowPassBaseline,
    "savgol": SavitzkyGolayBaseline,
    "wiener": WienerBaseline,
    "linear": LinearProjectionBaseline,
    "mlp": MLPBaseline,
    "gru": GRUBaseline,
    "transformer": TransformerBaseline,
    "tcn": TCNBaseline,
    "tcn_causal": TCNBaseline,
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
    if name == "butterworth":
        return ButterworthLowPassBaseline(
            sampling_frequency=float(model_config.get("sampling_frequency", 40.0)),
            cutoff_hz=float(model_config.get("butter_cutoff_hz", 5.0)),
            order=int(model_config.get("butter_order", 4)),
        )
    if name == "savgol":
        return SavitzkyGolayBaseline(
            window_length=int(model_config.get("savgol_window_length", 9)),
            polyorder=int(model_config.get("savgol_polyorder", 3)),
        )
    if name == "wiener":
        return WienerBaseline(window_size=int(model_config.get("wiener_window_size", 7)))
    if name == "linear":
        return LinearProjectionBaseline(input_dim=input_dim, output_dim=output_dim)
    if name == "mlp":
        return MLPBaseline(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=int(model_config.get("mlp_hidden_dim", 128)),
            dropout=float(model_config.get("dropout", 0.1)),
        )
    if name == "gru":
        return GRUBaseline(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=int(model_config.get("gru_hidden_dim", 128)),
            num_layers=int(model_config.get("gru_num_layers", 2)),
            dropout=float(model_config.get("dropout", 0.1)),
        )
    if name == "transformer":
        return TransformerBaseline(
            input_dim=input_dim,
            output_dim=output_dim,
            model_dim=int(model_config.get("transformer_model_dim", 128)),
            num_layers=int(model_config.get("transformer_num_layers", 3)),
            num_heads=int(model_config.get("transformer_num_heads", 4)),
            feedforward_dim=int(model_config.get("transformer_ff_dim", 256)),
            dropout=float(model_config.get("dropout", 0.1)),
        )
    return TCNBaseline(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=int(model_config.get("hidden_dim", 64)),
        num_layers=int(model_config.get("num_layers", 4)),
        kernel_size=int(model_config.get("kernel_size", 3)),
        dropout=float(model_config.get("dropout", 0.1)),
        attach_latent_dim=int(model_config.get("attach_latent_dim", 8)),
        causal=bool(model_config.get("causal", name == "tcn_causal")),
    )
