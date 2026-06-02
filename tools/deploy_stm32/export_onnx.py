from __future__ import annotations

import argparse
import sys
from pathlib import Path

from deploy_common import (
    DEFAULT_CHECKPOINT,
    DEFAULT_OUTPUT_ROOT,
    load_checkpoint_and_model,
    repo_root_from_file,
)


def build_export_wrapper(torch, model, output_mode: str):
    class ExportWrapper(torch.nn.Module):
        def __init__(self, wrapped_model, mode: str):
            super().__init__()
            self.wrapped_model = wrapped_model
            self.mode = mode

        def forward(self, inputs):
            outputs = self.wrapped_model(inputs)
            predictions = outputs["predictions"] if isinstance(outputs, dict) else outputs
            if self.mode == "last_step":
                return predictions[:, -1, :]
            if self.mode == "sequence":
                return predictions
            raise RuntimeError(f"Unsupported output mode: {self.mode}")

    return ExportWrapper(model, output_mode).eval()


def build_cubeai_friendly_export_wrapper(torch, model, output_mode: str, window_size: int):
    class CubeAiFriendlyExportWrapper(torch.nn.Module):
        def __init__(self, wrapped_model, mode: str, fixed_window_size: int):
            super().__init__()
            self.wrapped_model = wrapped_model
            self.mode = mode
            self.fixed_window_size = int(fixed_window_size)
            for block_index, block in enumerate(self.wrapped_model.blocks):
                padding = int(getattr(block, "padding", 0))
                channels = int(self.wrapped_model.hidden_dim)
                self.register_buffer(
                    f"block_{block_index}_zero_padding",
                    torch.zeros(1, channels, padding, dtype=torch.float32),
                )

        def _zero_padding_for(self, block_index: int):
            return getattr(self, f"block_{block_index}_zero_padding")

        def _causal_conv(self, block_index: int, convolution, values):
            zero_padding = self._zero_padding_for(block_index)
            if zero_padding.shape[-1] > 0:
                values = torch.cat([zero_padding, values], dim=-1)
            return convolution(values)

        def _block_forward(self, block_index: int, block, inputs):
            residual = inputs
            outputs = self._causal_conv(block_index, block.conv1, inputs)
            outputs = block.activation(outputs)
            outputs = self._causal_conv(block_index, block.conv2, outputs)
            outputs = block.activation(outputs)
            return outputs + residual

        def forward(self, inputs):
            base = inputs.transpose(1, 2)
            outputs = self.wrapped_model.input_projection(base)
            for block_index, block in enumerate(self.wrapped_model.blocks):
                outputs = self._block_forward(block_index, block, outputs)

            conditioned = outputs
            if self.wrapped_model.use_attachment_latent:
                latent_features = self.wrapped_model.attachment_encoder.latent_projection(outputs)
                gate = torch.sigmoid(self.wrapped_model.feature_gate(latent_features))
                shift = self.wrapped_model.feature_shift(latent_features)
                conditioned = outputs * (1.0 + gate) + shift

            residual = self.wrapped_model.residual_projection(conditioned)
            if self.wrapped_model.base_projection is not None:
                base_signal = self.wrapped_model.base_projection(base)
            else:
                base_signal = base
            predictions_ch_first = base_signal + residual
            if self.mode == "last_step":
                return predictions_ch_first[:, :, self.fixed_window_size - 1]
            if self.mode == "sequence":
                return predictions_ch_first.transpose(1, 2)
            raise RuntimeError(f"Unsupported output mode: {self.mode}")

    return CubeAiFriendlyExportWrapper(model, output_mode, window_size).eval()


def force_static_io_shapes(onnx_path: Path, window_size: int, channels: int, output_mode: str) -> None:
    import onnx

    model = onnx.load(str(onnx_path))
    input_dims = model.graph.input[0].type.tensor_type.shape.dim
    for dim, value in zip(input_dims, [1, int(window_size), int(channels)]):
        dim.dim_param = ""
        dim.dim_value = int(value)

    output_shape = [1, int(channels)] if output_mode == "last_step" else [1, int(window_size), int(channels)]
    output_dims = model.graph.output[0].type.tensor_type.shape.dim
    while len(output_dims) < len(output_shape):
        output_dims.add()
    for dim, value in zip(output_dims, output_shape):
        dim.dim_param = ""
        dim.dim_value = int(value)
    onnx.save(model, str(onnx_path))


def export_onnx(
    checkpoint: Path,
    output: Path,
    window_size: int | None = None,
    output_mode: str = "last_step",
    opset: int = 17,
    device: str = "cpu",
    cubeai_friendly: bool = False,
) -> Path:
    torch, _, config, model = load_checkpoint_and_model(checkpoint, device_name=device)
    input_dim = len(config["input_channels"])
    resolved_window = int(window_size or config.get("window_size", 64))
    if cubeai_friendly:
        wrapper = build_cubeai_friendly_export_wrapper(torch, model, output_mode, resolved_window)
    else:
        wrapper = build_export_wrapper(torch, model, output_mode)
    wrapper.to(device)
    dummy_input = torch.zeros(1, resolved_window, input_dim, dtype=torch.float32, device=device)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        dummy_input,
        str(output),
        input_names=["imu_window"],
        output_names=["compensated_imu" if output_mode == "last_step" else "compensated_sequence"],
        opset_version=int(opset),
        do_constant_folding=True,
        dynamic_axes=None,
    )
    force_static_io_shapes(output, resolved_window, input_dim, output_mode)
    return output


def main() -> None:
    repo_root = repo_root_from_file()
    parser = argparse.ArgumentParser(description="Export the causal TCN checkpoint to fixed-shape ONNX.")
    parser.add_argument("--checkpoint", type=Path, default=repo_root / DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / DEFAULT_OUTPUT_ROOT / "onnx" / "tcn_causal_last_step.onnx",
    )
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--output-mode", choices=["last_step", "sequence"], default="last_step")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--cubeai-friendly",
        action="store_true",
        help="Export a fixed-shape graph that avoids dynamic ONNX Pad for STM32Cube.AI.",
    )
    args = parser.parse_args()
    try:
        path = export_onnx(
            checkpoint=args.checkpoint,
            output=args.output,
            window_size=args.window_size,
            output_mode=args.output_mode,
            opset=args.opset,
            device=args.device,
            cubeai_friendly=args.cubeai_friendly,
        )
    except Exception as exc:
        print(f"[ERROR] ONNX export failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"Saved ONNX model to {path.resolve()}")


if __name__ == "__main__":
    main()
