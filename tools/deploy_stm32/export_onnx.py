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


def export_onnx(
    checkpoint: Path,
    output: Path,
    window_size: int | None = None,
    output_mode: str = "last_step",
    opset: int = 17,
    device: str = "cpu",
) -> Path:
    torch, _, config, model = load_checkpoint_and_model(checkpoint, device_name=device)
    input_dim = len(config["input_channels"])
    resolved_window = int(window_size or config.get("window_size", 64))
    wrapper = build_export_wrapper(torch, model, output_mode)
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
    args = parser.parse_args()
    try:
        path = export_onnx(
            checkpoint=args.checkpoint,
            output=args.output,
            window_size=args.window_size,
            output_mode=args.output_mode,
            opset=args.opset,
            device=args.device,
        )
    except Exception as exc:
        print(f"[ERROR] ONNX export failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"Saved ONNX model to {path.resolve()}")


if __name__ == "__main__":
    main()
