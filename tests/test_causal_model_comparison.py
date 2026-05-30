import tempfile
import unittest
from pathlib import Path

import pandas as pd

from project.main_causal_model_comparison import (
    CAUSAL_TABLE_COLUMNS,
    LOSS_PROFILES,
    _apply_loss_profile,
    _load_setting_config,
    _write_comparison_table,
)
from project.models import build_model
from project.utils.torch_compat import TORCH_AVAILABLE, torch


class TestCausalModelComparison(unittest.TestCase):
    def test_final_composite_loss_profile_applies_expected_weights(self):
        config = {"loss_weights": {"time_l1": 1.0, "mse": 1.0}}
        updated = _apply_loss_profile(config, "final_composite")
        self.assertEqual(LOSS_PROFILES["final_composite"]["weights"], {"time_l1": 1.0, "mse": 1.0, "spectral": 0.2})
        self.assertIn("MSE=1.0", LOSS_PROFILES["final_composite"]["description"])
        self.assertEqual(updated["loss_weights"], LOSS_PROFILES["final_composite"]["weights"])
        self.assertEqual(config["loss_weights"]["mse"], 1.0)

    def test_nested_causal_config_resolves_repository_root(self):
        config = _load_setting_config(
            model_name="tcn_causal",
            config_dir=Path("project/configs/causal_models"),
            output_root=Path("outputs/causal_model_comparison"),
            setting="by_experiment",
            loss_profile="final_composite",
        )
        self.assertEqual(Path(config["repo_root"]).resolve(), Path.cwd().resolve())
        self.assertIn("causal_model_comparison", config["outputs_root"])

    def test_write_comparison_table_exports_fixed_columns(self):
        row = {
            "Setting": "by_experiment",
            "Model": "TCN-causal",
            "Causal type": "dilated convolution",
            "RMSE": "1.0000 +/- 0.1000",
            "Pearson": "0.9000 +/- 0.0100",
            "PSD Dist.": "0.1000 +/- 0.0100",
            "HF Improve.": "25.0000 +/- 1.0000",
            "Parameters": "12345",
            "FP32 size (MB)": "0.0471",
            "CPU forward ms/window": "1.2000 +/- 0.1000",
            "Streaming ms/step": "0.0500 +/- 0.0050",
            "Deployment verdict": "yes_embedded_friendly",
            "Loss profile": "final_composite",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _write_comparison_table([row], Path(tmpdir))
            for path in paths.values():
                self.assertTrue(path.exists())
            table = pd.read_csv(paths["csv"])
            self.assertEqual(list(table.columns), CAUSAL_TABLE_COLUMNS)
            self.assertEqual(table.loc[0, "Model"], "TCN-causal")

    def test_all_causal_aliases_return_sequence_prediction_bundles(self):
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch is not available in this environment.")
        batch = torch.randn(2, 12, 6)
        model_configs = {
            "tcn_causal": {
                "hidden_dim": 8,
                "num_layers": 2,
                "kernel_size": 3,
                "dropout": 0.0,
                "attach_latent_dim": 4,
                "causal": True,
            },
            "transformer_causal": {
                "transformer_model_dim": 16,
                "transformer_num_layers": 1,
                "transformer_num_heads": 4,
                "transformer_ff_dim": 32,
                "dropout": 0.0,
                "causal": True,
            },
            "gru_causal": {"gru_hidden_dim": 8, "gru_num_layers": 1, "dropout": 0.0},
            "lstm_causal": {"lstm_hidden_dim": 8, "lstm_num_layers": 1, "dropout": 0.0},
            "mlp_causal": {"mlp_hidden_dim": 8, "dropout": 0.0},
        }
        for model_name, config in model_configs.items():
            model = build_model(model_name, input_dim=6, output_dim=6, model_config=config).eval()
            outputs = model(batch)
            predictions = outputs["predictions"] if isinstance(outputs, dict) else outputs
            self.assertEqual(tuple(predictions.shape), (2, 12, 6), model_name)
            self.assertTrue(hasattr(model, "forward_step"), model_name)

    def test_transformer_causal_output_does_not_change_when_future_inputs_change(self):
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch is not available in this environment.")
        torch.manual_seed(7)
        model = build_model(
            "transformer_causal",
            input_dim=6,
            output_dim=6,
            model_config={
                "transformer_model_dim": 16,
                "transformer_num_layers": 1,
                "transformer_num_heads": 4,
                "transformer_ff_dim": 32,
                "dropout": 0.0,
                "causal": True,
            },
        ).eval()
        base = torch.randn(1, 10, 6)
        changed_future = base.clone()
        changed_future[:, 6:, :] = torch.randn(1, 4, 6) * 100.0
        with torch.no_grad():
            base_outputs = model(base)["predictions"]
            changed_outputs = model(changed_future)["predictions"]
        self.assertTrue(torch.allclose(base_outputs[:, :6, :], changed_outputs[:, :6, :], atol=1e-5))

    def test_forward_step_matches_offline_causal_forward(self):
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch is not available in this environment.")
        torch.manual_seed(11)
        batch = torch.randn(2, 10, 6)
        model_configs = {
            "tcn_causal": {
                "hidden_dim": 8,
                "num_layers": 2,
                "kernel_size": 3,
                "dropout": 0.0,
                "attach_latent_dim": 4,
                "causal": True,
            },
            "transformer_causal": {
                "transformer_model_dim": 16,
                "transformer_num_layers": 1,
                "transformer_num_heads": 4,
                "transformer_ff_dim": 32,
                "dropout": 0.0,
                "causal": True,
            },
            "gru_causal": {"gru_hidden_dim": 8, "gru_num_layers": 1, "dropout": 0.0},
            "lstm_causal": {"lstm_hidden_dim": 8, "lstm_num_layers": 1, "dropout": 0.0},
            "mlp_causal": {"mlp_hidden_dim": 8, "dropout": 0.0},
        }
        for model_name, config in model_configs.items():
            model = build_model(model_name, input_dim=6, output_dim=6, model_config=config).eval()
            with torch.no_grad():
                offline_outputs = model(batch)
                offline_predictions = (
                    offline_outputs["predictions"] if isinstance(offline_outputs, dict) else offline_outputs
                )
                stream_state = model.init_stream_state(batch_size=batch.shape[0], device=batch.device, dtype=batch.dtype)
                streamed_steps = []
                for step_index in range(batch.shape[1]):
                    step_outputs = model.forward_step(batch[:, step_index, :], stream_state=stream_state)
                    stream_state = step_outputs.get("stream_state", stream_state)
                    step_prediction = step_outputs.get("prediction_step")
                    if step_prediction is None:
                        step_prediction = step_outputs["predictions"].squeeze(1)
                    streamed_steps.append(step_prediction.unsqueeze(1))
                streamed_predictions = torch.cat(streamed_steps, dim=1)
            self.assertTrue(
                torch.allclose(offline_predictions, streamed_predictions, atol=1e-5),
                model_name,
            )


if __name__ == "__main__":
    unittest.main()
