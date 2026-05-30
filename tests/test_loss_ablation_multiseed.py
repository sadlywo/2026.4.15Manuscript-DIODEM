from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from project.experiments.ablation import aggregate_ablation_seed_rows, run_ablation_suite
from project.main_make_loss_ablation_table import _build_output_rows


class LossAblationMultiSeedTests(unittest.TestCase):
    def test_run_ablation_suite_creates_variant_directory_before_saving_config(self):
        with TemporaryDirectory() as tmp_dir:
            repo_root = Path(tmp_dir)
            base_config = {
                "repo_root": str(repo_root),
                "processed_root": "processed",
                "outputs_root": "outputs/supervised_tcn_causal",
                "seed": 42,
                "model_name": "tcn_causal",
                "model": {"attach_latent_dim": 8},
                "loss_weights": {"time_l1": 1.0, "mse": 1.0, "spectral": 0.2},
                "evaluation": {"checkpoint_name": "best.pt", "baseline_models": [], "trained_model_checkpoints": []},
            }
            metrics = {
                "rmse_mean": 0.1,
                "pearson_mean": 0.9,
                "psd_distance_mean": 0.006,
                "hf_ratio_improvement_mean": 5.0,
                "acc_norm_rmse": 0.11,
                "gyr_norm_rmse": 0.12,
                "num_windows": 100,
            }

            with patch("project.experiments.ablation.require_torch"), patch(
                "project.experiments.ablation.build_processed_splits"
            ), patch("project.experiments.ablation.load_json", return_value=metrics):
                summary_path = run_ablation_suite(
                    base_config,
                    variant_names=["full_model"],
                    train=False,
                    evaluate=False,
                )

            config_path = repo_root / "outputs" / "supervised_tcn_causal_ablations" / "full_model" / "resolved_config.json"
            self.assertTrue(config_path.exists())
            self.assertEqual(summary_path, repo_root / "outputs" / "supervised_tcn_causal_ablations" / "ablation_summary.csv")

    def test_run_ablation_suite_rejects_removed_variants(self):
        base_config = {
            "repo_root": ".",
            "processed_root": "processed",
            "outputs_root": "outputs/supervised_tcn_causal",
            "seed": 42,
        }

        with patch("project.experiments.ablation.require_torch"), patch("project.experiments.ablation.build_processed_splits"):
            with self.assertRaisesRegex(ValueError, "no_derivative_loss"):
                run_ablation_suite(
                    base_config,
                    variant_names=["no_derivative_loss"],
                    train=False,
                    evaluate=False,
                )

    def test_aggregate_ablation_seed_rows_reports_mean_and_std(self):
        rows = [
            {
                "variant_name": "full_model",
                "description": "Full",
                "outputs_root": "outputs/ablations/full_model/seed_runs/seed_0042",
                "seed": 42,
                "attach_latent_dim": 8,
                "time_l1": 1.0,
                "mse": 1.0,
                "spectral": 0.2,
                "rmse_mean": 0.10,
                "pearson_mean": 0.90,
                "psd_distance_mean": 0.006,
                "hf_ratio_improvement_mean": 5.0,
                "acc_norm_rmse": 0.11,
                "gyr_norm_rmse": 0.12,
                "num_windows": 100,
            },
            {
                "variant_name": "full_model",
                "description": "Full",
                "outputs_root": "outputs/ablations/full_model/seed_runs/seed_0043",
                "seed": 43,
                "attach_latent_dim": 8,
                "time_l1": 1.0,
                "mse": 1.0,
                "spectral": 0.2,
                "rmse_mean": 0.14,
                "pearson_mean": 0.94,
                "psd_distance_mean": 0.010,
                "hf_ratio_improvement_mean": 6.0,
                "acc_norm_rmse": 0.15,
                "gyr_norm_rmse": 0.16,
                "num_windows": 100,
            },
        ]

        frame = aggregate_ablation_seed_rows(rows)

        self.assertEqual(len(frame), 1)
        row = frame.iloc[0]
        self.assertEqual(row["num_seeds"], 2)
        self.assertAlmostEqual(row["rmse_mean"], 0.12)
        self.assertAlmostEqual(row["rmse_std"], 0.02)
        self.assertAlmostEqual(row["psd_distance_mean"], 0.008)
        self.assertAlmostEqual(row["psd_distance_std"], 0.002)
        self.assertEqual(row["seed_list"], "42,43")

    def test_supplementary_table_formats_complete_loss_variant_multiseed_schema(self):
        summary_rows = [
            _summary_row("full_model", attach_latent_dim="8", time_l1="1.0", mse="1.0", spectral="0.2", rmse="0.1000"),
            _summary_row("no_l1_loss", attach_latent_dim="8", time_l1="0.0", mse="1.0", spectral="0.2", rmse="0.1300"),
            _summary_row("no_mse_loss", attach_latent_dim="8", time_l1="1.0", mse="0.0", spectral="0.2", rmse="0.1250"),
            _summary_row("no_spectral_loss", attach_latent_dim="8", time_l1="1.0", mse="1.0", spectral="0.0", rmse="0.1200"),
            _summary_row("mse_only", attach_latent_dim="8", time_l1="0.0", mse="1.0", spectral="0.0", rmse="0.1500"),
            _summary_row("no_attachment_latent", attach_latent_dim="0", time_l1="1.0", mse="1.0", spectral="0.2", rmse="0.1400"),
        ]

        output_rows = _build_output_rows(summary_rows)
        headers = list(output_rows[0].keys())

        self.assertEqual(
            [row["Variant"] for row in output_rows],
            ["Full model", "w/o L1 loss", "w/o MSE loss", "w/o spectral loss", "MSE only", "w/o attachment latent"],
        )
        self.assertEqual(output_rows[0]["RMSE"], "0.1000 +/- 0.0100")
        self.assertEqual(output_rows[4]["Delta RMSE vs Full"], "+50.00%")
        self.assertEqual(output_rows[1]["L1"], "N")
        self.assertEqual(output_rows[2]["MSE"], "N")
        self.assertEqual(output_rows[0]["Seeds"], "2")
        self.assertNotIn("Deriv.", headers)
        self.assertNotIn("Att-L2", headers)
        self.assertNotIn("Att-Temp", headers)


def _summary_row(
    variant_name: str,
    attach_latent_dim: str,
    time_l1: str,
    mse: str,
    spectral: str,
    rmse: str,
) -> dict[str, str]:
    return {
        "variant_name": variant_name,
        "attach_latent_dim": attach_latent_dim,
        "time_l1": time_l1,
        "mse": mse,
        "spectral": spectral,
        "rmse_mean": rmse,
        "rmse_std": "0.0100",
        "pearson_mean": "0.9000",
        "pearson_std": "0.0200",
        "psd_distance_mean": "0.00600",
        "psd_distance_std": "0.00020",
        "hf_ratio_improvement_mean": "5.000",
        "hf_ratio_improvement_std": "0.100",
        "acc_norm_rmse": "0.1100",
        "acc_norm_rmse_std": "0.0100",
        "gyr_norm_rmse": "0.1200",
        "gyr_norm_rmse_std": "0.0200",
        "num_windows": "100",
        "num_seeds": "2",
    }


if __name__ == "__main__":
    unittest.main()
