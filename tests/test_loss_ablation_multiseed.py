from __future__ import annotations

import unittest

from project.experiments.ablation import aggregate_ablation_seed_rows
from project.main_make_loss_ablation_table import _build_output_rows


class LossAblationMultiSeedTests(unittest.TestCase):
    def test_aggregate_ablation_seed_rows_reports_mean_and_std(self):
        rows = [
            {
                "variant_name": "full_model",
                "description": "Full",
                "outputs_root": "outputs/ablations/full_model/seed_runs/seed_0042",
                "seed": 42,
                "attach_latent_dim": 8,
                "time_l1": 1.0,
                "mse": 0.5,
                "derivative": 0.3,
                "spectral": 0.2,
                "smoothness": 0.0,
                "attach_l2": 0.001,
                "attach_temporal": 0.001,
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
                "mse": 0.5,
                "derivative": 0.3,
                "spectral": 0.2,
                "smoothness": 0.0,
                "attach_l2": 0.001,
                "attach_temporal": 0.001,
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

    def test_supplementary_table_formats_multiseed_mean_plus_std(self):
        summary_rows = [
            {
                "variant_name": "full_model",
                "attach_latent_dim": "8",
                "time_l1": "1.0",
                "mse": "0.5",
                "derivative": "0.3",
                "spectral": "0.2",
                "attach_l2": "0.001",
                "attach_temporal": "0.001",
                "rmse_mean": "0.1000",
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
            },
            {
                "variant_name": "no_derivative_loss",
                "attach_latent_dim": "8",
                "time_l1": "1.0",
                "mse": "0.5",
                "derivative": "0.0",
                "spectral": "0.2",
                "attach_l2": "0.001",
                "attach_temporal": "0.001",
                "rmse_mean": "0.1100",
                "rmse_std": "0.0300",
                "pearson_mean": "0.8800",
                "pearson_std": "0.0100",
                "psd_distance_mean": "0.00660",
                "psd_distance_std": "0.00030",
                "hf_ratio_improvement_mean": "4.800",
                "hf_ratio_improvement_std": "0.200",
                "acc_norm_rmse": "0.1200",
                "acc_norm_rmse_std": "0.0200",
                "gyr_norm_rmse": "0.1300",
                "gyr_norm_rmse_std": "0.0200",
                "num_windows": "100",
                "num_seeds": "2",
            },
            {
                "variant_name": "no_spectral_loss",
                "attach_latent_dim": "8",
                "time_l1": "1.0",
                "mse": "0.5",
                "derivative": "0.3",
                "spectral": "0.0",
                "attach_l2": "0.001",
                "attach_temporal": "0.001",
                "rmse_mean": "0.1200",
                "rmse_std": "0.0200",
                "pearson_mean": "0.8700",
                "pearson_std": "0.0100",
                "psd_distance_mean": "0.00700",
                "psd_distance_std": "0.00040",
                "hf_ratio_improvement_mean": "4.700",
                "hf_ratio_improvement_std": "0.200",
                "acc_norm_rmse": "0.1300",
                "acc_norm_rmse_std": "0.0200",
                "gyr_norm_rmse": "0.1400",
                "gyr_norm_rmse_std": "0.0200",
                "num_windows": "100",
                "num_seeds": "2",
            },
            {
                "variant_name": "no_attachment_regularization",
                "attach_latent_dim": "8",
                "time_l1": "1.0",
                "mse": "0.5",
                "derivative": "0.3",
                "spectral": "0.2",
                "attach_l2": "0.0",
                "attach_temporal": "0.0",
                "rmse_mean": "0.1300",
                "rmse_std": "0.0200",
                "pearson_mean": "0.8600",
                "pearson_std": "0.0100",
                "psd_distance_mean": "0.00750",
                "psd_distance_std": "0.00030",
                "hf_ratio_improvement_mean": "4.600",
                "hf_ratio_improvement_std": "0.200",
                "acc_norm_rmse": "0.1400",
                "acc_norm_rmse_std": "0.0200",
                "gyr_norm_rmse": "0.1500",
                "gyr_norm_rmse_std": "0.0200",
                "num_windows": "100",
                "num_seeds": "2",
            },
            {
                "variant_name": "no_attachment_latent",
                "attach_latent_dim": "0",
                "time_l1": "1.0",
                "mse": "0.5",
                "derivative": "0.3",
                "spectral": "0.2",
                "attach_l2": "0.0",
                "attach_temporal": "0.0",
                "rmse_mean": "0.1400",
                "rmse_std": "0.0200",
                "pearson_mean": "0.8500",
                "pearson_std": "0.0100",
                "psd_distance_mean": "0.00800",
                "psd_distance_std": "0.00030",
                "hf_ratio_improvement_mean": "4.500",
                "hf_ratio_improvement_std": "0.200",
                "acc_norm_rmse": "0.1500",
                "acc_norm_rmse_std": "0.0200",
                "gyr_norm_rmse": "0.1600",
                "gyr_norm_rmse_std": "0.0200",
                "num_windows": "100",
                "num_seeds": "2",
            },
            {
                "variant_name": "mse_only",
                "attach_latent_dim": "0",
                "time_l1": "0.0",
                "mse": "1.0",
                "derivative": "0.0",
                "spectral": "0.0",
                "attach_l2": "0.0",
                "attach_temporal": "0.0",
                "rmse_mean": "0.1500",
                "rmse_std": "0.0200",
                "pearson_mean": "0.8400",
                "pearson_std": "0.0100",
                "psd_distance_mean": "0.00900",
                "psd_distance_std": "0.00030",
                "hf_ratio_improvement_mean": "4.400",
                "hf_ratio_improvement_std": "0.200",
                "acc_norm_rmse": "0.1600",
                "acc_norm_rmse_std": "0.0200",
                "gyr_norm_rmse": "0.1700",
                "gyr_norm_rmse_std": "0.0200",
                "num_windows": "100",
                "num_seeds": "2",
            },
        ]

        output_rows = _build_output_rows(summary_rows)
        full_row = output_rows[0]
        no_deriv_row = output_rows[1]

        self.assertEqual(full_row["RMSE"], "0.1000 +/- 0.0100")
        self.assertEqual(full_row["PSD Dist."], "0.00600 +/- 0.00020")
        self.assertEqual(full_row["Seeds"], "2")
        self.assertEqual(no_deriv_row["RMSE"], "0.1100 +/- 0.0300")
        self.assertEqual(no_deriv_row["Delta RMSE vs Full"], "+10.00%")


if __name__ == "__main__":
    unittest.main()
