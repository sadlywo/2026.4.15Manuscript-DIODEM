from __future__ import annotations

import unittest

from project.main_loss_weight_sensitivity import (
    DEFAULT_WEIGHT_SENSITIVITY_SEEDS,
    FINAL_LOSS_WEIGHTS,
    LOCAL_WEIGHT_SWEEP_VALUES,
    METRIC_SPECS,
    _build_manuscript_rows,
    _build_numeric_summary_rows,
    _build_plot_source_rows,
    _build_variant_config,
    _build_variants,
    _extract_metric_stats,
    _final_loss_weights,
    _resolve_seeds,
)


class LossWeightSensitivityTests(unittest.TestCase):
    def test_local_preset_has_thirteen_expected_unique_variants(self):
        variants = _build_variants("local")

        self.assertEqual(
            [variant["name"] for variant in variants],
            [
                "baseline",
                "l1_0p25",
                "l1_0p5",
                "l1_1p5",
                "l1_2p0",
                "mse_0p5",
                "mse_0p75",
                "mse_1p25",
                "mse_1p5",
                "spec_0p05",
                "spec_0p1",
                "spec_0p3",
                "spec_0p4",
            ],
        )
        self.assertEqual(len(variants), 13)

    def test_each_loss_weight_axis_has_five_sweep_values_including_baseline(self):
        base_config = {
            "outputs_root": "outputs/supervised_tcn_causal",
            "model_name": "tcn_causal",
            "loss_weights": dict(FINAL_LOSS_WEIGHTS),
            "evaluation": {"checkpoint_name": "best.pt", "baseline_models": [], "trained_model_checkpoints": []},
        }
        variants = _build_variants("local")

        for term, expected_values in LOCAL_WEIGHT_SWEEP_VALUES.items():
            observed = [FINAL_LOSS_WEIGHTS[term]]
            for variant in variants:
                if variant["changed_term"] != term:
                    continue
                config = _build_variant_config(base_config, variant, outputs_root=f"outputs/{variant['name']}", seed=42)
                observed.append(_final_loss_weights(config)[term])
            self.assertEqual(sorted(observed), expected_values)

    def test_default_seed_resolution_uses_five_seed_protocol(self):
        base_config = {"seed": 99, "experiment_seeds": []}

        self.assertEqual(_resolve_seeds(base_config, cli_seeds=None, cli_seed=None), DEFAULT_WEIGHT_SENSITIVITY_SEEDS)
        self.assertEqual(_resolve_seeds(base_config, cli_seeds=[42, 43, 42], cli_seed=None), [42, 43])
        self.assertEqual(_resolve_seeds(base_config, cli_seeds=None, cli_seed=7), [7])

    def test_baseline_weights_match_final_loss_and_variants_change_one_term(self):
        base_config = {
            "outputs_root": "outputs/supervised_tcn_causal",
            "model_name": "tcn_causal",
            "loss_weights": dict(FINAL_LOSS_WEIGHTS),
            "evaluation": {"checkpoint_name": "best.pt", "baseline_models": ["identity"], "trained_model_checkpoints": []},
        }

        variants = _build_variants("local")
        baseline_config = _build_variant_config(base_config, variants[0], outputs_root="outputs/baseline", seed=42)
        self.assertEqual(_final_loss_weights(baseline_config), FINAL_LOSS_WEIGHTS)

        for variant in variants[1:]:
            config = _build_variant_config(base_config, variant, outputs_root=f"outputs/{variant['name']}", seed=42)
            weights = _final_loss_weights(config)
            changed_terms = [term for term, baseline_value in FINAL_LOSS_WEIGHTS.items() if weights[term] != baseline_value]
            self.assertEqual(changed_terms, [variant["changed_term"]])
            self.assertEqual(config["evaluation"]["baseline_models"], [])
            self.assertEqual(config["evaluation"]["trained_model_checkpoints"], [])

    def test_metric_extraction_reads_multiseed_mean_and_std_columns(self):
        metrics_row = {
            "rmse_mean_mean": "0.1234",
            "rmse_mean_std": "0.0100",
            "pearson_mean_mean": "0.9000",
            "pearson_mean_std": "0.0200",
            "psd_distance_mean_mean": "0.00600",
            "psd_distance_mean_std": "0.00020",
            "hf_ratio_improvement_mean_mean": "5.000",
            "hf_ratio_improvement_mean_std": "0.100",
            "acc_norm_rmse_mean": "0.1100",
            "acc_norm_rmse_std": "0.0100",
            "gyr_norm_rmse_mean": "0.1200",
            "gyr_norm_rmse_std": "0.0200",
        }

        stats = _extract_metric_stats(metrics_row)

        self.assertAlmostEqual(stats["rmse_mean"], 0.1234)
        self.assertAlmostEqual(stats["rmse_std"], 0.0100)
        self.assertAlmostEqual(stats["psd_distance_mean"], 0.00600)
        self.assertAlmostEqual(stats["hf_ratio_improvement_std"], 0.100)

    def test_numeric_manuscript_and_plot_tables_include_delta_and_no_removed_losses(self):
        summary_rows = [
            _summary_row("baseline", "Baseline", "none", rmse=0.10, psd=0.006, hf=5.0),
            _summary_row("l1_0p5", "L1 = 0.5", "time_l1", rmse=0.12, psd=0.007, hf=4.5, time_l1=0.5),
        ]

        numeric_rows = _build_numeric_summary_rows(summary_rows)
        manuscript_rows = _build_manuscript_rows(numeric_rows)
        plot_rows = _build_plot_source_rows(numeric_rows)

        self.assertIn("rmse_mean", numeric_rows[0])
        self.assertIn("rmse_std", numeric_rows[0])
        self.assertIn("num_seeds", numeric_rows[0])
        self.assertIn("seed_list", numeric_rows[0])
        self.assertIn("rmse_delta_vs_baseline", numeric_rows[1])
        self.assertAlmostEqual(numeric_rows[1]["rmse_delta_vs_baseline"], 0.02)
        self.assertAlmostEqual(numeric_rows[1]["rmse_delta_percent_vs_baseline"], 20.0)

        headers = list(manuscript_rows[0].keys())
        self.assertEqual(manuscript_rows[0]["RMSE"], "0.1000 +/- 0.0100")
        self.assertEqual(manuscript_rows[1]["Delta RMSE vs baseline"], "+20.00%")
        self.assertNotIn("derivative", ",".join(headers).lower())
        self.assertNotIn("attach_l2", ",".join(headers).lower())
        self.assertNotIn("attach_temporal", ",".join(headers).lower())

        self.assertEqual(len(plot_rows), len(summary_rows) * len(METRIC_SPECS))
        rmse_plot_row = next(row for row in plot_rows if row["variant"] == "l1_0p5" and row["metric"] == "rmse")
        self.assertEqual(rmse_plot_row["metric_direction"], "lower")
        self.assertAlmostEqual(rmse_plot_row["delta_percent_vs_baseline"], 20.0)


def _summary_row(
    variant: str,
    label: str,
    changed_term: str,
    rmse: float,
    psd: float,
    hf: float,
    time_l1: float = 1.0,
    mse: float = 1.0,
    spectral: float = 0.2,
) -> dict[str, object]:
    return {
        "variant": variant,
        "variant_label": label,
        "changed_term": changed_term,
        "purpose": "test row",
        "time_l1": time_l1,
        "mse": mse,
        "spectral": spectral,
        "num_seeds": 5,
        "seed_list": "42,43,44,45,46",
        "rmse_mean": rmse,
        "rmse_std": 0.01,
        "pearson_mean": 0.90,
        "pearson_std": 0.02,
        "psd_distance_mean": psd,
        "psd_distance_std": 0.0002,
        "hf_ratio_improvement_mean": hf,
        "hf_ratio_improvement_std": 0.1,
        "acc_norm_rmse_mean": 0.11,
        "acc_norm_rmse_std": 0.01,
        "gyr_norm_rmse_mean": 0.12,
        "gyr_norm_rmse_std": 0.02,
    }


if __name__ == "__main__":
    unittest.main()
