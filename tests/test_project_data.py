import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from project.data.dataset_builder import build_pair_table, resolve_metadata_csv_path
from project.data.splits import assign_split_labels
from project.data.window_dataset import (
    apply_normalization,
    fit_normalization_stats,
    generate_window_start_indices,
)
from project.evaluation.evaluate import (
    _build_comparison_frame,
    _build_group_delta_frame,
    _get_baseline_model_names,
)
from project.experiments.ablation import build_ablation_config
from project.experiments.runtime import apply_runtime_overrides, build_seed_run_config, resolve_experiment_seeds
from project.models import build_model
from project.training.losses import CompositeLoss
from project.training.metrics import summarize_window_metrics
from project.utils.torch_compat import TORCH_AVAILABLE, torch


class TestProjectDataHelpers(unittest.TestCase):
    def setUp(self):
        self.pairs_df = pd.DataFrame(
            [
                {
                    "kc_type": "arm",
                    "experiment_id": "exp01",
                    "motion_folder": "motion12_shaking",
                    "motion_name": "shaking",
                    "segment_id": "seg3",
                    "rigid_path": "rigid_1.csv",
                    "nonrigid_path": "nonrigid_1.csv",
                    "sampling_frequency": 40.0,
                    "n_samples": 128,
                    "is_anomaly_case": True,
                },
                {
                    "kc_type": "arm",
                    "experiment_id": "exp02",
                    "motion_folder": "motion03_slow1",
                    "motion_name": "slow1",
                    "segment_id": "seg1",
                    "rigid_path": "rigid_2.csv",
                    "nonrigid_path": "nonrigid_2.csv",
                    "sampling_frequency": 40.0,
                    "n_samples": 160,
                    "is_anomaly_case": False,
                },
                {
                    "kc_type": "gait",
                    "experiment_id": "exp10",
                    "motion_folder": "motion02_gait_fast",
                    "motion_name": "gait_fast",
                    "segment_id": "seg2",
                    "rigid_path": "rigid_3.csv",
                    "nonrigid_path": "nonrigid_3.csv",
                    "sampling_frequency": 40.0,
                    "n_samples": 160,
                    "is_anomaly_case": False,
                },
            ]
        )

    def test_assign_split_labels_by_experiment(self):
        config = {
            "strategy": "by_experiment",
            "by_experiment": {
                "train": ["exp02"],
                "val": ["exp01"],
                "test": ["exp10"],
            },
            "anomaly": {"mode": "include_all"},
        }
        labeled = assign_split_labels(self.pairs_df, config)
        self.assertEqual(
            labeled.loc[labeled["experiment_id"] == "exp02", "split"].iloc[0],
            "train",
        )
        self.assertEqual(
            labeled.loc[labeled["experiment_id"] == "exp01", "split"].iloc[0],
            "val",
        )
        self.assertEqual(
            labeled.loc[labeled["experiment_id"] == "exp10", "split"].iloc[0],
            "test",
        )

    def test_generate_window_start_indices(self):
        starts = generate_window_start_indices(length=160, window_size=64, stride=16)
        self.assertEqual(starts, [0, 16, 32, 48, 64, 80, 96])

    def test_assign_split_labels_excludes_all_anomaly_cases(self):
        config = {
            "strategy": "by_experiment",
            "by_experiment": {
                "train": ["exp02"],
                "val": ["exp01"],
                "test": ["exp10"],
            },
            "anomaly": {"mode": "exclude_all"},
        }
        labeled = assign_split_labels(self.pairs_df, config)
        self.assertEqual(set(labeled["experiment_id"]), {"exp02", "exp10"})
        self.assertFalse(labeled["is_anomaly_case"].any())

    def test_assign_split_labels_moves_anomaly_cases_to_test(self):
        config = {
            "strategy": "by_experiment",
            "by_experiment": {
                "train": ["exp01", "exp02"],
                "val": [],
                "test": ["exp10"],
            },
            "anomaly": {"mode": "test_only"},
        }
        labeled = assign_split_labels(self.pairs_df, config)
        exp01_row = labeled.loc[labeled["experiment_id"] == "exp01"].iloc[0]
        self.assertEqual(exp01_row["split"], "test")
        self.assertTrue(bool(exp01_row["is_anomaly_case"]))

    def test_fit_and_apply_per_channel_normalization(self):
        samples = {
            "inputs": np.array(
                [
                    [[1.0, 2.0], [3.0, 4.0]],
                    [[5.0, 6.0], [7.0, 8.0]],
                ],
                dtype=np.float32,
            ),
            "targets": np.array(
                [
                    [[2.0, 3.0], [4.0, 5.0]],
                    [[6.0, 7.0], [8.0, 9.0]],
                ],
                dtype=np.float32,
            ),
        }
        stats = fit_normalization_stats(samples, normalization="per_channel_zscore")
        normalized = apply_normalization(samples, stats, normalization="per_channel_zscore")
        self.assertEqual(tuple(normalized["inputs"].shape), (2, 2, 2))
        self.assertTrue(np.allclose(normalized["inputs"].mean(axis=(0, 1)), 0.0, atol=1e-6))
        self.assertEqual(normalized["inputs"].dtype, np.float32)
        self.assertEqual(normalized["targets"].dtype, np.float32)

    def test_apply_normalization_keeps_float32_with_json_loaded_stats(self):
        samples = {
            "inputs": np.array(
                [
                    [[1.0, 2.0], [3.0, 4.0]],
                    [[5.0, 6.0], [7.0, 8.0]],
                ],
                dtype=np.float32,
            ),
            "targets": np.array(
                [
                    [[2.0, 3.0], [4.0, 5.0]],
                    [[6.0, 7.0], [8.0, 9.0]],
                ],
                dtype=np.float32,
            ),
        }
        stats = {
            "mode": "per_channel_zscore",
            "input_mean": [4.0, 5.0],
            "input_std": [2.2360679, 2.2360679],
            "target_mean": [5.0, 6.0],
            "target_std": [2.2360679, 2.2360679],
        }
        normalized = apply_normalization(samples, stats, normalization="per_channel_zscore")
        self.assertEqual(normalized["inputs"].dtype, np.float32)
        self.assertEqual(normalized["targets"].dtype, np.float32)

    def test_build_pair_table_marks_anomaly_case(self):
        metadata_df = pd.DataFrame(
            [
                {
                    "kc_type": "arm",
                    "experiment_id": "exp01",
                    "motion_folder": "motion12_shaking",
                    "motion_name": "shaking",
                    "file_type": "imu_rigid",
                    "path": "rigid.csv",
                    "sampling_frequency": 40.0,
                    "n_samples": 128,
                },
                {
                    "kc_type": "arm",
                    "experiment_id": "exp01",
                    "motion_folder": "motion12_shaking",
                    "motion_name": "shaking",
                    "file_type": "imu_nonrigid",
                    "path": "nonrigid.csv",
                    "sampling_frequency": 40.0,
                    "n_samples": 128,
                },
            ]
        )
        selected_df = pd.DataFrame(
            [
                {
                    "kc_type": "arm",
                    "experiment_id": "exp01",
                    "motion_folder": "motion12_shaking",
                    "segment": "seg3",
                    "selection_reason": "anomaly_case_exp01_seg3",
                }
            ]
        )
        pairs = build_pair_table(metadata_df, selected_df)
        self.assertEqual(len(pairs), 5)
        anomaly_rows = pairs[pairs["segment_id"] == "seg3"]
        self.assertTrue(anomaly_rows["is_anomaly_case"].iloc[0])

    def test_summarize_window_metrics_groups_by_motion(self):
        predictions = np.array(
            [
                [[1.0, 1.0], [1.0, 1.0]],
                [[2.0, 2.0], [2.0, 2.0]],
            ],
            dtype=np.float32,
        )
        targets = np.array(
            [
                [[0.0, 0.0], [0.0, 0.0]],
                [[1.0, 1.0], [1.0, 1.0]],
            ],
            dtype=np.float32,
        )
        inputs = np.array(
            [
                [[2.0, 2.0], [2.0, 2.0]],
                [[3.0, 3.0], [3.0, 3.0]],
            ],
            dtype=np.float32,
        )
        metadata = [
            {"motion_name": "slow1", "segment_id": "seg1", "experiment_id": "exp01"},
            {"motion_name": "fast", "segment_id": "seg2", "experiment_id": "exp02"},
        ]
        summary = summarize_window_metrics(
            predictions=predictions,
            targets=targets,
            inputs=inputs,
            metadata=metadata,
            channels=["acc_x", "gyr_x"],
            sampling_frequency=40.0,
        )
        self.assertIn("overall", summary)
        self.assertIn("per_motion", summary)
        self.assertEqual(set(summary["per_motion"]["motion_name"]), {"slow1", "fast"})

    def test_resolve_metadata_csv_path_recovers_from_old_absolute_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir) / "dataset"
            csv_path = dataset_root / "arm" / "exp02" / "motion01_canonical1" / "exp02_motion01_imu_rigid.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            csv_path.write_text(
                "# sampling frequency: 40\n"
                "# units are seconds/meters/radians/a.u.\n"
                "seg1_acc_x,seg1_acc_y,seg1_acc_z,seg1_gyr_x,seg1_gyr_y,seg1_gyr_z\n"
                "1,2,3,4,5,6\n",
                encoding="utf-8",
            )
            pair_row = pd.Series(
                {
                    "kc_type": "arm",
                    "experiment_id": "exp02",
                    "motion_folder": "motion01_canonical1",
                    "rigid_path": r"z:\old_machine\dataset\arm\exp02\motion01_canonical1\exp02_motion01_imu_rigid.csv",
                }
            )
            resolved = resolve_metadata_csv_path(pair_row, "rigid_path", dataset_root)
            self.assertEqual(resolved.resolve(), csv_path.resolve())

    def test_get_baseline_model_names_skips_primary_and_duplicates(self):
        config = {
            "model_name": "tcn",
            "evaluation": {"baseline_models": ["identity", "lowpass", "butterworth", "identity", "tcn"]},
        }
        self.assertEqual(_get_baseline_model_names(config), ["identity", "lowpass", "butterworth"])

    def test_build_comparison_frame_marks_model_roles(self):
        frame = _build_comparison_frame(
            overall_by_model={
                "tcn": {"rmse_mean": 0.1, "pearson_mean": 0.9, "num_windows": 10},
                "gru": {"rmse_mean": 0.11, "pearson_mean": 0.88, "num_windows": 10},
                "identity": {"rmse_mean": 0.3, "pearson_mean": 0.6, "num_windows": 10},
            },
            primary_model_name="tcn",
            model_roles={"gru": "trained_comparison"},
        )
        self.assertEqual(list(frame["model_name"]), ["tcn", "gru", "identity"])
        self.assertEqual(list(frame["model_role"]), ["trained", "trained_comparison", "baseline"])

    def test_build_group_delta_frame_computes_metric_deltas(self):
        candidate_frame = pd.DataFrame(
            [
                {"motion_name": "fast", "rmse_mean": 0.10, "pearson_mean": 0.92, "psd_distance_mean": 0.01},
                {"motion_name": "slow", "rmse_mean": 0.05, "pearson_mean": 0.95, "psd_distance_mean": 0.005},
            ]
        )
        reference_frame = pd.DataFrame(
            [
                {"motion_name": "fast", "rmse_mean": 0.14, "pearson_mean": 0.88, "psd_distance_mean": 0.02},
                {"motion_name": "slow", "rmse_mean": 0.07, "pearson_mean": 0.93, "psd_distance_mean": 0.009},
            ]
        )
        delta_frame = _build_group_delta_frame(
            candidate_frame=candidate_frame,
            reference_frame=reference_frame,
            group_column="motion_name",
            candidate_name="tcn",
            reference_name="lowpass",
        )
        fast_row = delta_frame.loc[delta_frame["motion_name"] == "fast"].iloc[0]
        self.assertAlmostEqual(float(fast_row["rmse_mean_delta_tcn_minus_lowpass"]), -0.04, places=6)
        self.assertAlmostEqual(float(fast_row["pearson_mean_delta_tcn_minus_lowpass"]), 0.04, places=6)

    def test_tcn_attachment_model_returns_latent_state_bundle(self):
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch is not available in this environment.")
        model = build_model(
            model_name="tcn",
            input_dim=6,
            output_dim=6,
            model_config={
                "hidden_dim": 16,
                "num_layers": 2,
                "kernel_size": 3,
                "dropout": 0.0,
                "attach_latent_dim": 4,
            },
        )
        batch = torch.randn(3, 8, 6)
        outputs = model(batch)
        self.assertIsInstance(outputs, dict)
        self.assertEqual(tuple(outputs["predictions"].shape), (3, 8, 6))
        self.assertEqual(tuple(outputs["residual"].shape), (3, 8, 6))
        self.assertEqual(tuple(outputs["z_attach"].shape), (3, 4))
        self.assertEqual(tuple(outputs["z_attach_sequence"].shape), (3, 8, 4))

    def test_classic_filter_baselines_return_expected_shapes(self):
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch is not available in this environment.")
        batch = torch.randn(2, 64, 6)
        shared_config = {
            "sampling_frequency": 40.0,
            "butter_cutoff_hz": 5.0,
            "butter_order": 2,
            "savgol_window_length": 7,
            "savgol_polyorder": 2,
            "wiener_window_size": 5,
        }
        for model_name in ("butterworth", "savgol", "wiener"):
            model = build_model(model_name=model_name, input_dim=6, output_dim=6, model_config=shared_config)
            outputs = model(batch)
            self.assertEqual(tuple(outputs.shape), (2, 64, 6))

    def test_tcn_model_supports_disabling_attachment_latent(self):
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch is not available in this environment.")
        model = build_model(
            model_name="tcn",
            input_dim=6,
            output_dim=6,
            model_config={
                "hidden_dim": 16,
                "num_layers": 2,
                "kernel_size": 3,
                "dropout": 0.0,
                "attach_latent_dim": 0,
            },
        )
        batch = torch.randn(2, 8, 6)
        outputs = model(batch)
        self.assertEqual(tuple(outputs["predictions"].shape), (2, 8, 6))
        self.assertEqual(tuple(outputs["residual"].shape), (2, 8, 6))
        self.assertNotIn("z_attach", outputs)
        self.assertNotIn("z_attach_sequence", outputs)

    def test_composite_loss_accepts_attachment_aux_outputs(self):
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch is not available in this environment.")
        criterion = CompositeLoss(
            {
                "time_l1": 1.0,
                "mse": 0.5,
                "derivative": 0.25,
                "spectral": 0.1,
                "attach_l2": 0.01,
                "attach_temporal": 0.01,
            }
        )
        predictions = torch.randn(2, 10, 3)
        targets = torch.randn(2, 10, 3)
        aux_outputs = {
            "z_attach": torch.randn(2, 4),
            "z_attach_sequence": torch.randn(2, 10, 4),
        }
        terms = criterion(predictions, targets, aux_outputs=aux_outputs)
        self.assertIn("l1", terms)
        self.assertIn("derivative", terms)
        self.assertIn("spectral", terms)
        self.assertIn("attach_l2", terms)
        self.assertIn("attach_temporal", terms)
        self.assertGreaterEqual(float(terms["total"]), 0.0)

    def test_gru_and_transformer_models_return_prediction_bundle(self):
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch is not available in this environment.")
        batch = torch.randn(2, 12, 6)

        gru_model = build_model(
            model_name="gru",
            input_dim=6,
            output_dim=6,
            model_config={"gru_hidden_dim": 16, "gru_num_layers": 2, "dropout": 0.0},
        )
        gru_outputs = gru_model(batch)
        self.assertIsInstance(gru_outputs, dict)
        self.assertEqual(tuple(gru_outputs["predictions"].shape), (2, 12, 6))
        self.assertEqual(tuple(gru_outputs["residual"].shape), (2, 12, 6))

        transformer_model = build_model(
            model_name="transformer",
            input_dim=6,
            output_dim=6,
            model_config={
                "transformer_model_dim": 16,
                "transformer_num_layers": 2,
                "transformer_num_heads": 4,
                "transformer_ff_dim": 32,
                "dropout": 0.0,
            },
        )
        transformer_outputs = transformer_model(batch)
        self.assertIsInstance(transformer_outputs, dict)
        self.assertEqual(tuple(transformer_outputs["predictions"].shape), (2, 12, 6))
        self.assertEqual(tuple(transformer_outputs["residual"].shape), (2, 12, 6))

    def test_build_ablation_config_disables_baseline_comparisons_and_applies_overrides(self):
        base_config = {
            "repo_root": "/tmp/repo",
            "outputs_root": "outputs/supervised",
            "model_name": "tcn",
            "model": {"attach_latent_dim": 8},
            "loss_weights": {
                "time_l1": 1.0,
                "mse": 0.5,
                "derivative": 0.3,
                "spectral": 0.2,
                "attach_l2": 0.001,
                "attach_temporal": 0.001,
            },
            "evaluation": {
                "checkpoint_name": "best.pt",
                "baseline_models": ["identity", "lowpass"],
                "trained_model_checkpoints": [{"label": "gru", "checkpoint": "/tmp/gru.pt"}],
            },
        }
        variant = {
            "name": "no_attachment_latent",
            "description": "Disable attachment latent code.",
            "overrides": {
                "model": {"attach_latent_dim": 0},
                "loss_weights": {"attach_l2": 0.0, "attach_temporal": 0.0},
            },
        }
        config = build_ablation_config(base_config, variant, outputs_root="outputs/supervised_ablations/no_attachment_latent")
        self.assertEqual(config["ablation_variant"], "no_attachment_latent")
        self.assertEqual(config["model"]["attach_latent_dim"], 0)
        self.assertEqual(config["loss_weights"]["attach_l2"], 0.0)
        self.assertEqual(config["evaluation"]["baseline_models"], [])
        self.assertEqual(config["evaluation"]["trained_model_checkpoints"], [])

    def test_runtime_overrides_suffix_paths_and_apply_seed_runs(self):
        base_config = {
            "seed": 42,
            "split_strategy": "by_experiment",
            "anomaly": {"mode": "exclude_all"},
            "processed_root": "processed",
            "outputs_root": "outputs/supervised",
        }
        overridden = apply_runtime_overrides(
            base_config,
            split_strategy="by_motion_type",
            anomaly_mode="test_only",
            run_name=None,
        )
        self.assertEqual(overridden["split_strategy"], "by_motion_type")
        self.assertEqual(overridden["anomaly"]["mode"], "test_only")
        self.assertIn("processed_by_motion_type_anomaly_test_only", overridden["processed_root"])
        self.assertIn("supervised_by_motion_type_anomaly_test_only", overridden["outputs_root"])

        seeds = resolve_experiment_seeds(overridden, explicit_seeds=[42, 43, 42])
        self.assertEqual(seeds, [42, 43])

        seed_config = build_seed_run_config(overridden, seed=43, multi_seed=True)
        self.assertEqual(seed_config["seed"], 43)
        self.assertIn("seed_runs", seed_config["outputs_root"])


if __name__ == "__main__":
    unittest.main()
