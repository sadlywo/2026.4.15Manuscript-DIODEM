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
from project.training.metrics import summarize_window_metrics


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


if __name__ == "__main__":
    unittest.main()
