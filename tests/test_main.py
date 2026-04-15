import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from main import (
    compute_signal_metrics,
    extract_sampling_frequency,
    infer_file_type_from_name,
    parse_motion_folder_name,
    sanitize_motion_label,
    standardize_signal_groups,
)


class TestMainHelpers(unittest.TestCase):
    def test_infer_file_type_from_name(self):
        self.assertEqual(
            infer_file_type_from_name("exp01_motion01_imu_nonrigid.csv"),
            "imu_nonrigid",
        )
        self.assertEqual(
            infer_file_type_from_name("exp01_motion01_imu_rigid.csv"),
            "imu_rigid",
        )
        self.assertEqual(infer_file_type_from_name("exp01_motion01_omc.csv"), "omc")

    def test_parse_motion_folder_name(self):
        parsed = parse_motion_folder_name("motion12_shaking")
        self.assertEqual(parsed["motion_index"], "motion12")
        self.assertEqual(parsed["motion_name"], "shaking")

    def test_extract_sampling_frequency(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "sample.csv"
            csv_path.write_text(
                "# sampling frequency: 40\n"
                "# units are seconds/meters/radians/a.u.\n"
                "seg1_acc_x,seg1_acc_y\n"
                "1,2\n",
                encoding="utf-8",
            )
            self.assertEqual(extract_sampling_frequency(csv_path), 40.0)

    def test_standardize_signal_groups(self):
        df = pd.DataFrame(
            {
                "seg1_acc_x": [1.0, 2.0],
                "seg1_acc_y": [2.0, 3.0],
                "seg1_acc_z": [3.0, 4.0],
                "seg1_gyr_x": [0.1, 0.2],
                "seg1_gyr_y": [0.2, 0.3],
                "seg1_gyr_z": [0.3, 0.4],
                "seg1_mag_x": [5.0, 5.0],
                "seg1_mag_y": [6.0, 6.0],
                "seg1_mag_z": [7.0, 7.0],
            }
        )
        groups = standardize_signal_groups(df)
        self.assertIn("seg1", groups)
        self.assertAlmostEqual(groups["seg1"]["acc_norm"].iloc[0], np.sqrt(14.0))
        self.assertAlmostEqual(groups["seg1"]["gyr_norm"].iloc[0], np.sqrt(0.14))

    def test_compute_signal_metrics(self):
        fs = 40.0
        t = np.arange(0.0, 10.0, 1.0 / fs)
        rigid = np.sin(2 * np.pi * 1.0 * t)
        nonrigid = rigid + 0.25 * np.sin(2 * np.pi * 8.0 * t)
        metrics = compute_signal_metrics(rigid, nonrigid, fs)
        self.assertGreater(metrics["rmse"], 0.0)
        self.assertGreater(metrics["high_band_energy_ratio"], 1.0)
        self.assertLessEqual(metrics["pearson_r"], 1.0)

    def test_sanitize_motion_label(self):
        self.assertEqual(sanitize_motion_label("fast_slow_mix"), "fast_slow_mix")
        self.assertEqual(sanitize_motion_label("gait-fast"), "gait_fast")


if __name__ == "__main__":
    unittest.main()
