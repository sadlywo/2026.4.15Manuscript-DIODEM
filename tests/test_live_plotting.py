import csv
import tempfile
import unittest
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

from acquisition.live_compensate import INPUT_CHANNELS, PREDICTION_CHANNELS, run_capture
from acquisition.live_plot import RollingSignalBuffer
from acquisition.plot_capture import create_capture_figure


class FakeReader:
    def __init__(self, samples):
        self.samples = list(samples)

    def read_sample(self):
        return self.samples.pop(0)

    def close(self):
        pass


class FakeCompensator:
    def push(self, sample):
        values = np.asarray(sample, dtype=np.float32)
        return {"prediction": values + 1.0, "latency_ms": 0.1}


def _sample(base):
    return {channel: float(base + index) for index, channel in enumerate(INPUT_CHANNELS)}


class TestLivePlotting(unittest.TestCase):
    def test_rolling_signal_buffer_keeps_latest_samples_in_channel_order(self):
        buffer = RollingSignalBuffer(max_points=2)
        buffer.append(elapsed_s=0.0, raw_values=[1, 2, 3, 4, 5, 6], pred_values=[7, 8, 9, 10, 11, 12])
        buffer.append(elapsed_s=0.1, raw_values=[11, 12, 13, 14, 15, 16], pred_values=[17, 18, 19, 20, 21, 22])
        buffer.append(elapsed_s=0.2, raw_values=[21, 22, 23, 24, 25, 26], pred_values=[27, 28, 29, 30, 31, 32])

        snapshot = buffer.snapshot()

        self.assertTrue(np.allclose(snapshot.time_s, [0.1, 0.2]))
        self.assertEqual(snapshot.raw.shape, (2, 6))
        self.assertEqual(snapshot.pred.shape, (2, 6))
        self.assertTrue(np.allclose(snapshot.raw[:, 0], [11, 21]))
        self.assertTrue(np.allclose(snapshot.pred[:, 5], [22, 32]))

    def test_run_capture_calls_sample_callback_for_live_plotting(self):
        callbacks = []

        with tempfile.TemporaryDirectory() as tmpdir:
            summary = run_capture(
                reader=FakeReader([_sample(1), _sample(11)]),
                compensator=FakeCompensator(),
                output_csv=Path(tmpdir) / "capture.csv",
                max_samples=2,
                print_every=0,
                sleep_fn=lambda _seconds: None,
                output_stream=None,
                sample_callback=lambda row: callbacks.append(row),
            )

        self.assertEqual(summary["samples_written"], 2)
        self.assertEqual(len(callbacks), 2)
        self.assertEqual(callbacks[0]["acc_x"], 1.0)
        self.assertEqual(callbacks[0]["pred_acc_x"], 2.0)

    def test_create_capture_figure_exports_static_publication_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "capture.csv"
            fieldnames = ["elapsed_s", *INPUT_CHANNELS, *PREDICTION_CHANNELS]
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for index in range(5):
                    row = {"elapsed_s": index * 0.025}
                    for channel_index, channel in enumerate(INPUT_CHANNELS):
                        row[channel] = float(index + channel_index)
                    for channel_index, channel in enumerate(PREDICTION_CHANNELS):
                        row[channel] = float(index + channel_index + 0.5)
                    writer.writerow(row)

            outputs = create_capture_figure(
                csv_path=csv_path,
                output_stem=Path(tmpdir) / "capture_figure",
                formats=("png", "svg"),
            )

            self.assertEqual({path.suffix for path in outputs}, {".png", ".svg"})
            for path in outputs:
                self.assertTrue(path.exists())
                self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
