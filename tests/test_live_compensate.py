import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from acquisition.live_compensate import CSV_FIELDS, INPUT_CHANNELS, run_capture
from acquisition.sinct485 import Sinct485Error


class FakeReader:
    def __init__(self, samples):
        self.samples = list(samples)
        self.closed = False

    def read_sample(self):
        item = self.samples.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def close(self):
        self.closed = True


class FakeCompensator:
    def __init__(self):
        self.received = []

    def push(self, sample):
        self.received.append(np.asarray(sample, dtype=np.float32))
        return {
            "prediction": np.asarray(sample, dtype=np.float32) + 10.0,
            "latency_ms": 0.25,
        }


def sample_from_base(base):
    return {channel: float(base + index) for index, channel in enumerate(INPUT_CHANNELS)}


class TestLiveCompensate(unittest.TestCase):
    def test_run_capture_preserves_channel_order_and_csv_header(self):
        reader = FakeReader([sample_from_base(1), sample_from_base(11)])
        compensator = FakeCompensator()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_csv = Path(tmpdir) / "capture.csv"
            summary = run_capture(
                reader=reader,
                compensator=compensator,
                output_csv=output_csv,
                rate_hz=40.0,
                max_samples=2,
                print_every=0,
                sleep_fn=lambda _seconds: None,
                output_stream=None,
            )

            self.assertEqual(summary["samples_written"], 2)
            self.assertTrue(reader.closed)
            self.assertTrue(np.allclose(compensator.received[0], np.asarray([1, 2, 3, 4, 5, 6], dtype=np.float32)))

            with output_csv.open("r", encoding="utf-8", newline="") as handle:
                csv_reader = csv.DictReader(handle)
                self.assertEqual(csv_reader.fieldnames, CSV_FIELDS)
                rows = list(csv_reader)

        self.assertEqual(rows[0]["warmup"], "True")
        self.assertEqual(float(rows[0]["pred_acc_x"]), 11.0)
        self.assertIn("loop_latency_ms", rows[0])
        self.assertIn("late_by_ms", rows[0])

    def test_run_capture_stops_after_consecutive_read_errors(self):
        reader = FakeReader([Sinct485Error("no response"), Sinct485Error("no response")])
        compensator = FakeCompensator()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_csv = Path(tmpdir) / "capture.csv"
            with self.assertRaisesRegex(RuntimeError, "consecutive"):
                run_capture(
                    reader=reader,
                    compensator=compensator,
                    output_csv=output_csv,
                    rate_hz=40.0,
                    max_samples=5,
                    max_consecutive_errors=2,
                    print_every=0,
                    sleep_fn=lambda _seconds: None,
                    output_stream=None,
                )

        self.assertTrue(reader.closed)


if __name__ == "__main__":
    unittest.main()
