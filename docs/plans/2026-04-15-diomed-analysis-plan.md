# DIODEM Analysis Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a runnable Python analysis pipeline for the DIODEM dataset that scans files, reads IMU/OMC CSVs robustly, computes rigid/nonrigid comparison metrics, selects representative examples, and generates figures plus a report.

**Architecture:** Use a single-file script `main.py` so the user can run it directly in a fresh repository. Organize the script into small pure functions for scanning, CSV parsing, segment signal extraction, pairwise metrics, representative sample selection, figure creation, and report generation. Add a focused `unittest` file for the parsing and signal-grouping helpers to keep the core assumptions executable and verifiable.

**Tech Stack:** Python, pathlib, re, pandas, numpy, scipy, matplotlib, unittest

---

### Task 1: Create executable test coverage for parsing helpers

**Files:**
- Create: `tests/test_main.py`
- Modify: `main.py`

**Step 1: Write the failing test**

```python
import unittest

from main import infer_file_type_from_name, parse_motion_folder_name


class TestParsingHelpers(unittest.TestCase):
    def test_infer_file_type_from_name(self):
        self.assertEqual(
            infer_file_type_from_name("exp01_motion01_imu_nonrigid.csv"),
            "imu_nonrigid",
        )

    def test_parse_motion_folder_name(self):
        parsed = parse_motion_folder_name("motion12_shaking")
        self.assertEqual(parsed["motion_index"], "motion12")
        self.assertEqual(parsed["motion_name"], "shaking")
```

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_main -v`
Expected: FAIL because `main.py` and the helper functions do not exist yet.

**Step 3: Write minimal implementation**

Create `main.py` with the tested helper functions and import-safe structure.

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_main -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/test_main.py main.py docs/plans/2026-04-15-diomed-analysis-plan.md
git commit -m "feat: add DIODEM analysis pipeline scaffolding"
```

### Task 2: Add robust metadata scanning and CSV loading

**Files:**
- Modify: `main.py`
- Test: `tests/test_main.py`

**Step 1: Write the failing test**

Add tests for sampling-frequency parsing, numeric conversion, and segment discovery from representative headers.

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_main -v`
Expected: FAIL with missing reader or grouping functions.

**Step 3: Write minimal implementation**

Implement CSV readers, metadata builders, warning collection, and time-vector generation.

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_main -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/test_main.py main.py
git commit -m "feat: add DIODEM metadata scanning and CSV loading"
```

### Task 3: Implement pairwise rigid/nonrigid metrics and selection

**Files:**
- Modify: `main.py`
- Test: `tests/test_main.py`

**Step 1: Write the failing test**

Add a small synthetic test for RMSE/correlation/high-frequency metrics on paired signals.

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_main -v`
Expected: FAIL because the metrics function is missing or incomplete.

**Step 3: Write minimal implementation**

Implement per-segment IMU pairing, Welch PSD summaries, aggregate metrics, and representative sample selection.

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_main -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/test_main.py main.py
git commit -m "feat: add DIODEM pairwise metrics and sample selection"
```

### Task 4: Implement figures, tables, and report outputs

**Files:**
- Modify: `main.py`

**Step 1: Write the failing test**

Add at least one smoke test that the output path formatter returns a stable figure filename.

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_main -v`
Expected: FAIL because the formatter is missing.

**Step 3: Write minimal implementation**

Implement output directory creation, overview figures, representative sample plots, OMC trend plotting, and report generation.

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_main -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/test_main.py main.py
git commit -m "feat: add DIODEM figures and summary report"
```

### Task 5: Verify on the real dataset

**Files:**
- Modify: `main.py` if needed

**Step 1: Run the script**

Run: `python main.py`
Expected: metadata CSV, pairwise metrics CSV, selected examples CSV, figures, and `outputs/report.txt` are created.

**Step 2: Fix runtime issues**

Address parsing edge cases, warnings, or plotting issues discovered on the actual dataset.

**Step 3: Run tests and diagnostics again**

Run: `python -m unittest tests.test_main -v`
Expected: PASS.

**Step 4: Perform final verification**

Run: `python main.py`
Expected: clean completion with concise stage logs.

**Step 5: Commit**

```bash
git add main.py tests/test_main.py outputs
git commit -m "feat: add complete DIODEM exploratory analysis pipeline"
```
