# DIODEM Supervised Compensation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a runnable PyTorch project skeleton for strongly supervised motion-artifact compensation from nonrigid IMU windows to rigid IMU windows on DIODEM.

**Architecture:** Create a new `project/` package that reads the existing `outputs/metadata_summary.csv` and `outputs/selected_examples.csv`, builds strictly paired segment-level windows, caches train/val/test samples, trains baseline models with a unified `[B, T, C] -> [B, T, C]` interface, and evaluates grouped metrics plus prediction figures. Keep data preparation mostly NumPy/Pandas-based so core validation remains runnable even if the local Torch runtime is temporarily unavailable.

**Tech Stack:** Python, pathlib, pandas, numpy, scipy, matplotlib, PyYAML, PyTorch

---

### Task 1: Add failing tests for data helpers

**Files:**
- Create: `tests/test_project_data.py`
- Create: `project/data/splits.py`
- Create: `project/data/window_dataset.py`
- Create: `project/data/dataset_builder.py`

**Step 1: Write the failing test**

Write tests for:
- strict by-experiment splitting with no leakage
- sliding-window index generation
- normalization stat fitting and application
- anomaly flag propagation from selected-example metadata

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_project_data -v`
Expected: FAIL because the `project` package and helpers do not exist yet.

**Step 3: Write minimal implementation**

Implement the pure-Python split and window helpers first, then the pair-table builder.

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_project_data -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/test_project_data.py project/data
git commit -m "feat: add DIODEM supervised data pipeline helpers"
```

### Task 2: Add configuration and IO scaffolding

**Files:**
- Create: `project/configs/default.yaml`
- Create: `project/utils/io.py`
- Create: `project/utils/seed.py`
- Create: `project/utils/logger.py`
- Create: `project/__init__.py`
- Create: `project/data/__init__.py`
- Create: `project/utils/__init__.py`

**Step 1: Write the failing test**

Add a test for YAML loading and required config fields.

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_project_data -v`
Expected: FAIL because the config helpers or file are missing.

**Step 3: Write minimal implementation**

Implement YAML loading, filesystem helpers, IMU CSV readers, channel selection, and reproducibility helpers.

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_project_data -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add project/configs/default.yaml project/utils tests/test_project_data.py
git commit -m "feat: add config and IO utilities"
```

### Task 3: Implement baseline model interfaces

**Files:**
- Create: `project/models/__init__.py`
- Create: `project/models/baselines.py`
- Create: `project/models/mlp_model.py`
- Create: `project/models/tcn_model.py`

**Step 1: Write the failing test**

Add a smoke test for model factory names and expected output shapes when Torch is available.

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_project_data -v`
Expected: FAIL or SKIP on the missing model factory.

**Step 3: Write minimal implementation**

Implement `IdentityBaseline`, `LowPassBaseline`, `LinearProjectionBaseline`, `MLPBaseline`, and `TCNBaseline` with a shared forward contract.

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_project_data -v`
Expected: PASS for data tests and shape smoke test when Torch imports successfully.

**Step 5: Commit**

```bash
git add project/models tests/test_project_data.py
git commit -m "feat: add DIODEM compensation baselines"
```

### Task 4: Implement training and evaluation pipeline

**Files:**
- Create: `project/training/__init__.py`
- Create: `project/training/losses.py`
- Create: `project/training/metrics.py`
- Create: `project/training/engine.py`
- Create: `project/training/train.py`
- Create: `project/evaluation/__init__.py`
- Create: `project/evaluation/evaluate.py`
- Create: `project/evaluation/visualize_predictions.py`
- Create: `project/main_train.py`
- Create: `project/main_eval.py`

**Step 1: Write the failing test**

Add a smoke test for grouped-metric aggregation that does not require a full training run.

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_project_data -v`
Expected: FAIL because grouped metric helpers are missing.

**Step 3: Write minimal implementation**

Implement composite losses, grouped evaluation, checkpointing, early stopping, prediction export, and figure generation.

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_project_data -v`
Expected: PASS for the covered helpers.

**Step 5: Commit**

```bash
git add project/training project/evaluation project/main_train.py project/main_eval.py tests/test_project_data.py
git commit -m "feat: add DIODEM supervised training and evaluation pipeline"
```

### Task 5: Verify end-to-end entry points

**Files:**
- Modify: `project/*` as needed

**Step 1: Build cached samples**

Run: `python -m project.data.dataset_builder --config project/configs/default.yaml`
Expected: processed train/val/test caches and normalization stats are created.

**Step 2: Run unit tests**

Run: `python -m unittest tests.test_main tests.test_project_data -v`
Expected: PASS.

**Step 3: Run static compilation**

Run: `python -m py_compile project/main_train.py project/main_eval.py`
Expected: PASS.

**Step 4: Run train/eval if Torch runtime is healthy**

Run: `python project/main_train.py --config project/configs/default.yaml`
Expected: checkpoints and training curves are created.

Run: `python project/main_eval.py --config project/configs/default.yaml`
Expected: JSON/CSV metrics and prediction figures are created.

**Step 5: Commit**

```bash
git add project tests docs/plans
git commit -m "feat: add DIODEM supervised artifact compensation project skeleton"
```
