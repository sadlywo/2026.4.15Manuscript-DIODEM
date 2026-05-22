# DIODEM IMU Artifact Compensation

This repository contains the DIODEM workflow for soft-attachment IMU artifact analysis, supervised compensation model training/evaluation, and SINCT-485 live acquisition with real-time model compensation.

## Directory Map

```text
.
|-- acquisition/                  # SINCT-485 RS485/Modbus acquisition, live compensation, live plotting, CSV-to-figure export
|-- dataset/                      # Raw DIODEM CSV files: rigid IMU, nonrigid IMU, and OMC references
|-- docs/                         # Project notes, deployment guide, paper/patent drafts, and manuscript figures
|-- outputs/                      # Generated metrics, figures, model checkpoints, replay demos, and paper tables
|-- processed*/                   # Cached train/val/test window bundles and normalization stats for each split/setting
|-- project/                      # Main Python package for data preparation, models, training, evaluation, inference, and experiments
|-- tests/                        # Unit and integration tests for analysis, model helpers, acquisition, and live plotting
|-- main.py                       # Original dataset analysis script that scans raw DIODEM files and builds summary outputs
|-- requirements.txt              # Python dependencies
```

## Important Subdirectories

- `project/configs/`: YAML configs for TCN, causal TCN, GRU, and Transformer experiments.
- `project/data/`: pair-table construction, split assignment, sliding-window caches, and normalization helpers.
- `project/models/`: baseline filters plus MLP, GRU, Transformer, and attachment-aware TCN models.
- `project/inference/`: streaming inference wrapper around trained causal checkpoints.
- `project/training/`: training loop, losses, and metrics.
- `project/evaluation/`: checkpoint evaluation, metric summaries, and prediction visualizations.
- `project/experiments/`: split/seed/ablation orchestration helpers.
- `docs/deployment/`: real-time deployment guide.
- `outputs/live_capture/`: local SINCT-485 live captures and exported live-capture figures. This folder is ignored by git because it is experiment-session output.

## Recommended Runtime Model

For real-time use, use the causal TCN checkpoint:

```text
outputs/supervised_tcn_causal_by_experiment/seed_runs/seed_0042/training/checkpoints/best.pt
```

The model expects one 40 Hz sample at a time in this channel order:

```text
acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z
```

The output has the matching compensated channel order:

```text
pred_acc_x, pred_acc_y, pred_acc_z, pred_gyr_x, pred_gyr_y, pred_gyr_z
```

## Real-Time SINCT-485 Command

Install dependencies:

```powershell
pip install -r requirements.txt
```

Windows example with live plotting and post-capture Nature-style figures:

```powershell
python acquisition/live_compensate.py `
  --port COM3 `
  --duration-sec 60 `
  --checkpoint outputs/supervised_tcn_causal_by_experiment/seed_runs/seed_0042/training/checkpoints/best.pt `
  --device cpu `
  --plot `
  --plot-window-sec 10 `
  --save-summary-figure
```

Jetson/Linux example:

```bash
python acquisition/live_compensate.py \
  --port /dev/ttyUSB0 \
  --duration-sec 60 \
  --checkpoint outputs/supervised_tcn_causal_by_experiment/seed_runs/seed_0042/training/checkpoints/best.pt \
  --device cpu \
  --plot \
  --plot-window-sec 10 \
  --save-summary-figure
```

Create publication-style figures from an existing live-capture CSV:

```powershell
python acquisition/plot_capture.py `
  --input-csv outputs/live_capture/sinct485_YYYYMMDD_HHMMSS.csv `
  --output-stem outputs/live_capture/sinct485_YYYYMMDD_HHMMSS_nature `
  --formats svg,pdf,png,tiff
```

## Offline Replay / Model Inference

Replay a prepared six-channel CSV through the causal model:

```powershell
python project/main_realtime_infer.py `
  --checkpoint outputs/supervised_tcn_causal_by_experiment/seed_runs/seed_0042/training/checkpoints/best.pt `
  --mode csv-file `
  --input-csv outputs/replay_demo/exp01_canonical_seg1_nonrigid_6ch.csv `
  --output-csv outputs/replay_demo/exp01_canonical_seg1_predictions.csv `
  --include-input
```

## Development Checks

Run the full test suite:

```powershell
python -m pytest -q
```

Run only acquisition and live-plotting tests:

```powershell
python -m pytest tests/test_sinct485_protocol.py tests/test_live_compensate.py tests/test_live_plotting.py -q
```

## Cleanup Policy

The repository should not track generated Python bytecode or test caches. The following were removed as project hygiene:

- `__pycache__/`
- `*.pyc`
- `.pytest_cache/`

Large scientific outputs, processed caches, checkpoints, and manuscript assets were kept because they are part of experiment reproducibility.
