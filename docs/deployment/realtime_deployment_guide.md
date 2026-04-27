# Real-Time Deployment Guide

## Recommended model

For real hardware deployment, use `tcn_causal`, not the offline-best `transformer`.

Reason:

- `tcn_causal` supports strict step-wise inference through `forward_step()`.
- The project already verified streaming/offline consistency for this model.
- It is small enough for Raspberry Pi or Jetson class devices.
- `transformer` is still the best offline accuracy baseline, but it is not the right first model for live compensation.

## Recommended deployment path

Use this pipeline:

1. Sensor acquisition process reads one IMU sample at a time.
2. The acquisition process formats each sample in the same channel order used during training:
   `acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z`
3. The sample is passed into the causal model.
4. The model returns one compensated sample:
   `pred_acc_x, pred_acc_y, pred_acc_z, pred_gyr_x, pred_gyr_y, pred_gyr_z`
5. Save or forward the compensated output for logging, control, or visualization.

## Input and output contract

### Input

- Sample rate: `40 Hz`
- Input dimension: `6`
- Channel order:
  - `acc_x`
  - `acc_y`
  - `acc_z`
  - `gyr_x`
  - `gyr_y`
  - `gyr_z`

### Output

- Output dimension: `6`
- Target channel order:
  - `pred_acc_x`
  - `pred_acc_y`
  - `pred_acc_z`
  - `pred_gyr_x`
  - `pred_gyr_y`
  - `pred_gyr_z`

The runtime wrapper automatically applies the same normalization used during training and then de-normalizes the prediction back to the physical signal domain.

## New runtime entry point

Use [project/main_realtime_infer.py](/e:/VSCode_Study/2026.4.15Manuscript-DIODEM/project/main_realtime_infer.py).

It supports:

- replay from a CSV file
- sample-by-sample live inference from `stdin`

## Example 1: replay a CSV file

```powershell
python project/main_realtime_infer.py `
  --checkpoint outputs/supervised_tcn_causal_by_experiment/seed_runs/seed_0042/training/checkpoints/best.pt `
  --mode csv-file `
  --input-csv data\my_live_capture.csv `
  --output-csv data\my_live_capture_predictions.csv `
  --include-input
```

Required CSV columns:

```text
acc_x,acc_y,acc_z,gyr_x,gyr_y,gyr_z
```

## Example 2: connect a live sensor process through stdin

If your acquisition process prints one sample per line:

```text
0.12,-0.04,9.78,0.01,-0.02,0.05
```

then you can pipe it into the model:

```powershell
python your_sensor_reader.py | python project/main_realtime_infer.py `
  --checkpoint outputs/supervised_tcn_causal_by_experiment/seed_runs/seed_0042/training/checkpoints/best.pt `
  --mode stdin-stream `
  --stdin-format csv `
  --stdout-format jsonl
```

Each output line will contain one compensated sample and its latency.

## Suggested hardware order

Use this sequence:

1. `x86 laptop / workstation`
2. `Jetson`
3. `Raspberry Pi`

This usually reduces debugging time because the full Python runtime is easier to stabilize on x86 first.

## Suggested real experiment procedure

1. Run the deployed model on recorded CSV data first.
2. Compare raw input and compensated output timing and scale.
3. Then connect the live sensor stream through `stdin`.
4. Log both raw and compensated signals.
5. Measure end-to-end delay separately from pure model latency.

## What to measure on hardware

- per-sample inference latency
- end-to-end delay from sensor arrival to compensated output
- dropped samples
- long-run stability
- CPU usage
- memory usage

## Important caveat

The current deployment path verifies model-side streaming inference.
For a full hardware experiment, you still need a sensor reader process that converts your IMU device output into the six-channel sample format expected by the model.
