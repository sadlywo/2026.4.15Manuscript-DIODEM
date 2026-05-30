# Model Deployment Report

## Current Model

- Checkpoint: `outputs/supervised_tcn_causal_by_experiment/seed_runs/seed_0042/training/checkpoints/best.pt`
- Model: `tcn_causal`
- Input shape: `[1, 64, 6]`
- Output shape for first STM32 pass: `[1, 6]`
- Sampling frequency used in training: 40 Hz
- Receptive field from existing streaming report: 61 samples
- Parameter count from existing deployment summary: 101326
- FP32 parameter size from existing deployment summary: 0.3865 MB

## Current Environment Audit

- Audit report: `deploy_stm32/model_audit_report.json`
- Checkpoint exists: yes
- Normalization stats exist: yes
- Cached test vectors exist: yes
- PyTorch status on this machine: blocked by `torch/lib/c10.dll` initialization failure
- ONNX export status: blocked until PyTorch can import and load the checkpoint

## Reproducibility Commands

```powershell
python tools/deploy_stm32/audit_model.py
python tools/deploy_stm32/export_onnx.py
python tools/deploy_stm32/validate_onnx.py
python tools/deploy_stm32/prepare_stm32_vectors.py
```

## STM32Cube.AI Analysis

Record after running STM32Cube.AI:

| Item | Value |
|---|---|
| STM32Cube.AI version | Pending |
| ONNX model | Pending |
| Target MCU | STM32H723ZG |
| Input tensor | Pending |
| Output tensor | Pending |
| Flash estimate | Pending |
| RAM estimate | Pending |
| Activation buffer | Pending |
| MACC | Pending |
| Estimated latency | Pending |
| Unsupported operators | Pending |

## PC and STM32 Numerical Comparison

| Stage | Metric | Pass criterion | Result |
|---|---|---|---|
| PyTorch vs ONNX | max abs error | `< 1e-5`, relax to `< 1e-4` if justified | Pending |
| PyTorch vs ONNX | RMSE | Report per channel | Pending |
| PC FP32 vs STM32 | max abs error | Defined after Cube.AI FP32 run | Pending |
| PC FP32 vs STM32 | RMSE | Report per channel | Pending |

## Real-Time Measurements

| Scenario | Sample rate | Mean infer us | P95 infer us | Max infer us | Drops | Result |
|---|---:|---:|---:|---:|---:|---|
| CSV replay | 40 Hz | Pending | Pending | Pending | Pending | Pending |
| Live IMU | 40 Hz | Pending | Pending | Pending | Pending | Pending |
| Long run | 40 Hz | Pending | Pending | Pending | Pending | Pending |

## INT8 Follow-Up

INT8 should be evaluated only after FP32 is numerically aligned across PyTorch,
ONNX Runtime, and STM32Cube.AI. Calibration data should cover static, vibration,
large-motion, and long-run segments.
