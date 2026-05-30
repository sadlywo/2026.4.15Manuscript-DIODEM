# STM32 Deployment README

This folder contains reproducible deployment assets for converting the trained
FP32 causal TCN into a NUCLEO-H723ZG validation workflow.

## Recommended Order

1. Audit the current model and data paths:

   ```powershell
   python tools/deploy_stm32/audit_model.py
   ```

2. Fix the Python environment if the audit reports that PyTorch cannot be
   imported. The current local failure observed during planning was a Windows
   DLL initialization error from `torch/lib/c10.dll`.

3. Export fixed-shape ONNX after PyTorch works:

   ```powershell
   python tools/deploy_stm32/export_onnx.py `
     --checkpoint outputs/supervised_tcn_causal_by_experiment/seed_runs/seed_0042/training/checkpoints/best.pt `
     --output deploy_stm32/onnx/tcn_causal_last_step.onnx `
     --output-mode last_step `
     --opset 17
   ```

4. Validate ONNX against PyTorch:

   ```powershell
   python tools/deploy_stm32/validate_onnx.py `
     --onnx-model deploy_stm32/onnx/tcn_causal_last_step.onnx `
     --output deploy_stm32/onnx_validation_report.json
   ```

5. Analyze the ONNX model with STM32Cube.AI for NUCLEO-H723ZG and record the
   generated Flash, RAM, activation buffer, and estimated inference latency in
   `deploy_stm32/model_deployment_report.md`.

6. Copy the files under `deploy_stm32/stm32_cubeai_template/Core` into the
   CubeMX project only after X-CUBE-AI has generated its `network.*` files.

## Model Contract

- Input window: `[1, 64, 6]`
- Input channel order: `acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z`
- Gyroscope unit: rad/s
- Accelerometer unit: m/s^2
- ONNX first-stage output: last compensated timestep `[1, 6]`
- Runtime strategy: 64-sample causal ring buffer, output one compensated sample
  for each new IMU sample once the buffer is warm.

## Acceptance Gates

- PyTorch can load the checkpoint.
- PyTorch streaming and window output agree on cached test windows.
- ONNX Runtime output agrees with PyTorch wrapper output.
- STM32 output agrees with PC FP32 reference vectors.
- Measured end-to-end loop time is below the target sampling period.

