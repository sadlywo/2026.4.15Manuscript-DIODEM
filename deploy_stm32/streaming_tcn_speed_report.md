# Streaming Causal TCN STM32 Speed Report

Generated at: 2026-06-02 16:04:58

## Experiment Configuration

- Board: NUCLEO-H723ZG
- Serial port: COM6 @ 115200 baud
- Trials: 5
- Input stream: 6-axis IMU at 40 Hz
- Window index: 0
- Host sample delay during replay: 0.15 s
- Pytest command: `D:\Anaconda3\envs\pinn_imu\python.exe -m pytest tests/test_deploy_stm32_tools.py tests/test_platformio_nucleo_h723zg_project.py tests/test_streaming_tcn_experiment_runner.py -q`
- PlatformIO build command: `C:\Users\XIAO\.platformio\penv\Scripts\pio.exe run`
- PlatformIO upload command: `C:\Users\XIAO\.platformio\penv\Scripts\pio.exe run -t upload`

## Toolchain Results

- pytest: return code 0
- PlatformIO build: return code 0
- PlatformIO upload: return code 0
- RAM: 10.6% (34848 / 327680 bytes)
- Flash: 41.4% (434564 / 1048576 bytes)

## Measured on STM32

- Streaming TCN latency: 1.39 +/- 0.0008367 ms
- Raw inference time: 1390 +/- 0.8367 us
- Max abs error: 4.77e-07 +/- 0
- RMSE: 2.41e-07 +/- 0
- Effective max sampling frequency: 719.53 Hz
- 40 Hz period: 25.00 ms
- Compute utilization at 40 Hz: 5.56%
- 40 Hz real-time verdict: PASS

## Previous measured baseline

- Cube.AI 64-point window latency: 93.615 ms
- Streaming forward_step speed-up: 67.4x

## Model-level comparison

| Architecture | Scope | Params | FP32 MB | INT8 MB | PC per-window ms | STM32 latency ms | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| TCN-causal | Measured on STM32 | 101326 | 0.3865 | not measured | not measured | 1.39 | Handwritten streaming forward_step on NUCLEO-H723ZG. |
| GRU-causal | Model-level comparison | 152070 | 0.5801 | not measured | 3.296 | not measured | From outputs\causal_model_comparison\gru_causal_by_motion_type\evaluation\metrics\multiseed_model_deployment_summary.csv; not measured on STM32 in this stage. |
| LSTM-causal | Model-level comparison | 202502 | 0.7725 | not measured | 1.487 | not measured | From outputs\causal_model_comparison\lstm_causal_by_motion_type\evaluation\metrics\multiseed_model_deployment_summary.csv; not measured on STM32 in this stage. |
| Transformer-causal | Model-level comparison | 399110 | 1.522 | not measured | 1.663 | not measured | From outputs\causal_model_comparison\transformer_causal_by_motion_type\evaluation\metrics\multiseed_model_deployment_summary.csv; not measured on STM32 in this stage. |
| MLP-causal | Model-level comparison | 18182 | 0.06936 | not measured | 0.1278 | not measured | From outputs\causal_model_comparison\mlp_causal_by_motion_type\evaluation\metrics\multiseed_model_deployment_summary.csv; not measured on STM32 in this stage. |

## INT8 Status

- No measured INT8 STM32 artifact is included in this experiment stage.
- INT8 model sizes are therefore reported as `not measured`, not inferred measurements.
- PTQ/QAT should be evaluated only after the FP32 streaming path is locked and compared against the same golden vectors.

## Interpretation

- The measured computation time is far below the 25.00 ms period required for 40 Hz streaming.
- The TCN inference itself is not the bottleneck for this setup.
- The serial protocol and host-to-board data transfer can dominate replay wall time, especially with text CSV frames.
- Recommendation: keep the handwritten streaming causal TCN path for NUCLEO-H723ZG FP32 deployment, then evaluate binary serial framing and INT8 only if system-level bandwidth or memory demands require it.
