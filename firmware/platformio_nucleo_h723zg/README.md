# DIODEM PlatformIO NUCLEO-H723ZG Firmware

This is an isolated PlatformIO project for the NUCLEO-H723ZG board. It uses the
STM32Cube HAL framework so the later X-CUBE-AI generated files can be integrated
without changing the workflow away from VS Code.

## What This Firmware Does Now

- Configures USART3 on PD8/PD9 for the ST-LINK virtual COM port.
- Prints a boot banner at 115200 baud.
- Accepts one six-axis IMU sample per line:

  ```text
  acc_x,acc_y,acc_z,gyr_x,gyr_y,gyr_z
  ```

- Maintains a 64-sample ring buffer matching the current causal TCN deployment
  window.
- Calls `DiodeM_AI_RunWindow()`. This project currently provides
  `src/ai_inference_stub.c`, which is a buildable placeholder.

## Build and Upload

Install PlatformIO in VS Code, then run from this folder:

```powershell
pio run
pio run -t upload
pio device monitor -p COM6 -b 115200
```

The local command line currently does not have `pio` in PATH, so compilation was
not run during generation.

## Replacing the AI Stub

After exporting the ONNX model and generating STM32Cube.AI C code:

1. Remove `src/ai_inference_stub.c` from the build or rename it.
2. Add the Cube.AI generated `network.*` and `network_data.*` files.
3. Port the implementation from
   `deploy_stm32/stm32_cubeai_template/Core/Src/ai_inference.c`.
4. Keep the `DiodeM_AI_RunWindow()` interface unchanged.

## Serial Test

After upload, open COM6 and send:

```text
0,0,9.80665,0,0,0
```

For the first 63 samples the board replies with `WARMUP`. From sample 64 onward
it replies with compensated output fields. With the current stub, the output is
only a plumbing check, not the trained neural network.

