# STM32 Real-Time Protocol

## Text CSV Protocol

Use this mode first because it is easy to inspect with a serial terminal.

PC to STM32, one sample per line:

```text
seq,acc_x,acc_y,acc_z,gyr_x,gyr_y,gyr_z
0,0.012,-0.040,9.781,0.001,-0.002,0.005
```

STM32 to PC, one compensated sample per line:

```text
seq,pred_acc_x,pred_acc_y,pred_acc_z,pred_gyr_x,pred_gyr_y,pred_gyr_z,infer_us,status
0,0.010,-0.038,9.779,0.001,-0.002,0.004,850,OK
```

Fields:

- `seq`: monotonically increasing unsigned integer.
- `acc_*`: accelerometer values in m/s^2.
- `gyr_*`: gyroscope values in rad/s.
- `pred_*`: denormalized compensated model output in the same physical units.
- `infer_us`: measured inference time in microseconds.
- `status`: `WARMUP`, `OK`, or `ERR`.

## Binary Protocol

Use binary framing after the CSV protocol has passed PC-vs-STM32 comparison.

Frame layout, little endian:

```text
magic_u16 = 0x4449
version_u8 = 1
type_u8 = 1 input, 2 output, 3 error
seq_u32
payload_len_u16
payload float32[6] for input or output
crc16_modbus_u16 over all previous bytes
```

The MCU should reject frames with a bad magic, unsupported version, unexpected
payload length, or CRC mismatch. The host should treat missing `seq` values as
dropped samples.

## Timing Rules

- The trained model is based on 40 Hz data. If the IMU runs faster, resample or
  decimate to 40 Hz before feeding the model unless a retrained model is used.
- First-stage deployment uses a 64-sample window, so the first valid output is
  available after the ring buffer is warm.
- A later layer-wise state-buffer implementation can remove the repeated window
  compute but must match the Python `forward_step()` reference.

