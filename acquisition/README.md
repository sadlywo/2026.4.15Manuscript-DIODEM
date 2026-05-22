# SINCT-485 Live Acquisition

This folder contains the live SINCT-485 acquisition path for the DIODEM compensation model.
It polls a WitMotion SINCT-485 IMU over RS485/Modbus RTU, converts the motion registers to
the six channels expected by the trained model, and writes raw plus compensated samples to CSV.

## Hardware Setup

- Connect the SINCT-485 to the computer or Jetson through a USB-RS485 adapter.
- Wire RS485 `A/B` consistently with the adapter labels. If the device does not respond, swap `A/B`.
- Make sure the IMU and adapter share a valid power/ground setup according to the sensor datasheet.
- Close vendor serial tools before running this script, because only one process can own the port.

Default serial settings:

```text
baudrate: 9600
data bits: 8
parity: none
stop bits: 1
timeout: 0.05 s
Modbus slave id: 0x50
polling rate: 40 Hz
```

The reader polls holding registers `0x34-0x39` with Modbus function code `0x03`.
The resulting model input order is fixed:

```text
acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z
```

Acceleration is converted to `m/s^2`; gyroscope values are converted to `rad/s`.

## Windows Example

Install dependencies first:

```powershell
pip install -r requirements.txt
```

Run a 60 second capture on `COM3`:

```powershell
python acquisition/live_compensate.py `
  --port COM3 `
  --duration-sec 60 `
  --checkpoint outputs/supervised_tcn_causal_by_experiment/seed_runs/seed_0042/training/checkpoints/best.pt `
  --device cpu
```

Run with a real-time plotting window:

```powershell
python acquisition/live_compensate.py `
  --port COM3 `
  --duration-sec 60 `
  --plot `
  --plot-window-sec 10 `
  --save-summary-figure
```

If your SINCT-485 uses a different Modbus address:

```powershell
python acquisition/live_compensate.py --port COM3 --slave-id 0x51 --duration-sec 60
```

## Jetson/Linux Example

Install dependencies in the Python environment used by the project:

```bash
pip install -r requirements.txt
```

Run a 60 second capture:

```bash
python acquisition/live_compensate.py \
  --port /dev/ttyUSB0 \
  --duration-sec 60 \
  --checkpoint outputs/supervised_tcn_causal_by_experiment/seed_runs/seed_0042/training/checkpoints/best.pt \
  --device cpu \
  --plot
```

If Linux reports permission denied for `/dev/ttyUSB0`, add the user to the serial group or run with
appropriate device permissions.

## Output

By default, CSV logs are saved under:

```text
outputs/live_capture/sinct485_YYYYMMDD_HHMMSS.csv
```

Each row contains:

```text
timestamp_iso, elapsed_s, seq, warmup,
acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z,
pred_acc_x, pred_acc_y, pred_acc_z, pred_gyr_x, pred_gyr_y, pred_gyr_z,
read_latency_ms, infer_latency_ms, loop_latency_ms, late_by_ms
```

The first 61 samples are marked `warmup=True`, matching the causal TCN receptive field. Keep them
in the log, but analyze steady-state performance separately.

## Real-Time Plotting

Use `--plot` to open a matplotlib window during acquisition. The window shows a 2 x 3 grid:

```text
Acc x, Acc y, Acc z
Gyr x, Gyr y, Gyr z
```

Each panel overlays the raw SINCT-485 value and the model-compensated value. The plot uses a rolling
time window, so long captures remain responsive.

Useful plot options:

```text
--plot-window-sec 10              Show the latest 10 seconds.
--plot-update-interval-ms 100     Redraw at most every 100 ms.
--save-summary-figure             Export a static Nature-style figure after capture.
--summary-figure-stem PATH        Choose the summary figure output path without suffix.
```

You can also create the static figure from an existing CSV:

```powershell
python acquisition/plot_capture.py `
  --input-csv outputs/live_capture/sinct485_YYYYMMDD_HHMMSS.csv `
  --output-stem outputs/live_capture/sinct485_YYYYMMDD_HHMMSS_nature `
  --formats svg,pdf,png,tiff
```

## Useful Options

```text
--output-csv PATH              Choose the CSV output path.
--rate-hz 40                   Polling rate. Keep 40 Hz for the trained model.
--print-every 40               Print one status line every 40 samples. Use 0 for quiet mode.
--max-consecutive-errors 20    Stop after repeated read failures.
--max-samples N                Capture exactly N valid samples for bench testing.
--baudrate 9600                Override serial baud rate.
--timeout 0.05                 Override serial timeout in seconds.
--plot                         Show raw and compensated streams live.
--save-summary-figure          Save static publication-style plots after capture.
```

## Troubleshooting

- No response: check COM port, RS485 `A/B` wiring, power, baudrate, and `--slave-id`.
- CRC errors: check wiring quality, cable length, grounding, and whether another tool is changing the device output mode.
- Port busy: close WitMotion tools, serial monitors, or previous Python processes.
- Sampling loop timeout: inspect `late_by_ms`; if it is consistently positive, reduce terminal output, use a faster host, or move inference to CUDA on Jetson.
- Static signal sanity check: while stationary, acceleration magnitude should be near gravity and gyroscope channels should be near zero.
