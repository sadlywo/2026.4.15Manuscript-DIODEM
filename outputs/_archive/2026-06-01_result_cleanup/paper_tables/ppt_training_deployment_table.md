| Module | Setting | Value | Presentation Note |
| --- | --- | --- | --- |
| Loss | Total loss | `L_total = 1.0*L1 + 0.5*MSE + 0.3*L_der + 0.2*L_spec + 0.001*L_attach_l2 + 0.001*L_attach_temp` | Composite objective for waveform + temporal + spectral consistency |
| Loss | Time-domain losses | `L1 (1.0) + MSE (0.5)` | Signal reconstruction in the time domain |
| Loss | Derivative loss | `0.3` | Matches temporal first-order differences |
| Loss | Spectral loss | `0.2` | Matches FFT magnitude spectra |
| Loss | Attachment regularization | `attach_l2 = 0.001, attach_temporal = 0.001` | Stabilizes latent attachment state |
| Training | Batch size | `64` | Mini-batch training |
| Training | Epochs / patience | `50 epochs / patience 8` | Early stopping based training |
| Training | Learning rate / weight decay | `0.001 / 0.0001` | Current optimizer setting |
| Deployment | Parameter count | `101,326` | Lightweight model footprint |
| Deployment | Model size | `0.387 MB FP32` | Compact enough for embedded Linux devices |
| Deployment | CPU inference | `1.301 +- 0.002 ms per window` | Measured in current evaluation pipeline |
| Streaming | Streaming latency | `mean 1.191 ms, p95 1.209 ms` | Current real-time inference evidence |
| Streaming | Streaming consistency | `RMSE ~ 3.4e-7, max abs ~ 1.8e-6` | Streaming output is almost identical to offline causal forward |
| Model role | Offline upper bound | `Transformer` | Best offline reconstruction accuracy |
| Model role | Real-time candidate | `TCN-causal` | Best balance between accuracy and deployability |
