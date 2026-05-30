**Supplementary Table Sx. Loss-weight sensitivity study design.**

This table defines the `local` sensitivity preset used to assess whether the final composite-loss coefficients produce a robust empirical trade-off. Unless otherwise stated, only the listed loss coefficients are changed. Seeds: 42, 43, 44, 45, 46.

| Variant | Label | Changed term | Purpose | time_l1 | mse | spectral |
|---|---|---|---|---|---|---|
| baseline | Baseline | none | Final loss setting used in the main paper. | 1 | 0.5 | 0.2 |
| l1_0p25 | L1 = 0.25 | time_l1 | Reduce the L1 reconstruction weight to probe point-wise robustness. | 0.25 | 0.5 | 0.2 |
| l1_0p5 | L1 = 0.5 | time_l1 | Reduce the L1 reconstruction weight to probe point-wise robustness. | 0.5 | 0.5 | 0.2 |
| l1_1p5 | L1 = 1.5 | time_l1 | Increase the L1 reconstruction weight to probe point-wise robustness. | 1.5 | 0.5 | 0.2 |
| l1_2p0 | L1 = 2 | time_l1 | Increase the L1 reconstruction weight to probe point-wise robustness. | 2 | 0.5 | 0.2 |
| mse_0p125 | MSE = 0.125 | mse | Reduce the MSE contribution to probe the quadratic reconstruction penalty. | 1 | 0.125 | 0.2 |
| mse_0p25 | MSE = 0.25 | mse | Reduce the MSE contribution to probe the quadratic reconstruction penalty. | 1 | 0.25 | 0.2 |
| mse_0p75 | MSE = 0.75 | mse | Increase the MSE contribution to probe the quadratic reconstruction penalty. | 1 | 0.75 | 0.2 |
| mse_1p0 | MSE = 1 | mse | Increase the MSE contribution to probe the quadratic reconstruction penalty. | 1 | 1 | 0.2 |
| spec_0p05 | Spectral = 0.05 | spectral | Reduce spectral consistency to probe frequency-domain alignment. | 1 | 0.5 | 0.05 |
| spec_0p1 | Spectral = 0.1 | spectral | Reduce spectral consistency to probe frequency-domain alignment. | 1 | 0.5 | 0.1 |
| spec_0p3 | Spectral = 0.3 | spectral | Increase spectral consistency to probe frequency-domain alignment. | 1 | 0.5 | 0.3 |
| spec_0p4 | Spectral = 0.4 | spectral | Increase spectral consistency to probe frequency-domain alignment. | 1 | 0.5 | 0.4 |

Notes:
- `baseline` matches the final loss used in the main experiments.
- `time_l1`, `mse`, and `spectral` are the only active loss weights in the current objective.
