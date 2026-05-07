**Supplementary Table Sx. Recommended loss-weight sensitivity study design.**

This table defines the `complete` sensitivity preset used to assess whether the composite-loss coefficients produce a robust empirical trade-off. Unless otherwise stated, all settings inherit the same model backbone, data split, and preprocessing pipeline; only the listed loss coefficients are changed. Recommended seeds: 42.

| Variant | Changed term | Purpose | time_l1 | mse | derivative | spectral | attach_l2 | attach_temporal |
|---|---|---|---|---|---|---|---|---|
| baseline | none | Reference setting used in the main paper. | 1 | 0.5 | 0.3 | 0.2 | 1e-03 | 1e-03 |
| time_l1_0p5 | time_l1 | Reduce the weight of L1 reconstruction to test weaker point-wise robustness. | 0.5 | 0.5 | 0.3 | 0.2 | 1e-03 | 1e-03 |
| time_l1_2p0 | time_l1 | Increase the weight of L1 reconstruction to test stronger point-wise robustness. | 2 | 0.5 | 0.3 | 0.2 | 1e-03 | 1e-03 |
| mse_0p25 | mse | Reduce the MSE contribution to test a lighter quadratic penalty. | 1 | 0.25 | 0.3 | 0.2 | 1e-03 | 1e-03 |
| mse_1p0 | mse | Increase the MSE contribution to test a stronger quadratic penalty. | 1 | 1 | 0.3 | 0.2 | 1e-03 | 1e-03 |
| derivative_0p1 | derivative | Reduce derivative consistency to test weaker temporal dynamic preservation. | 1 | 0.5 | 0.1 | 0.2 | 1e-03 | 1e-03 |
| derivative_0p5 | derivative | Increase derivative consistency to test stronger temporal dynamic preservation. | 1 | 0.5 | 0.5 | 0.2 | 1e-03 | 1e-03 |
| spectral_0p1 | spectral | Reduce spectral consistency to test weaker frequency-domain alignment. | 1 | 0.5 | 0.3 | 0.1 | 1e-03 | 1e-03 |
| spectral_0p4 | spectral | Increase spectral consistency to test stronger frequency-domain alignment. | 1 | 0.5 | 0.3 | 0.4 | 1e-03 | 1e-03 |
| attach_l2_0 | attach_l2 | Disable latent magnitude regularization. | 1 | 0.5 | 0.3 | 0.2 | 0 | 1e-03 |
| attach_l2_1e4 | attach_l2 | Weaken latent magnitude regularization by one order of magnitude. | 1 | 0.5 | 0.3 | 0.2 | 1e-04 | 1e-03 |
| attach_l2_1e2 | attach_l2 | Strengthen latent magnitude regularization by one order of magnitude. | 1 | 0.5 | 0.3 | 0.2 | 0.01 | 1e-03 |
| attach_temporal_0 | attach_temporal | Disable latent temporal smoothness regularization. | 1 | 0.5 | 0.3 | 0.2 | 1e-03 | 0 |
| attach_temporal_1e4 | attach_temporal | Weaken latent temporal smoothness regularization by one order of magnitude. | 1 | 0.5 | 0.3 | 0.2 | 1e-03 | 1e-04 |
| attach_temporal_1e2 | attach_temporal | Strengthen latent temporal smoothness regularization by one order of magnitude. | 1 | 0.5 | 0.3 | 0.2 | 1e-03 | 0.01 |

Suggested interpretation:
- `time_l1` and `mse` probe the balance between absolute and quadratic reconstruction penalties.
- `derivative` tests the sensitivity of temporal dynamic preservation.
- `spectral` tests the sensitivity of frequency-domain consistency.
- `attach_l2` and `attach_temporal` test whether the attachment latent is under- or over-regularized.
