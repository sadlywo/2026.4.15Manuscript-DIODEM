**Supplementary Table Sx. Loss-function ablation study on the test split.**

Each row corresponds to one ablation setting derived from the same attachment-aware TCN backbone. The table reports whether each loss component or latent mechanism is enabled, together with the resulting test performance. Metrics are reported as mean +/- standard deviation across seeds when multi-seed runs are available. Lower RMSE and PSD distance are better, whereas higher HF improvement is better.

| Variant | Latent | L1 | MSE | Deriv. | Spectral | Att-L2 | Att-Temp | RMSE | Delta RMSE vs Full | Pearson | PSD Dist. | Delta PSD vs Full | HF Improve. | Delta HF vs Full | Acc Norm RMSE | Gyr Norm RMSE | Test windows | Seeds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Full model | Y | Y | Y | Y | Y | Y | Y | 0.3905 +/- 0.0033 | +0.00% | 0.8511 +/- 0.0003 | 0.05560 +/- 0.00198 | +0.00% | 8.291 +/- 0.047 | +0.00% | 0.5195 +/- 0.0019 | 0.1783 +/- 0.0025 | 8633 | 3 |
| w/o derivative loss | Y | Y | Y | N | Y | Y | Y | 0.3890 +/- 0.0018 | -0.38% | 0.8489 +/- 0.0007 | 0.05314 +/- 0.00075 | -4.43% | 8.311 +/- 0.022 | +0.24% | 0.5198 +/- 0.0012 | 0.1779 +/- 0.0012 | 8633 | 3 |
| w/o spectral loss | Y | Y | Y | Y | N | Y | Y | 0.3916 +/- 0.0028 | +0.30% | 0.8477 +/- 0.0008 | 0.05963 +/- 0.00073 | +7.24% | 8.254 +/- 0.029 | -0.45% | 0.5201 +/- 0.0023 | 0.1787 +/- 0.0015 | 8633 | 3 |
| w/o attachment regularization | Y | Y | Y | Y | Y | N | N | 0.3879 +/- 0.0019 | -0.66% | 0.8501 +/- 0.0012 | 0.05338 +/- 0.00088 | -4.00% | 8.310 +/- 0.037 | +0.23% | 0.5182 +/- 0.0031 | 0.1761 +/- 0.0015 | 8633 | 3 |
| w/o attachment latent | N | Y | Y | Y | Y | N | N | 0.3914 +/- 0.0014 | +0.24% | 0.8507 +/- 0.0009 | 0.05712 +/- 0.00035 | +2.73% | 8.299 +/- 0.021 | +0.10% | 0.5277 +/- 0.0020 | 0.1805 +/- 0.0002 | 8633 | 3 |
| MSE only | Y | N | Y | N | N | N | N | 0.4082 +/- 0.0043 | +4.54% | 0.8384 +/- 0.0004 | 0.06499 +/- 0.00093 | +16.89% | 8.116 +/- 0.018 | -2.11% | 0.5359 +/- 0.0057 | 0.1836 +/- 0.0015 | 8633 | 3 |

Notes:
- `Latent` indicates whether the attachment latent branch is present.
- `Deriv.` denotes the temporal derivative consistency term.
- `Spectral` denotes the frequency-domain consistency term.
- `Att-L2` and `Att-Temp` denote the latent magnitude and latent temporal smoothness regularizers, respectively.
- Although removing the spectral term slightly reduces point-wise RMSE, it worsens frequency-domain alignment and high-frequency consistency, which supports retaining the spectral component in the final objective.
- Using only MSE leads to the clearest overall degradation, indicating that point-wise reconstruction alone is insufficient for stable artifact compensation.
