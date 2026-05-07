**Supplementary Table Sx. Loss-function ablation study on the test split.**

Each row corresponds to one ablation setting derived from the same attachment-aware TCN backbone. The table reports whether each loss component or latent mechanism is enabled, together with the resulting test performance on 8,633 windows. Lower RMSE and PSD distance are better, whereas higher HF improvement is better.

| Variant | Latent | L1 | MSE | Deriv. | Spectral | Att-L2 | Att-Temp | RMSE | Delta RMSE vs Full | Pearson | PSD Dist. | Delta PSD vs Full | HF Improve. | Delta HF vs Full | Acc Norm RMSE | Gyr Norm RMSE | Test windows |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Full model | Y | Y | Y | Y | Y | Y | Y | 0.1155 | +0.00% | 0.8510 | 0.00655 | +0.00% | 5.754 | +0.00% | 0.1182 | 0.1217 | 8633 |
| w/o derivative loss | Y | Y | Y | N | Y | Y | Y | 0.1168 | +1.11% | 0.8490 | 0.00668 | +1.96% | 5.682 | -1.26% | 0.1190 | 0.1227 | 8633 |
| w/o spectral loss | Y | Y | Y | Y | N | Y | Y | 0.1149 | -0.45% | 0.8482 | 0.00680 | +3.74% | 5.604 | -2.61% | 0.1170 | 0.1205 | 8633 |
| w/o attachment regularization | Y | Y | Y | Y | Y | N | N | 0.1152 | -0.22% | 0.8516 | 0.00641 | -2.24% | 5.704 | -0.87% | 0.1184 | 0.1211 | 8633 |
| w/o attachment latent | N | Y | Y | Y | Y | N | N | 0.1169 | +1.25% | 0.8513 | 0.00696 | +6.14% | 5.687 | -1.17% | 0.1190 | 0.1249 | 8633 |
| MSE only | Y | N | Y | N | N | N | N | 0.1247 | +8.03% | 0.8363 | 0.00791 | +20.67% | 5.456 | -5.19% | 0.1213 | 0.1300 | 8633 |

Notes:
- `Latent` indicates whether the attachment latent branch is present.
- `Deriv.` denotes the temporal derivative consistency term.
- `Spectral` denotes the frequency-domain consistency term.
- `Att-L2` and `Att-Temp` denote the latent magnitude and latent temporal smoothness regularizers, respectively.
- Although removing the spectral term slightly reduces point-wise RMSE, it worsens frequency-domain alignment and high-frequency consistency, which supports retaining the spectral component in the final objective.
- Using only MSE leads to the clearest overall degradation, indicating that point-wise reconstruction alone is insufficient for stable artifact compensation.
