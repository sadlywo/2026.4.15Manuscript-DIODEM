**Supplementary Table Sy. Loss-weight sensitivity results.**

Rows report aggregated multi-seed results for the `complete` sensitivity preset. Lower RMSE and PSD distance are better, whereas higher HF improvement is better.

| Variant | Changed term | time_l1 | mse | derivative | spectral | attach_l2 | attach_temporal | RMSE | Pearson | PSD Dist. | HF Improve. | Acc Norm RMSE | Gyr Norm RMSE | Delta RMSE vs baseline | Delta PSD vs baseline | Delta HF vs baseline |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | none | 1 | 0.5 | 0.3 | 0.2 | 1e-03 | 1e-03 | 0.3870 | 0.8505 | 0.05283 | 8.309 | 0.5160 | 0.1765 | +0.00% | +0.00% | +0.00% |
| time_l1_0p5 | time_l1 | 0.5 | 0.5 | 0.3 | 0.2 | 1e-03 | 1e-03 | 0.3913 | 0.8517 | 0.05544 | 8.349 | 0.5224 | 0.1786 | +1.11% | +4.94% | +0.48% |
| time_l1_2p0 | time_l1 | 2 | 0.5 | 0.3 | 0.2 | 1e-03 | 1e-03 | 0.3856 | 0.8490 | 0.05347 | 8.306 | 0.5167 | 0.1763 | -0.36% | +1.21% | -0.04% |
| mse_0p25 | mse | 1 | 0.25 | 0.3 | 0.2 | 1e-03 | 1e-03 | 0.3856 | 0.8512 | 0.05283 | 8.350 | 0.5180 | 0.1777 | -0.36% | +0.00% | +0.49% |
| mse_1p0 | mse | 1 | 1 | 0.3 | 0.2 | 1e-03 | 1e-03 | 0.3877 | 0.8507 | 0.05429 | 8.319 | 0.5143 | 0.1753 | +0.18% | +2.76% | +0.12% |
| derivative_0p1 | derivative | 1 | 0.5 | 0.1 | 0.2 | 1e-03 | 1e-03 | 0.3904 | 0.8484 | 0.05457 | 8.305 | 0.5200 | 0.1783 | +0.88% | +3.29% | -0.05% |
| derivative_0p5 | derivative | 1 | 0.5 | 0.5 | 0.2 | 1e-03 | 1e-03 | 0.3871 | 0.8517 | 0.05434 | 8.339 | 0.5171 | 0.1777 | +0.03% | +2.86% | +0.36% |
| spectral_0p1 | spectral | 1 | 0.5 | 0.3 | 0.1 | 1e-03 | 1e-03 | 0.3903 | 0.8495 | 0.05677 | 8.292 | 0.5200 | 0.1790 | +0.85% | +7.46% | -0.20% |
| spectral_0p4 | spectral | 1 | 0.5 | 0.3 | 0.4 | 1e-03 | 1e-03 | 0.3871 | 0.8513 | 0.05217 | 8.371 | 0.5209 | 0.1769 | +0.03% | -1.25% | +0.75% |
| attach_l2_0 | attach_l2 | 1 | 0.5 | 0.3 | 0.2 | 0 | 1e-03 | 0.3880 | 0.8506 | 0.05370 | 8.341 | 0.5172 | 0.1765 | +0.26% | +1.65% | +0.39% |
| attach_l2_1e4 | attach_l2 | 1 | 0.5 | 0.3 | 0.2 | 1e-04 | 1e-03 | 0.3885 | 0.8510 | 0.05446 | 8.339 | 0.5173 | 0.1779 | +0.39% | +3.09% | +0.36% |
| attach_l2_1e2 | attach_l2 | 1 | 0.5 | 0.3 | 0.2 | 0.01 | 1e-03 | 0.3903 | 0.8502 | 0.05561 | 8.312 | 0.5196 | 0.1788 | +0.85% | +5.26% | +0.04% |
| attach_temporal_0 | attach_temporal | 1 | 0.5 | 0.3 | 0.2 | 1e-03 | 0 | 0.3904 | 0.8509 | 0.05566 | 8.308 | 0.5204 | 0.1783 | +0.88% | +5.36% | -0.01% |
| attach_temporal_1e4 | attach_temporal | 1 | 0.5 | 0.3 | 0.2 | 1e-03 | 1e-04 | 0.3884 | 0.8508 | 0.05449 | 8.320 | 0.5177 | 0.1773 | +0.36% | +3.14% | +0.13% |
| attach_temporal_1e2 | attach_temporal | 1 | 0.5 | 0.3 | 0.2 | 1e-03 | 0.01 | 0.3909 | 0.8506 | 0.05558 | 8.319 | 0.5204 | 0.1786 | +1.01% | +5.21% | +0.12% |

Notes:
- Variants should be interpreted relative to the `baseline` row, which matches the main-paper loss coefficients.
- When a larger weight improves one metric while degrading another, the final choice should be justified as an empirical trade-off rather than as a globally optimal setting.
