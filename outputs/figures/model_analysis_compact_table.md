| Model       | Role in manuscript                 | RMSE             | PSD distance     | Parameters   | CPU ms/window   |
|:------------|:-----------------------------------|:-----------------|:-----------------|:-------------|:----------------|
| Transformer | Offline accuracy reference         | 0.3472 +- 0.0026 | 0.0493 +- 0.0009 | 399,110      | 1.319 +- 0.016  |
| TCN-causal  | Deployment-oriented proposed model | 0.3870 +- 0.0013 | 0.0528 +- 0.0008 | 101,326      | 1.301 +- 0.002  |
| TCN         | Non-causal TCN comparison          | 0.3908 +- 0.0059 | 0.0545 +- 0.0004 | 101,326      | 1.336 +- 0.012  |
| GRU         | Recurrent learning baseline        | 0.4020 +- 0.0020 | 0.0569 +- 0.0009 | 152,070      | 2.699 +- 0.051  |
| Low-pass    | Strongest classical filter         | 0.5575 +- 0.0000 | 0.1323 +- 0.0000 | 0            | 0.030 +- 0.000  |
