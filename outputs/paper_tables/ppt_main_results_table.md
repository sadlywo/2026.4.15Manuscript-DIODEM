| Setting | Best Offline Model (RMSE) | Real-Time Candidate (RMSE) | Original TCN (RMSE) | Lowpass Baseline (RMSE) | TCN-causal vs Lowpass | TCN-causal vs Raw Input | Key Takeaway |
| --- | --- | --- | --- | --- | --- | --- | --- |
| By-experiment | Transformer `0.3472 +- 0.0026` | TCN-causal `0.3870 +- 0.0013` | TCN `0.3908 +- 0.0059` | `0.5575 +- 0.0000` | `30.6%` lower RMSE | `58.9%` lower RMSE | Best overall in-distribution performance; causalization does not hurt accuracy |
| By-motion-type | Transformer `0.5227 +- 0.0079` | TCN-causal `0.5753 +- 0.0046` | TCN `0.5738 +- 0.0027` | `0.7865 +- 0.0000` | `26.8%` lower RMSE | `57.0%` lower RMSE | Unseen motion types remain the main generalization bottleneck |
| Anomaly test-only | Transformer `0.3622 +- 0.0027` | TCN-causal `0.4027 +- 0.0012` | TCN `0.4064 +- 0.0058` | `0.5743 +- 0.0000` | `29.9%` lower RMSE | `58.1%` lower RMSE | Anomalies are easier than motion-shift; causal model stays robust |
