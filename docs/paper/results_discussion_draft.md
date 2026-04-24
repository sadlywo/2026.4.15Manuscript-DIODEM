## Results and Discussion

The paper-ready tables generated from the current experiments are saved in:

- [Table 1: by-experiment full comparison](/E:/VSCode_Study/2026.4.15Manuscript-DIODEM/outputs/paper_tables/table1_by_experiment_full_comparison.md)
- [Table 2: generalization core models](/E:/VSCode_Study/2026.4.15Manuscript-DIODEM/outputs/paper_tables/table2_generalization_core_models.md)
- [Table 3: deployment and streaming summary](/E:/VSCode_Study/2026.4.15Manuscript-DIODEM/outputs/paper_tables/table3_deployment_streaming_summary.md)
- [Extended main results table](/E:/VSCode_Study/2026.4.15Manuscript-DIODEM/outputs/paper_tables/paper_main_results_extended.md)

### 4.1 Performance Under the Standard by-Experiment Split

Table 1 summarizes the multi-seed results under the standard `by_experiment` setting. Among all learned models, the Transformer achieved the best overall reconstruction performance, with an RMSE of `0.3472 +- 0.0026`, a Pearson correlation of `0.8842 +- 0.0012`, and the lowest PSD distance (`0.0493 +- 0.0009`). The proposed deployment-oriented causal TCN (`tcn_causal`) ranked second, reaching `0.3870 +- 0.0013` RMSE, which was substantially better than the original non-causal TCN (`0.3908 +- 0.0059`), GRU (`0.4020 +- 0.0020`), and all classical filters.

Relative to the strongest classical baseline (`lowpass`), `tcn_causal` reduced RMSE from `0.5575` to `0.3870`, corresponding to an additional reduction of approximately `30.6%`. This pattern was consistent across channel-wise errors as well, with acceleration RMSE decreasing from `0.9288` to `0.6342` and gyroscope RMSE decreasing from `0.1862` to `0.1399`. These results indicate that the learned compensator does not merely smooth the signal, but rather recovers a rigid-reference-like inertial trajectory more faithfully than fixed filtering rules.

The comparison between `tcn_causal` and the original `tcn` is especially important. Although the two models share the same order of magnitude in parameter count and latency, `tcn_causal` achieved a slightly lower RMSE and better PSD alignment. Motion-level Wilcoxon tests confirmed that this improvement was not accidental: the reduction in RMSE was significant (`p = 0.0022`), and the improvements in acceleration RMSE, gyroscope RMSE, PSD distance, and high-frequency improvement were also significant. Pearson correlation did not differ significantly between the two (`p = 0.1470`), suggesting that causalization improved reconstruction fidelity without materially changing the overall correlation structure.

These results support two conclusions. First, the learned compensation paradigm clearly outperforms classical filtering under the standard test split. Second, introducing an explicitly causal formulation did not degrade performance; instead, it slightly improved the TCN baseline while preserving its lightweight structure. This is a critical result for subsequent real-time deployment.

### 4.2 Generalization to Unseen Motion Types and Anomaly Cases

Table 2 compares the main models across `by_experiment`, `by_motion_type`, and `anomaly_test_only`. The most striking finding is that the largest performance degradation occurred under `by_motion_type`, not under `anomaly_test_only`. For `tcn_causal`, RMSE increased from `0.3870` in `by_experiment` to `0.5753` in `by_motion_type`, corresponding to a relative increase of approximately `48.7%`. In contrast, the increase under `anomaly_test_only` was modest (`0.4027`, about `4.0%` above the standard split). This pattern strongly suggests that cross-motion distribution shift is the primary generalization challenge in the current framework, whereas anomaly-focused degradation is comparatively limited.

Even in the more difficult settings, the learned models remained clearly better than classical filtering. Under `by_motion_type`, `tcn_causal` reduced RMSE from `0.7865` (`lowpass`) to `0.5753`, an improvement of about `26.8%`. Under `anomaly_test_only`, the corresponding reduction was approximately `29.9%` (`0.5743` to `0.4027`). Therefore, the superiority of learned compensation over filtering is not confined to the standard split, but persists under both unseen motion categories and anomaly-focused evaluation.

The relative ranking among learned models was also stable. The Transformer remained the strongest model under all three settings, reaching `0.5227 +- 0.0079` RMSE in `by_motion_type` and `0.3622 +- 0.0027` in `anomaly_test_only`. The causal TCN was consistently the second-best learned model and outperformed the original TCN in both `by_experiment` and `anomaly_test_only`. Under `by_motion_type`, however, `tcn_causal` and `tcn` were essentially tied (`0.5753` vs `0.5738`), and motion-level tests indicated no significant difference between them. This suggests that causalization does not solve the core cross-motion generalization challenge by itself; rather, that challenge appears to be dominated by the mismatch between training and test motion dynamics.

The motion-level results further clarify where generalization is most difficult. In `by_motion_type`, the hardest motions for `tcn_causal` were `shaking` (`0.8370` RMSE) and `explosiv` (`0.8034` RMSE), followed by `gait_fast` and `gait_slow`. When directly compared with the Transformer, the largest Transformer gains were concentrated on `shaking`, `explosiv`, and `rotation`, whereas `tcn_causal` remained slightly better on `gait_fast` and `gait_slow`. This indicates that the Transformer’s main advantage lies in modeling the most difficult unseen high-dynamic motions. Under `anomaly_test_only`, the hardest case remained `shaking` (`1.0219` RMSE), indicating that anomaly-related degradation is concentrated in highly oscillatory motion patterns rather than being uniformly distributed across all activities.

From a scientific perspective, these findings reinforce the interpretation that IMU attachment artifacts are highly motion-dependent distortions. The fact that the models generalize relatively well to anomaly-focused testing but degrade more clearly under unseen motion types suggests that the current bottleneck is not simply robustness to noise or outliers. Instead, it is the ability to extrapolate to new movement dynamics that were not sufficiently represented during training.

### 4.3 Deployment and Streaming Validation

Table 3 summarizes the deployment-oriented results. The causal TCN remains highly compact, with `101,326` parameters and an FP32 model size of approximately `0.387 MB`. Its CPU forward latency was `1.301 +- 0.002 ms/window`, which was only slightly lower than the Transformer (`1.319 +- 0.016 ms/window`) in the current benchmark environment, while being roughly four times smaller in parameter count (`399,110` for the Transformer). The GRU was both larger than the TCN and noticeably slower (`2.699 +- 0.051 ms/window`), leaving little reason to prefer it under the present operating conditions.

The most important deployment result, however, comes from the dedicated streaming validation. For the `by_experiment` causal TCN checkpoint, the streaming implementation produced outputs that were numerically almost identical to the offline causal forward pass, with a mean streaming-versus-offline RMSE of approximately `3.4e-7` and a mean maximum absolute difference of about `1.8e-6`. The mean step latency was `1.191 ms`, with a `p95` latency of `1.209 ms`. At a sampling frequency of `40 Hz`, where each sample period is `25 ms`, this latency occupies less than `5%` of the available real-time budget.

This result is significant because it closes the gap between “a model that is theoretically causal” and “a model that is practically usable in streaming mode.” The present evidence shows that the proposed `tcn_causal` is not merely a lightweight offline model; it is already a valid online compensation candidate with numerically faithful step-wise inference and low per-step latency. This strongly supports the next research phase, namely edge-device deployment and real-time bench-top validation.

### 4.4 Overall Interpretation

Taken together, the present results support a coherent interpretation of the framework. First, learning-based compensation consistently outperforms classical filters across all tested settings, confirming that attachment artifacts cannot be adequately modeled as generic high-frequency noise alone. Second, the best offline accuracy is obtained by the Transformer, especially for the most difficult high-dynamic and previously unseen motions. Third, the causal TCN offers a highly attractive trade-off: it is substantially stronger than classical filtering, slightly better than the original TCN in the standard and anomaly-focused settings, and already supported by explicit streaming validation.

Accordingly, the current evidence supports a dual-model conclusion for the paper. The Transformer should be positioned as the highest-performing offline benchmark, while the causal TCN should be positioned as the most suitable model for real-time and embedded deployment. This distinction is not a weakness; rather, it gives the work a clearer scientific and engineering structure by separating the role of an accuracy upper bound from that of a deployment-oriented compensator.

### 4.5 Recommended Next Step

Based on the present results, the most logical next step is no longer to enlarge the model family or to continue optimizing small offline metrics. Instead, the next stage should focus on verifying the causal TCN in an actual streaming system. Specifically, the priority should be:

1. real device-side streaming validation on Raspberry Pi or Jetson rather than host-side CPU profiling only;
2. short-window causal experiments such as `16`, `32`, and `64` samples to characterize the latency–accuracy trade-off;
3. bench-top real-time experiments to demonstrate that the online compensator provides measurable benefit under controlled soft-coupling conditions;
4. targeted improvement of the hardest motions, especially `shaking`, `explosiv`, and `rotation`, where the Transformer still shows a clear advantage.

In short, the current study has already moved beyond a purely offline artifact-compensation model. It now provides a lightweight causal compensator with competitive reconstruction quality, clear robustness trends, and initial but strong evidence for online embedded deployment.
