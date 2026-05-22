# Patent Figure and Table Insertion Notes

## Figure Recommendation

- Suggested title: Comparison of soft-attached IMU signal, rigid-reference IMU signal, and compensated output signal
- File: [patent_signal_comparison_figure.png](E:/VSCode_Study/2026.4.15Manuscript-DIODEM/docs/patent/assets/patent_signal_comparison_figure.png)
- Suggested description:
  This figure illustrates the relationship among the soft-attached IMU signal, the rigid-reference IMU signal, and the compensated output produced by the proposed method. The compensated output is visibly closer to the rigid-reference signal in both acceleration and gyroscope domains, indicating that the method effectively reduces measurement deviations caused by soft attachment, local slipping, and compliant coupling.

## Table 1 Recommendation

- Suggested title: Compensation performance comparison across methods
- File: [patent_method_comparison_table.png](E:/VSCode_Study/2026.4.15Manuscript-DIODEM/docs/patent/assets/patent_method_comparison_table.png)
- Suggested description:
  Table 1 compares the proposed method with representative filtering methods and learning-based models. Relative to conventional filtering, the proposed method achieves better overall performance in error metrics, correlation, and high-frequency consistency while retaining causal and online deployment capability.

## Table 2 Recommendation

- Suggested title: Effect of loss-function components on compensation performance
- File: [patent_loss_ablation_table.png](E:/VSCode_Study/2026.4.15Manuscript-DIODEM/docs/patent/assets/patent_loss_ablation_table.png)
- Suggested description:
  Table 2 shows the contribution of different loss-function components. The results indicate that relying only on a simple reconstruction term is insufficient, whereas introducing derivative consistency, spectral consistency, and attachment-state-related constraints leads to a more balanced performance in time-domain, frequency-domain, and dynamic-consistency metrics.
