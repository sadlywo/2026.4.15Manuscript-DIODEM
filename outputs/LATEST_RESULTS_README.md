# Latest Result Index

This directory has been cleaned so the active result set focuses on the current manuscript experiments.

## Active Results

- `supervised_tcn_causal_ablations/`
  - Latest loss-function ablation runs and seed-level summaries.
  - Main summary: `supervised_tcn_causal_ablations/ablation_summary.csv`
  - Per-seed summary: `supervised_tcn_causal_ablations/ablation_summary_by_seed.csv`

- `loss_ablation/figures/`
  - Latest loss ablation figure.
  - Main figure: `loss_ablation/figures/loss_ablation_three_panel_nature.*`

- `causal_model_comparison/`
  - Latest causal deep model comparison and full comparison with classical/static baselines.
  - Causal model table: `causal_model_comparison/tables/causal_model_comparison_table.csv`
  - Full baseline table: `causal_model_comparison/tables/full_model_comparison_with_statistical_baselines_numeric.csv`
  - Main figures: `causal_model_comparison/figures/full_model_comparison_with_statistical_baselines_nature.*`
    and `causal_model_comparison/figures/complete_model_comparison_with_static_methods_nature.*`

- `supervised_tcn_causal_weight_sensitivity_local/`
  - Latest local multi-seed loss-weight sensitivity study.

- `paper_tables/`
  - Current manuscript-ready tables for loss ablation and loss-weight sensitivity.
  - `supplementary_loss_ablation_table.csv` was regenerated from the latest `ablation_summary.csv`.

- `supervised_by_experiment/`, `supervised_by_motion_type/`, `supervised_anomaly_test_only/`
  - Static/classical baseline source outputs used by the full comparison table.

- `supplementary_tables/`
  - Latest supplementary deep model comparison table artifacts.

## Archive

Older result tables, figures, and superseded experiment outputs were moved to:

- `_archive/2026-06-01_result_cleanup/`

The full move log is available at:

- `_archive/2026-06-01_result_cleanup/archive_manifest.txt`
