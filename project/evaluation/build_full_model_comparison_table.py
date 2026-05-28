from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


SETTING_ORDER = ["by_experiment", "by_motion_type", "anomaly_test_only"]
SETTING_LABELS = {
    "by_experiment": "By experiment",
    "by_motion_type": "By motion type",
    "anomaly_test_only": "Anomaly test-only",
}
CAUSAL_MODEL_ORDER = ["TCN-causal", "Transformer-causal", "GRU-causal", "LSTM-causal", "MLP-causal"]
STATISTICAL_BASELINE_ORDER = ["identity", "lowpass", "butterworth", "savgol", "wiener"]
BASELINE_LABELS = {
    "identity": "Identity/raw",
    "lowpass": "Moving-average low-pass",
    "butterworth": "Butterworth low-pass",
    "savgol": "Savitzky-Golay",
    "wiener": "Wiener filter",
}
BASELINE_TYPES = {
    "identity": "raw signal baseline",
    "lowpass": "statistical/filter baseline",
    "butterworth": "statistical/filter baseline",
    "savgol": "statistical/filter baseline",
    "wiener": "statistical/filter baseline",
}
MODEL_FAMILY = {
    "TCN-causal": "proposed causal neural model",
    "Transformer-causal": "causal neural comparator",
    "GRU-causal": "causal neural comparator",
    "LSTM-causal": "causal neural comparator",
    "MLP-causal": "causal neural comparator",
}


def _parse_mean_std(value: Any) -> tuple[float, float]:
    matches = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(value))
    if not matches:
        return np.nan, np.nan
    mean = float(matches[0])
    std = float(matches[1]) if len(matches) > 1 else 0.0
    return mean, std


def _format_mean_std(mean: float, std: float, digits: int = 4, latex: bool = False) -> str:
    if pd.isna(mean):
        return "NA"
    pm = r" $\pm$ " if latex else " +/- "
    return f"{mean:.{digits}f}{pm}{0.0 if pd.isna(std) else std:.{digits}f}"


def _format_p_value(value: float | str) -> str:
    if isinstance(value, str):
        return value
    if pd.isna(value):
        return "NA"
    if value < 1e-4:
        return "<1e-4"
    return f"{value:.4f}"


def _load_causal_rows(causal_table: Path) -> pd.DataFrame:
    table = pd.read_csv(causal_table)
    records: List[Dict[str, Any]] = []
    for _, row in table.iterrows():
        rmse_mean, rmse_std = _parse_mean_std(row["RMSE"])
        pearson_mean, pearson_std = _parse_mean_std(row["Pearson"])
        psd_mean, psd_std = _parse_mean_std(row["PSD Dist."])
        hf_mean, hf_std = _parse_mean_std(row["HF Improve."])
        cpu_mean, cpu_std = _parse_mean_std(row["CPU forward ms/window"])
        stream_mean, stream_std = _parse_mean_std(row["Streaming ms/step"])
        method = str(row["Model"])
        records.append(
            {
                "setting": str(row["Setting"]),
                "method": method,
                "method_family": MODEL_FAMILY.get(method, "causal neural model"),
                "model_type": str(row["Causal type"]),
                "causal_online": "yes",
                "rmse_mean": rmse_mean,
                "rmse_std": rmse_std,
                "pearson_mean": pearson_mean,
                "pearson_std": pearson_std,
                "psd_distance_mean": psd_mean,
                "psd_distance_std": psd_std,
                "hf_improvement_mean": hf_mean,
                "hf_improvement_std": hf_std,
                "params_k": float(row["Parameters"]) / 1000.0,
                "fp32_size_mb": float(row["FP32 size (MB)"]),
                "cpu_window_ms_mean": cpu_mean,
                "cpu_window_ms_std": cpu_std,
                "stream_ms_step_mean": stream_mean,
                "stream_ms_step_std": stream_std,
                "embedded_feasibility": str(row["Deployment verdict"]),
            }
        )
    return pd.DataFrame(records)


def _supervised_paths(outputs_root: Path, setting: str) -> tuple[Path, Path]:
    run_dir = {
        "by_experiment": "supervised_by_experiment",
        "by_motion_type": "supervised_by_motion_type",
        "anomaly_test_only": "supervised_anomaly_test_only",
    }[setting]
    metrics_root = outputs_root / run_dir / "evaluation" / "metrics"
    return metrics_root / "multiseed_model_comparison.csv", metrics_root / "multiseed_model_deployment_summary.csv"


def _load_statistical_baseline_rows(outputs_root: Path) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for setting in SETTING_ORDER:
        comparison_path, deployment_path = _supervised_paths(outputs_root, setting)
        comparison = pd.read_csv(comparison_path)
        deployment = pd.read_csv(deployment_path) if deployment_path.exists() else pd.DataFrame()
        for baseline_name in STATISTICAL_BASELINE_ORDER:
            matched = comparison[comparison["model_name"].astype(str).str.lower() == baseline_name]
            if matched.empty:
                continue
            row = matched.iloc[0]
            deploy = deployment[deployment["model_name"].astype(str).str.lower() == baseline_name]
            deploy_row = deploy.iloc[0] if not deploy.empty else {}
            records.append(
                {
                    "setting": setting,
                    "method": BASELINE_LABELS.get(baseline_name, baseline_name),
                    "method_family": "classical/statistical baseline",
                    "model_type": BASELINE_TYPES.get(baseline_name, "statistical/filter baseline"),
                    "causal_online": "window/filter",
                    "rmse_mean": float(row["rmse_mean_mean"]),
                    "rmse_std": float(row["rmse_mean_std"]),
                    "pearson_mean": float(row["pearson_mean_mean"]),
                    "pearson_std": float(row["pearson_mean_std"]),
                    "psd_distance_mean": float(row["psd_distance_mean_mean"]),
                    "psd_distance_std": float(row["psd_distance_mean_std"]),
                    "hf_improvement_mean": float(row["hf_ratio_improvement_mean_mean"]),
                    "hf_improvement_std": float(row["hf_ratio_improvement_mean_std"]),
                    "params_k": float(deploy_row.get("parameter_count_mean", 0.0)) / 1000.0 if len(deploy) else 0.0,
                    "fp32_size_mb": float(deploy_row.get("parameter_size_mb_fp32_mean", 0.0)) if len(deploy) else 0.0,
                    "cpu_window_ms_mean": float(deploy_row.get("cpu_forward_ms_per_window_mean", np.nan)) if len(deploy) else np.nan,
                    "cpu_window_ms_std": float(deploy_row.get("cpu_forward_ms_per_window_std", 0.0)) if len(deploy) else 0.0,
                    "stream_ms_step_mean": np.nan,
                    "stream_ms_step_std": np.nan,
                    "embedded_feasibility": str(deploy_row.get("embedded_deployment_verdict", "yes_classical_filter")) if len(deploy) else "yes_classical_filter",
                }
            )
    return pd.DataFrame(records)


def _average_per_motion(files: Iterable[Path]) -> pd.DataFrame:
    frames = []
    for file_path in files:
        if file_path.exists():
            frame = pd.read_csv(file_path)
            frame["seed_id"] = file_path.parts[-4] if "seed_runs" in file_path.parts else "aggregate"
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    all_frames = pd.concat(frames, ignore_index=True)
    numeric_columns = [column for column in all_frames.columns if column not in {"motion_name", "seed_id"}]
    return all_frames.groupby("motion_name", as_index=False)[numeric_columns].mean(numeric_only=True)


def _causal_per_motion(outputs_root: Path, setting: str, method: str) -> pd.DataFrame:
    method_key = method.lower().replace("-", "_")
    files = sorted(
        (
            outputs_root
            / "causal_model_comparison"
            / f"{method_key}_{setting}"
            / "seed_runs"
        ).glob(f"seed_*/evaluation/metrics/{method_key}/per_motion_metrics.csv")
    )
    return _average_per_motion(files)


def _baseline_per_motion(outputs_root: Path, setting: str, method_label: str) -> pd.DataFrame:
    reverse = {label: key for key, label in BASELINE_LABELS.items()}
    baseline_key = reverse.get(method_label, method_label).lower()
    run_dir = {
        "by_experiment": "supervised_by_experiment",
        "by_motion_type": "supervised_by_motion_type",
        "anomaly_test_only": "supervised_anomaly_test_only",
    }[setting]
    files = sorted(
        (outputs_root / run_dir / "seed_runs").glob(
            f"seed_*/evaluation/metrics/{baseline_key}/per_motion_metrics.csv"
        )
    )
    return _average_per_motion(files)


def _wilcoxon_vs_reference(
    outputs_root: Path,
    setting: str,
    method: str,
    family: str,
    reference_cache: Dict[str, pd.DataFrame],
) -> tuple[float | str, int, float]:
    if method == "TCN-causal":
        return "reference", 0, 0.0
    reference = reference_cache.setdefault(setting, _causal_per_motion(outputs_root, setting, "TCN-causal"))
    if reference.empty:
        return np.nan, 0, np.nan
    candidate = (
        _baseline_per_motion(outputs_root, setting, method)
        if family == "classical/statistical baseline"
        else _causal_per_motion(outputs_root, setting, method)
    )
    if candidate.empty:
        return np.nan, 0, np.nan
    merged = reference[["motion_name", "rmse_mean"]].merge(
        candidate[["motion_name", "rmse_mean"]],
        on="motion_name",
        how="inner",
        suffixes=("_tcn_causal", "_candidate"),
    )
    if merged.empty:
        return np.nan, 0, np.nan
    deltas = merged["rmse_mean_candidate"] - merged["rmse_mean_tcn_causal"]
    mean_delta = float(deltas.mean())
    nonzero = deltas[~np.isclose(deltas, 0.0)]
    if len(nonzero) == 0:
        return 1.0, int(len(deltas)), mean_delta
    try:
        _, p_value = wilcoxon(nonzero)
    except ValueError:
        p_value = np.nan
    return float(p_value), int(len(deltas)), mean_delta


def _method_sort_key(row: pd.Series) -> tuple[int, int]:
    family = str(row["method_family"])
    method = str(row["method"])
    if method == "TCN-causal":
        return 0, 0
    if method in CAUSAL_MODEL_ORDER:
        return 1, CAUSAL_MODEL_ORDER.index(method)
    if family == "classical/statistical baseline":
        reverse = {label: key for key, label in BASELINE_LABELS.items()}
        key = reverse.get(method, method)
        return 2, STATISTICAL_BASELINE_ORDER.index(key) if key in STATISTICAL_BASELINE_ORDER else 99
    return 9, 99


def build_full_comparison(outputs_root: Path) -> pd.DataFrame:
    causal = _load_causal_rows(outputs_root / "causal_model_comparison" / "tables" / "causal_model_comparison_table.csv")
    baselines = _load_statistical_baseline_rows(outputs_root)
    table = pd.concat([causal, baselines], ignore_index=True)
    reference_cache: Dict[str, pd.DataFrame] = {}
    reference_rmse = {
        setting: float(
            table[(table["setting"] == setting) & (table["method"] == "TCN-causal")]["rmse_mean"].iloc[0]
        )
        for setting in SETTING_ORDER
    }
    p_values = []
    n_pairs = []
    delta_values = []
    for _, row in table.iterrows():
        p_value, n_pair, mean_delta = _wilcoxon_vs_reference(
            outputs_root=outputs_root,
            setting=str(row["setting"]),
            method=str(row["method"]),
            family=str(row["method_family"]),
            reference_cache=reference_cache,
        )
        p_values.append(p_value)
        n_pairs.append(n_pair)
        delta_values.append(mean_delta)
    table["rmse_wilcoxon_p_vs_tcn_causal"] = p_values
    table["paired_motion_groups"] = n_pairs
    table["rmse_delta_vs_tcn_causal"] = delta_values
    table["rmse_gap_vs_tcn_causal_percent"] = [
        100.0 * (row.rmse_mean - reference_rmse[row.setting]) / reference_rmse[row.setting]
        for row in table.itertuples(index=False)
    ]
    table["setting_order"] = table["setting"].map({name: index for index, name in enumerate(SETTING_ORDER)})
    table["method_sort_0"] = table.apply(lambda row: _method_sort_key(row)[0], axis=1)
    table["method_sort_1"] = table.apply(lambda row: _method_sort_key(row)[1], axis=1)
    table = table.sort_values(["setting_order", "method_sort_0", "method_sort_1"]).drop(
        columns=["setting_order", "method_sort_0", "method_sort_1"]
    )
    return table.reset_index(drop=True)


def _presentation_frame(table: pd.DataFrame, latex: bool = False) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, row in table.iterrows():
        rows.append(
            {
                "Setting": SETTING_LABELS.get(str(row["setting"]), str(row["setting"])),
                "Family": row["method_family"],
                "Method": row["method"],
                "Causal/online": row["causal_online"],
                "RMSE ↓": _format_mean_std(row["rmse_mean"], row["rmse_std"], latex=latex),
                "ΔRMSE vs TCN (%)": f'{row["rmse_gap_vs_tcn_causal_percent"]:.1f}',
                "Wilcoxon p vs TCN": _format_p_value(row["rmse_wilcoxon_p_vs_tcn_causal"]),
                "Motion pairs": int(row["paired_motion_groups"]),
                "Pearson r ↑": _format_mean_std(row["pearson_mean"], row["pearson_std"], latex=latex),
                "PSD dist. ↓": _format_mean_std(row["psd_distance_mean"], row["psd_distance_std"], latex=latex),
                "HF imp. ↑": _format_mean_std(row["hf_improvement_mean"], row["hf_improvement_std"], latex=latex),
                "Params (K)": f'{row["params_k"]:.1f}',
                "CPU window (ms) ↓": _format_mean_std(row["cpu_window_ms_mean"], row["cpu_window_ms_std"], digits=3, latex=latex),
                "Stream (ms/step) ↓": "NA"
                if pd.isna(row["stream_ms_step_mean"])
                else _format_mean_std(row["stream_ms_step_mean"], row["stream_ms_step_std"], digits=3, latex=latex),
                "Embedded feasibility": row["embedded_feasibility"],
            }
        )
    return pd.DataFrame(rows)


def _write_latex(frame: pd.DataFrame, path: Path) -> None:
    safe_frame = frame.copy()
    for column in ["Setting", "Family", "Method", "Causal/online", "Embedded feasibility"]:
        safe_frame[column] = safe_frame[column].astype(str).str.replace("_", r"\_", regex=False)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Complete comparison of causal neural models and classical statistical/filter baselines. Values are reported as mean $\pm$ standard deviation over three random seeds. $\Delta$RMSE denotes the percent RMSE gap relative to TCN-causal within the same evaluation setting. Wilcoxon $p$ values are paired tests over motion-level RMSE against TCN-causal.}",
        r"\label{tab:full_model_comparison_statistical_baselines}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lllcccccccccccl}",
        r"\toprule",
        " & ".join(safe_frame.columns) + r" \\",
        r"\midrule",
    ]
    last_setting = None
    for record in safe_frame.to_dict(orient="records"):
        if last_setting is not None and record["Setting"] != last_setting:
            lines.append(r"\midrule")
        last_setting = record["Setting"]
        lines.append(" & ".join(str(record[column]) for column in safe_frame.columns) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(table: pd.DataFrame, output_dir: Path) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    numeric_path = output_dir / "full_model_comparison_with_statistical_baselines_numeric.csv"
    csv_path = output_dir / "full_model_comparison_with_statistical_baselines.csv"
    md_path = output_dir / "full_model_comparison_with_statistical_baselines.md"
    tex_path = output_dir / "full_model_comparison_with_statistical_baselines.tex"

    table.to_csv(numeric_path, index=False, encoding="utf-8-sig")
    presentation = _presentation_frame(table, latex=False)
    presentation.to_csv(csv_path, index=False, encoding="utf-8-sig")
    caption = (
        "Complete comparison of causal neural models and classical statistical/filter baselines. "
        "Values are mean +/- std over three random seeds. Delta RMSE is relative to TCN-causal; "
        "Wilcoxon p values are paired over motion-level RMSE."
    )
    md_path.write_text(caption + "\n\n" + presentation.to_markdown(index=False), encoding="utf-8")
    _write_latex(_presentation_frame(table, latex=True), tex_path)
    return {"numeric": numeric_path, "csv": csv_path, "markdown": md_path, "latex": tex_path}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build complete model comparison table with statistical baselines.")
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/causal_model_comparison/tables"))
    args = parser.parse_args()
    table = build_full_comparison(args.outputs_root)
    paths = write_outputs(table, args.output_dir)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
