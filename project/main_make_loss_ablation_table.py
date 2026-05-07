from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


VARIANT_LABELS = {
    "full_model": "Full model",
    "no_derivative_loss": "w/o derivative loss",
    "no_spectral_loss": "w/o spectral loss",
    "no_attachment_regularization": "w/o attachment regularization",
    "no_attachment_latent": "w/o attachment latent",
    "mse_only": "MSE only",
}

VARIANT_ORDER = [
    "full_model",
    "no_derivative_loss",
    "no_spectral_loss",
    "no_attachment_regularization",
    "no_attachment_latent",
    "mse_only",
]


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _fmt(value: float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def _fmt_delta_pct(current: float, baseline: float) -> str:
    delta = (float(current) - float(baseline)) / float(baseline) * 100.0
    return f"{delta:+.2f}%"


def _onoff(value: float) -> str:
    return "Y" if float(value) > 0 else "N"


def _build_output_rows(summary_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    rows_by_name = {row["variant_name"]: row for row in summary_rows}
    full = rows_by_name["full_model"]
    full_rmse = float(full["rmse_mean"])
    full_psd = float(full["psd_distance_mean"])
    full_hf = float(full["hf_ratio_improvement_mean"])
    output_rows: List[Dict[str, str]] = []
    for variant_name in VARIANT_ORDER:
        row = rows_by_name[variant_name]
        rmse = float(row["rmse_mean"])
        psd = float(row["psd_distance_mean"])
        hf = float(row["hf_ratio_improvement_mean"])
        output_rows.append(
            {
                "Variant": VARIANT_LABELS[variant_name],
                "Latent": "Y" if int(row["attach_latent_dim"]) > 0 else "N",
                "L1": _onoff(float(row["time_l1"])),
                "MSE": _onoff(float(row["mse"])),
                "Deriv.": _onoff(float(row["derivative"])),
                "Spectral": _onoff(float(row["spectral"])),
                "Att-L2": _onoff(float(row["attach_l2"])),
                "Att-Temp": _onoff(float(row["attach_temporal"])),
                "RMSE": _fmt(rmse, 4),
                "Delta RMSE vs Full": _fmt_delta_pct(rmse, full_rmse),
                "Pearson": _fmt(float(row["pearson_mean"]), 4),
                "PSD Dist.": _fmt(psd, 5),
                "Delta PSD vs Full": _fmt_delta_pct(psd, full_psd),
                "HF Improve.": _fmt(hf, 3),
                "Delta HF vs Full": _fmt_delta_pct(hf, full_hf),
                "Acc Norm RMSE": _fmt(float(row["acc_norm_rmse"]), 4),
                "Gyr Norm RMSE": _fmt(float(row["gyr_norm_rmse"]), 4),
                "Test windows": str(int(float(row["num_windows"]))),
            }
        )
    return output_rows


def _write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = list(rows[0].keys())
    lines = []
    lines.append("**Supplementary Table Sx. Loss-function ablation study on the test split.**")
    lines.append("")
    lines.append(
        "Each row corresponds to one ablation setting derived from the same attachment-aware TCN backbone. "
        "The table reports whether each loss component or latent mechanism is enabled, together with the resulting "
        "test performance on 8,633 windows. Lower RMSE and PSD distance are better, whereas higher HF improvement is better."
    )
    lines.append("")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row[h] for h in headers) + " |")
    lines.append("")
    lines.append("Notes:")
    lines.append("- `Latent` indicates whether the attachment latent branch is present.")
    lines.append("- `Deriv.` denotes the temporal derivative consistency term.")
    lines.append("- `Spectral` denotes the frequency-domain consistency term.")
    lines.append("- `Att-L2` and `Att-Temp` denote the latent magnitude and latent temporal smoothness regularizers, respectively.")
    lines.append(
        "- Although removing the spectral term slightly reduces point-wise RMSE, it worsens frequency-domain alignment and high-frequency consistency, "
        "which supports retaining the spectral component in the final objective."
    )
    lines.append(
        "- Using only MSE leads to the clearest overall degradation, indicating that point-wise reconstruction alone is insufficient for stable artifact compensation."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a supplementary loss ablation table from ablation_summary.csv.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("outputs/supervised_ablations/ablation_summary.csv"),
        help="Input ablation summary CSV.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/paper_tables/supplementary_loss_ablation_table.csv"),
        help="Output CSV table for supplementary materials.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("outputs/paper_tables/supplementary_loss_ablation_table.md"),
        help="Output Markdown table for supplementary materials.",
    )
    args = parser.parse_args()

    rows = _read_rows(args.input_csv.resolve())
    output_rows = _build_output_rows(rows)
    _write_csv(args.output_csv.resolve(), output_rows)
    _write_markdown(args.output_md.resolve(), output_rows)

    print(f"Wrote supplementary ablation CSV to {args.output_csv.resolve()}")
    print(f"Wrote supplementary ablation Markdown to {args.output_md.resolve()}")


if __name__ == "__main__":
    main()
