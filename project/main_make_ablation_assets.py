from __future__ import annotations

import base64
import csv
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parent.parent
ABLATION_ROOT = ROOT / "outputs" / "supervised_ablations"
TABLE_PATH = ABLATION_ROOT / "ablation_summary.csv"
OUTPUT_DIR = ROOT / "outputs" / "paper_figures"

SAMPLE_STEM = "kc_arm_exp01_motion01_seg1_00064_residual.png"
PANEL_VARIANTS = [
    ("full_model", "(A) Full model", "L1 + MSE + spectral objective"),
    ("mse_only", "(B) MSE only", "Point-wise reconstruction only"),
    ("no_spectral_loss", "(C) w/o spectral loss", "Remove frequency-domain term"),
    ("no_attachment_latent", "(D) w/o attachment latent", "Disable attachment latent branch"),
]


def _load_ablation_rows(path: Path) -> Dict[str, Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row["variant_name"]: row for row in rows}


def _encode_png(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _short_metric_text(row: Dict[str, str]) -> str:
    return (
        f"RMSE={float(row['rmse_mean']):.4f} | "
        f"PSD={float(row['psd_distance_mean']):.4f} | "
        f"HF={float(row['hf_ratio_improvement_mean']):.3f}"
    )


def build_ablation_svg() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = _load_ablation_rows(TABLE_PATH)

    fig_width = 1420
    fig_height = 1180
    margin_x = 48
    top_band = 110
    panel_gap_x = 30
    panel_gap_y = 34
    panel_width = 645
    panel_height = 450

    parts: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{fig_width}" height="{fig_height}" viewBox="0 0 {fig_width} {fig_height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>',
        '.title { font: 700 30px Arial, Helvetica, sans-serif; fill: #111111; }',
        '.subtitle { font: 400 14px Arial, Helvetica, sans-serif; fill: #4b5563; }',
        '.panel-title { font: 700 18px Arial, Helvetica, sans-serif; fill: #111111; }',
        '.panel-subtitle { font: 400 13px Arial, Helvetica, sans-serif; fill: #4b5563; }',
        '.metric { font: 600 13px Arial, Helvetica, sans-serif; fill: #1f2937; }',
        '.caption { font: 400 13px Arial, Helvetica, sans-serif; fill: #374151; }',
        '.legend { font: 400 13px Arial, Helvetica, sans-serif; fill: #374151; }',
        '.box { fill: #ffffff; stroke: #d1d5db; stroke-width: 1.2; rx: 14; ry: 14; }',
        '.metric-box { fill: #f8fafc; stroke: #dbe3ea; stroke-width: 1; rx: 10; ry: 10; }',
        '.legend-line-blue { stroke: #1f77b4; stroke-width: 3.2; }',
        '.legend-line-orange { stroke: #ff7f0e; stroke-width: 3.2; }',
        '</style>',
        '<text x="48" y="48" class="title">Ablation Study of the Final Loss and Attachment Design</text>',
        '<text x="48" y="74" class="subtitle">Same evaluation sample across variants. Blue: nonrigid-rigid residual, Orange: prediction-rigid residual.</text>',
        '<line x1="48" y1="95" x2="1372" y2="95" stroke="#e5e7eb" stroke-width="1.2"/>',
        '<line x1="1010" y1="52" x2="1044" y2="52" class="legend-line-blue"/>',
        '<text x="1052" y="57" class="legend">nonrigid-rigid</text>',
        '<line x1="1170" y1="52" x2="1204" y2="52" class="legend-line-orange"/>',
        '<text x="1212" y="57" class="legend">pred-rigid</text>',
    ]

    for idx, (variant_name, panel_title, panel_subtitle) in enumerate(PANEL_VARIANTS):
        row = summary[variant_name]
        panel_col = idx % 2
        panel_row = idx // 2
        panel_x = margin_x + panel_col * (panel_width + panel_gap_x)
        panel_y = top_band + panel_row * (panel_height + panel_gap_y)

        image_path = ABLATION_ROOT / variant_name / "evaluation" / "figures" / SAMPLE_STEM
        encoded = _encode_png(image_path)
        metric_text = _short_metric_text(row)

        parts.extend(
            [
                f'<rect x="{panel_x}" y="{panel_y}" width="{panel_width}" height="{panel_height}" class="box"/>',
                f'<text x="{panel_x + 18}" y="{panel_y + 28}" class="panel-title">{panel_title}</text>',
                f'<text x="{panel_x + 18}" y="{panel_y + 49}" class="panel-subtitle">{panel_subtitle}</text>',
                f'<rect x="{panel_x + 16}" y="{panel_y + 58}" width="{panel_width - 32}" height="28" class="metric-box"/>',
                f'<text x="{panel_x + 30}" y="{panel_y + 77}" class="metric">{metric_text}</text>',
                f'<image x="{panel_x + 16}" y="{panel_y + 96}" width="{panel_width - 32}" height="{panel_height - 112}" preserveAspectRatio="xMidYMid meet" href="data:image/png;base64,{encoded}"/>',
            ]
        )

    parts.extend(
        [
            '<text x="48" y="1148" class="caption">Figure note: The MSE-only variant tests point-wise reconstruction alone, while the spectral and attachment-latent ablations isolate the two retained design contributions.</text>',
            '</svg>',
        ]
    )

    output_path = OUTPUT_DIR / "figure_ablation_residual_grid.svg"
    output_path.write_text("".join(parts), encoding="utf-8")
    return output_path


def main() -> None:
    output_path = build_ablation_svg()
    print(f"Saved ablation figure to {output_path}")


if __name__ == "__main__":
    main()
