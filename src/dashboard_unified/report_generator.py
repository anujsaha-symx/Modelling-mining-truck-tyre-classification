"""
Combined Health Report Generator
================================
Generates a polished executive summary report combining visual
inspection and telemetry analysis results. The report is designed
for vehicle maintenance decision-makers.
"""

from pathlib import Path

import pandas as pd

from src.dashboard_unified.health_logic import (
    HEALTH_COLORS,
    compute_health_score,
    classify_overall_health,
    classify_sensor_health,
    generate_combined_recommendations,
)

ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / 'outputs' / 'unified_dashboard'


def generate_combined_report(
    image_result: dict,
    osi_stats: dict,
    sev_dist: dict,
    component_means: dict,
    dominant_component: str,
) -> Path:
    """Write combined_health_report.md and return its path.

    Produces a professional executive summary suitable for
    vehicle maintenance decision-makers.
    """
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    image_class = image_result.get('final_class', 'Non-Tire')
    image_conf = image_result.get('confidence', 0.0)
    gatekeeper_result = image_result.get('gatekeeper_result', 'Non-Tyre')
    gatekeeper_conf = image_result.get('gatekeeper_confidence', 0.0)
    wear_detail = image_result.get('wear_result', {}).get('reason', 'N/A') if image_result.get('wear_result') else 'N/A'
    osi_level = classify_sensor_health(osi_stats['mean_osi'])
    health_colour, health_label = classify_overall_health(image_class, osi_level)
    img_score, sen_score, total_score = compute_health_score(
        image_class, osi_stats['mean_osi'],
    )

    dominant_short = dominant_component.replace('_Risk_Score', '') if dominant_component else 'N/A'

    recs = generate_combined_recommendations(
        image_class, osi_level, dominant_component, component_means,
    )

    if osi_level == 'Critical':
        priority = 'IMMEDIATE'
        interval = 'Within 24 hours'
    elif osi_level == 'Severe' or image_class == 'Bad-Tire':
        priority = 'HIGH'
        interval = 'Within 48 hours'
    elif osi_level == 'Moderate':
        priority = 'MEDIUM'
        interval = 'Within 1 week'
    else:
        priority = 'LOW'
        interval = 'Next scheduled maintenance'

    if image_class == 'Bad-Tire':
        confidence_desc = f'Visual inspection ({image_conf:.0%} confidence)'
    elif image_class == 'Good-Tire':
        confidence_desc = f'Visual inspection ({image_conf:.0%} confidence) + telemetry analysis'
    else:
        confidence_desc = 'Telemetry analysis only (no visual inspection available)'

    lines = [
        '# Vehicle Tyre Health Report',
        '',
        f'*Report generated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}*',
        '',
        '---',
        '',
        '## Executive Summary',
        '',
        f'**Overall Tyre Condition:** {health_colour} - {health_label}',
        f'**Health Score:** {total_score} / 100',
        '',
        '| Assessment Area | Status | Details |',
        '|----------------|--------|---------|',
        f'| Overall Condition | {health_colour} | {health_label} |',
        f'| Operational Severity | {osi_level} | Mean: {osi_stats["mean_osi"]:.2f}, Max: {osi_stats["max_osi"]:.2f} |',
        f'| Visual Inspection | {image_class} | Confidence: {image_conf:.0%} |',
        f'| Dominant Risk Factor | {dominant_short} | Mean score: {component_means.get(dominant_component, 0):.2f} |',
        f'| Recommended Maintenance Action | {recs[0].strip("*").strip() if recs else "Continue regular monitoring"} | See details below |',
        f'| Inspection Priority | {priority} | |',
        f'| Suggested Inspection Interval | {interval} | |',
        f'| Confidence Level | {confidence_desc} | |',
        '',
        '---',
        '',
        '## Visual Inspection Detail',
        '',
        f'- **Gatekeeper Verification:** {gatekeeper_result} (confidence: {gatekeeper_conf:.0%})',
        f'- **Wear Classification:** {image_class}',
        f'- **Wear Detail:** {wear_detail}',
        f'- **Visual Score:** {"{:.0f}/100".format(img_score) if img_score is not None else "N/A"}',
        '',
        '---',
        '',
        '## Telemetry Analysis Detail',
        '',
        f'- **Mean Operational Severity:** {osi_stats["mean_osi"]:.2f}',
        f'- **Maximum Operational Severity:** {osi_stats["max_osi"]:.2f}',
        f'- **95th Percentile:** {osi_stats["p95_osi"]:.2f}',
        f'- **Severity Level:** {osi_level}',
        f'- **Severe Events:** {osi_stats["severe_count"]}',
        f'- **Critical Events:** {osi_stats["critical_count"]}',
        f'- **Telemetry Score:** {sen_score:.1f}/100',
        '',
        '### Severity Time Distribution',
    ]
    for level, pct in sev_dist.items():
        lines.append(f'- {level}: {pct:.2f}%')

    lines += [
        '',
        '---',
        '',
        '## Risk Component Analysis',
        '',
        '| Component | Mean Score | Weight | Weighted Contribution |',
        '|-----------|-----------|--------|----------------------|',
    ]

    WEIGHTS = {
        'Pressure_Risk_Score': 0.15,
        'Thermal_Risk_Score': 0.15,
        'Load_Risk_Score': 0.20,
        'Vibration_Risk_Score': 0.15,
        'Braking_Risk_Score': 0.05,
        'Terrain_Risk_Score': 0.05,
        'Usage_Risk_Score': 0.15,
        'Anomaly_Score': 0.10,
    }

    for col, mean_val in sorted(component_means.items(), key=lambda x: -x[1]):
        short = col.replace('_Risk_Score', '')
        weight = WEIGHTS.get(col, 0)
        contrib = weight * mean_val
        lines.append(f'| {short} | {mean_val:.2f} | {weight:.2f} | {contrib:.2f} |')

    lines.append(f'')
    lines.append(f'**Dominant Risk Factor:** {dominant_short}')
    lines.append('')

    lines += [
        '---',
        '',
        '## Maintenance Recommendations',
        '',
    ]
    for r in recs:
        lines.append(r)

    lines += [
        '',
        '---',
        '',
        '## Severity Threshold Reference',
        '',
        '| Level | Range | Operational Meaning |',
        '|-------|-------|---------------------|',
        '| Normal | 0 - 25 | Within acceptable operating parameters |',
        '| Moderate | 25 - 50 | Elevated stress - increased monitoring advised |',
        '| Severe | 50 - 75 | High stress - inspection required |',
        '| Critical | 75 - 100 | Unsafe conditions - immediate action required |',
        '',
        '---',
        '*Generated by Mining Truck Tyre Health Monitoring System*',
    ]

    path = REPORT_DIR / 'combined_health_report.md'
    path.write_text('\n'.join(lines))
    return path
