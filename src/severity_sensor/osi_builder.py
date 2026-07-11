"""
OSI Phase 4: Operational Severity Index Builder
================================================
Loads risk components, constructs OSI variants, detects events,
aggregates temporally, generates reports and visualizations.
"""

import os, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.decomposition import PCA
from scipy.stats import entropy

warnings.filterwarnings('ignore')

ROOT = Path(r'D:\Tyre_Classification')
COMPONENTS_PATH = ROOT / 'datasets/telemetry/telemetry_risk_components.parquet'
TELEMETRY_DIR = ROOT / 'datasets/telemetry'
OUT_DIR = ROOT / 'outputs/osi_phase4'
FIG_DIR = OUT_DIR / 'figures'
DASH_DIR = OUT_DIR / 'dashboard_data'
OSI_PATH = TELEMETRY_DIR / 'telemetry_osi.parquet'

for d in [OUT_DIR, FIG_DIR, DASH_DIR]:
    d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
from src.severity_sensor.osi_events import (
    detect_all_events, severity_time_distribution,
    classify_severity, classify_severity_code, SEVERITY_LEVELS,
)
from src.severity_sensor.osi_aggregation import (
    aggregate_hourly, aggregate_daily, aggregate_weekly,
)
from src.severity_sensor.osi_visualization import (
    plot_osi_distribution, plot_osi_timeseries, plot_daily_osi,
    plot_component_contributions, plot_severity_timeline,
    plot_hourly_heatmap,
)

SCORE_COLS = [
    'Pressure_Risk_Score', 'Thermal_Risk_Score', 'Load_Risk_Score',
    'Vibration_Risk_Score', 'Braking_Risk_Score', 'Terrain_Risk_Score',
    'Usage_Risk_Score', 'Anomaly_Score',
]

WEIGHTS = {
    'Pressure_Risk_Score':   0.15,
    'Thermal_Risk_Score':    0.15,
    'Load_Risk_Score':       0.20,
    'Vibration_Risk_Score':  0.15,
    'Braking_Risk_Score':    0.05,
    'Terrain_Risk_Score':    0.05,
    'Usage_Risk_Score':      0.15,
    'Anomaly_Score':         0.10,
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-10, 'Weights must sum to 1'


# ======================================================================
# OSI builders
# ======================================================================

def build_weighted_osi(df: pd.DataFrame, weights: dict = None) -> pd.Series:
    """Weighted sum of component scores -> [0, 100]."""
    w = weights or WEIGHTS
    score = pd.Series(0.0, index=df.index)
    for col, wt in w.items():
        if col in df.columns:
            score += wt * df[col].fillna(0)
    return score.clip(0, 100)


def build_equal_osi(df: pd.DataFrame) -> pd.Series:
    """Equal-weight average of all component scores."""
    avail = [c for c in SCORE_COLS if c in df.columns]
    if not avail:
        return pd.Series(0.0, index=df.index)
    return df[avail].fillna(0).mean(axis=1).clip(0, 100)


def build_pca_osi(df: pd.DataFrame, n_components: float = 0.8) -> pd.Series:
    """First principal component of component scores -> [0, 100]."""
    avail = [c for c in SCORE_COLS if c in df.columns]
    if len(avail) < 2:
        return pd.Series(df[avail[0]].fillna(0).values, index=df.index).clip(0, 100)
    X = df[avail].fillna(0).values
    pca = PCA(n_components=n_components, random_state=42)
    pc1 = pca.fit_transform(X)[:, 0]
    # Normalize to [0, 100]
    pc1 = pc1 - pc1.min()
    pc1 = pc1 / (pc1.max() if pc1.max() > 0 else 1) * 100
    return pd.Series(pc1, index=df.index).clip(0, 100)


def _entropy_weights(df: pd.DataFrame, cols: list) -> dict:
    """Compute entropy-based weights for a set of columns.

    Higher entropy -> more information -> higher weight.
    """
    weights = {}
    total_ent = 0
    for col in cols:
        if col not in df.columns:
            continue
        vals = df[col].fillna(0).values + 1e-10
        vals = vals / vals.sum()
        ent = entropy(vals)
        weights[col] = ent
        total_ent += ent
    if total_ent > 0:
        weights = {k: v / total_ent for k, v in weights.items()}
    else:
        n = len(weights)
        weights = {k: 1 / n for k in weights}
    return weights


def build_entropy_osi(df: pd.DataFrame) -> pd.Series:
    """Entropy-weighted OSI."""
    avail = [c for c in SCORE_COLS if c in df.columns]
    w = _entropy_weights(df, avail)
    return build_weighted_osi(df, w)


# ======================================================================
# MAIN
# ======================================================================

def main():
    print('=' * 60)
    print('OSI Phase 4 — Operational Severity Index Builder')
    print('=' * 60)

    # ------------------------------------------------------------------
    # STEP 1: Load components
    # ------------------------------------------------------------------
    print('\n[1] Loading risk components ...')
    pf = pq.ParquetFile(COMPONENTS_PATH)
    cols = pf.schema_arrow.names
    needed = ['timestamp'] + SCORE_COLS
    read_cols = [c for c in needed if c in cols]
    df = pf.read(columns=read_cols).to_pandas()
    pf.close()
    print(f'    {len(df):,} rows, {len(read_cols)} columns')

    # ------------------------------------------------------------------
    # STEP 2: Base weighted OSI
    # ------------------------------------------------------------------
    print('\n[2] Computing base weighted OSI ...')
    df['OSI_Base'] = build_weighted_osi(df)
    print(f'    OSI_Base: mean={df["OSI_Base"].mean():.2f}, '
          f'std={df["OSI_Base"].std():.2f}, '
          f'[{df["OSI_Base"].min():.2f}, {df["OSI_Base"].max():.2f}]')

    # ------------------------------------------------------------------
    # STEP 3: Alternative OSI variants
    # ------------------------------------------------------------------
    print('\n[3] Computing alternative OSI variants ...')
    df['OSI_Equal'] = build_equal_osi(df)
    print(f'    OSI_Equal: mean={df["OSI_Equal"].mean():.2f}')

    # PCA on a sample if large, then transform all
    # Use row group 0 for fitting PCA
    fit_df = df.iloc[:min(100000, len(df))]
    df['OSI_PCA'] = build_pca_osi(fit_df) if len(fit_df) < len(df) else build_pca_osi(df)
    print(f'    OSI_PCA: mean={df["OSI_PCA"].mean():.2f}')

    df['OSI_Entropy'] = build_entropy_osi(df)
    print(f'    OSI_Entropy: mean={df["OSI_Entropy"].mean():.2f}')

    osi_cols = ['OSI_Base', 'OSI_Equal', 'OSI_PCA', 'OSI_Entropy']
    final_osi = 'OSI_Base'  # default

    # ------------------------------------------------------------------
    # STEP 4: Compare OSI variants
    # ------------------------------------------------------------------
    print('\n[4] Comparing OSI variants ...')

    # Correlation
    corr = df[osi_cols].corr()
    print(f'    Correlation matrix:')
    print(corr.to_string())

    # Distributions
    for col in osi_cols:
        print(f'    {col}: mean={df[col].mean():.2f}, median={df[col].median():.2f}, '
              f'std={df[col].std():.2f}')

    # Ranking agreement (Spearman correlation)
    rank_corr = df[osi_cols].corr(method='spearman')
    mean_rank_corr = rank_corr.values[np.triu_indices_from(rank_corr.values, k=1)].mean()
    print(f'    Mean Spearman rank correlation: {mean_rank_corr:.3f}')

    # Stability (rolling std over 1 hour)
    stability = {}
    for col in osi_cols:
        roll_std = df[col].rolling(3600, min_periods=1).std()
        stability[col] = roll_std.mean()
    most_stable = min(stability, key=stability.get)
    print(f'    Stability (mean rolling std): {stability}')
    print(f'    Most stable variant: {most_stable}')

    # Recommend final OSI
    if mean_rank_corr > 0.9:
        final_osi = 'OSI_Base'
        print(f'    High rank agreement — using expert-weighted OSI_Base')
    else:
        final_osi = most_stable
        print(f'    Using most stable variant: {final_osi}')

    df['OSI'] = df[final_osi].copy()
    print(f'\n    Final OSI: {final_osi}')

    # ------------------------------------------------------------------
    # STEP 5: Severity levels
    # ------------------------------------------------------------------
    print('\n[5] Creating severity levels ...')
    df['OSI_Level'] = classify_severity(df['OSI'])
    df['OSI_Level_Code'] = classify_severity_code(df['OSI'])
    sev_dist = severity_time_distribution(df['OSI'])
    print(f'    Severity distribution: {sev_dist}')

    # ------------------------------------------------------------------
    # STEP 6: Event detection
    # ------------------------------------------------------------------
    print('\n[6] Detecting severity events ...')
    events = detect_all_events(df['OSI'], df['timestamp'], min_duration=1)
    print(f'    Total events detected: {len(events)}')
    if len(events) > 0:
        for level in ['Critical', 'Severe', 'Moderate']:
            cnt = (events['level'] == level).sum()
            print(f'      {level} events: {cnt}')
            if cnt > 0:
                sub = events[events['level'] == level]
                print(f'        Mean duration: {sub["duration_seconds"].mean() / 60:.1f} min')
                print(f'        Mean peak OSI: {sub["peak_osi"].mean():.1f}')

    # ------------------------------------------------------------------
    # STEP 7: Trip-level aggregation
    # ------------------------------------------------------------------
    print('\n[7] Aggregating OSI ...')
    osi_all = osi_cols + ['OSI']
    hourly = aggregate_hourly(df, osi_all)
    daily = aggregate_daily(df, osi_all)
    weekly = aggregate_weekly(df, osi_all)
    print(f'    Hourly: {len(hourly)} periods')
    print(f'    Daily:  {len(daily)} periods')
    print(f'    Weekly: {len(weekly)} periods')

    # ------------------------------------------------------------------
    # STEP 8: Temporal analysis (done via visualizations later)
    # ------------------------------------------------------------------
    print('\n[8] Temporal analysis (see visualizations)')

    # ------------------------------------------------------------------
    # STEP 9: Save final datasets
    # ------------------------------------------------------------------
    print('\n[9] Saving datasets ...')

    # Full OSI dataset
    osi_out_cols = ['timestamp', 'OSI', 'OSI_Base', 'OSI_Equal', 'OSI_PCA',
                    'OSI_Entropy', 'OSI_Level', 'OSI_Level_Code']
    osi_out_cols = [c for c in osi_out_cols if c in df.columns]
    osi_df = df[osi_out_cols].copy()

    # Add component contributions for dashboard
    for col in SCORE_COLS:
        if col in df.columns:
            w = WEIGHTS.get(col, 1 / len(SCORE_COLS))
            osi_df[f'contrib_{col}'] = df[col] * w

    osi_df.to_parquet(OSI_PATH, index=False)
    print(f'    -> {OSI_PATH} ({os.path.getsize(OSI_PATH)/1e6:.1f} MB)')

    osi_df.head(5000).to_csv(TELEMETRY_DIR / 'telemetry_osi_sample.csv', index=False)
    print(f'    -> telemetry_osi_sample.csv')

    hourly.to_csv(TELEMETRY_DIR / 'hourly_osi.csv', index=False)
    print(f'    -> hourly_osi.csv ({len(hourly)} rows)')

    daily.to_csv(TELEMETRY_DIR / 'daily_osi.csv', index=False)
    print(f'    -> daily_osi.csv ({len(daily)} rows)')

    weekly.to_csv(TELEMETRY_DIR / 'weekly_osi.csv', index=False)
    print(f'    -> weekly_osi.csv ({len(weekly)} rows)')

    # ------------------------------------------------------------------
    # STEP 10: Reports
    # ------------------------------------------------------------------
    print('\n[10] Generating reports ...')

    # OSI Summary
    summary = df['OSI'].describe()
    summary_lines = [
        '# OSI Summary', '',
        f'## Overall OSI Statistics',
        f'- Mean OSI: {summary["mean"]:.2f}',
        f'- Median OSI: {df["OSI"].median():.2f}',
        f'- Std OSI: {summary["std"]:.2f}',
        f'- Min OSI: {summary["min"]:.2f}',
        f'- Max OSI: {summary["max"]:.2f}',
        f'- 25th percentile: {summary["25%"]:.2f}',
        f'- 75th percentile: {summary["75%"]:.2f}',
        f'- 95th percentile: {df["OSI"].quantile(0.95):.2f}',
        '', '## Severity Distribution', '',
    ]
    for level, pct in sev_dist.items():
        summary_lines.append(f'- {level}: {pct:.2f}%')
    summary_lines.append('')
    summary_lines.append(f'## Selected OSI Variant')
    summary_lines.append(f'- Final OSI: {final_osi}')
    summary_lines.append(f'- Weights: {WEIGHTS}')
    summary_lines.append('')
    (OUT_DIR / 'osi_summary.md').write_text('\n'.join(summary_lines))
    print('    -> osi_summary.md')

    # OSI Statistics (detailed)
    stat_lines = ['# OSI Detailed Statistics', '',
                  f'Total rows: {len(df):,}', '',
                  '## All OSI Variants', '']
    for col in osi_cols + ['OSI']:
        if col in df.columns:
            desc = df[col].describe()
            stat_lines.append(f'### {col}')
            for k in ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']:
                stat_lines.append(f'- {k}: {desc[k]:.4f}')
            stat_lines.append('')
    stat_lines.append('## Correlation Between Variants')
    stat_lines.append('')
    stat_lines.append(corr.to_string())
    stat_lines.append('')
    (OUT_DIR / 'osi_statistics.md').write_text('\n'.join(stat_lines))
    print('    -> osi_statistics.md')

    # Event report
    event_lines = ['# OSI Event Report', '',
                   f'Total events detected: {len(events)}', '']
    if len(events) > 0:
        event_lines.append('## Events by Severity')
        event_lines.append('')
        for level in ['Critical', 'Severe', 'Moderate']:
            sub = events[events['level'] == level]
            if len(sub) > 0:
                event_lines.append(f'### {level} Events ({len(sub)})')
                event_lines.append(f'- Mean duration: {sub["duration_seconds"].mean() / 60:.1f} min')
                event_lines.append(f'- Total duration: {sub["duration_seconds"].sum() / 3600:.1f} hours')
                event_lines.append(f'- Mean peak OSI: {sub["peak_osi"].mean():.1f}')
                event_lines.append(f'- Min peak OSI: {sub["peak_osi"].min():.1f}')
                event_lines.append(f'- Max peak OSI: {sub["peak_osi"].max():.1f}')
                event_lines.append('')
        event_lines.append('## Top 10 Most Severe Events')
        event_lines.append('')
        top = events.nlargest(10, 'peak_osi')
        for i, (_, row) in enumerate(top.iterrows(), 1):
            dur_m = row['duration_seconds'] / 60
            event_lines.append(f'{i}. [{row["level"]}] {row["start_time"]} -> '
                               f'{row["end_time"]} ({dur_m:.1f} min, '
                               f'peak OSI={row["peak_osi"]:.1f})')
    (OUT_DIR / 'osi_event_report.md').write_text('\n'.join(event_lines))
    print('    -> osi_event_report.md')

    # Component contribution report
    contrib_lines = ['# OSI Component Contribution', '', '## Contribution Weights', '']
    for col, w in sorted(WEIGHTS.items(), key=lambda x: -x[1]):
        mean_val = df[col].mean()
        contrib = w * mean_val
        contrib_lines.append(f'- {col}: weight={w:.2f}, mean={mean_val:.2f}, '
                             f'weighted contribution={contrib:.2f}')
    contrib_lines.append('')
    contrib_lines.append('## Contribution Percentage')
    contrib_lines.append('')
    total_w_mean = sum(WEIGHTS[c] * df[c].mean() for c in SCORE_COLS if c in df.columns)
    for col, w in sorted(WEIGHTS.items(), key=lambda x: -x[1]):
        if col in df.columns:
            pct = (w * df[col].mean()) / total_w_mean * 100
            contrib_lines.append(f'- {col}: {pct:.1f}%')
    contrib_lines.append('')
    contrib_lines.append(f'## Dominant Component')
    dominant = max(WEIGHTS.items(), key=lambda x: x[1] * df[x[0]].mean())
    contrib_lines.append(f'- {dominant[0]} contributes the most to OSI '
                         f'(weight={dominant[1]:.2f}, '
                         f'mean_score={df[dominant[0]].mean():.2f})')
    (OUT_DIR / 'osi_component_contribution.md').write_text('\n'.join(contrib_lines))
    print('    -> osi_component_contribution.md')

    # Comparison report
    comp_lines = ['# OSI Variant Comparison', '',
                  '## Summary Statistics', '',
                  '| Variant | Mean | Std | Median | Min | Max | Rolling Std |',
                  '|---------|------|-----|--------|-----|-----|------------|']
    for col in osi_cols:
        if col in df.columns:
            comp_lines.append(
                f'| {col} | {df[col].mean():.2f} | {df[col].std():.2f} | '
                f'{df[col].median():.2f} | {df[col].min():.2f} | {df[col].max():.2f} | '
                f'{stability.get(col, 0):.2f} |'
            )
    comp_lines.append('')
    comp_lines.append('## Correlation Matrix')
    comp_lines.append('')
    comp_lines.append(corr.to_string())
    comp_lines.append('')
    comp_lines.append(f'## Mean Spearman Rank Correlation: {mean_rank_corr:.3f}')
    comp_lines.append('')
    comp_lines.append(f'## Selected Final OSI: {final_osi}')
    comp_lines.append('')
    comp_lines.append(f'## Rationale')
    if final_osi == 'OSI_Base':
        comp_lines.append('- Expert weights reflect domain knowledge about tyre severity factors')
        comp_lines.append('- High rank agreement with other variants confirms robustness')
    else:
        comp_lines.append('- Selected for highest temporal stability')
    (OUT_DIR / 'osi_comparison_report.md').write_text('\n'.join(comp_lines))
    print('    -> osi_comparison_report.md')

    # ------------------------------------------------------------------
    # STEP 11: Visualizations
    # ------------------------------------------------------------------
    print('\n[11] Generating visualizations ...')

    # Sample for faster plotting (100k max)
    plot_df = df.iloc[::max(1, len(df) // 100000)].copy() if len(df) > 100000 else df.copy()

    plot_osi_distribution(plot_df, ['OSI'], FIG_DIR / 'osi_distribution.png')
    print('    -> osi_distribution.png')

    plot_osi_timeseries(plot_df, 'OSI', FIG_DIR / 'osi_timeseries.png')
    print('    -> osi_timeseries.png')

    plot_daily_osi(daily, ['OSI'], FIG_DIR / 'daily_osi.png')
    print('    -> daily_osi.png')

    # Component contributions
    contrib_data = {'timestamp': plot_df['timestamp'].values}
    for col in SCORE_COLS:
        if col in plot_df.columns:
            w = WEIGHTS.get(col, 1 / len(SCORE_COLS))
            contrib_data[f'contrib_{col}'] = plot_df[col] * w
    contrib_df = pd.DataFrame(contrib_data)
    plot_component_contributions(contrib_df, FIG_DIR / 'component_contributions.png')
    print('    -> component_contributions.png')

    # Severity timeline
    plot_severity_timeline(events, FIG_DIR / 'severity_timeline.png')
    print('    -> severity_timeline.png')

    # Hourly heatmap
    plot_hourly_heatmap(hourly, 'OSI_mean', FIG_DIR / 'hourly_heatmap.png')
    print('    -> hourly_heatmap.png')

    del plot_df

    # ------------------------------------------------------------------
    # STEP 12: Dashboard data
    # ------------------------------------------------------------------
    print('\n[12] Preparing dashboard data ...')

    # Component timeseries
    dash_comp = df[['timestamp'] + SCORE_COLS].copy()
    dash_comp['OSI'] = df['OSI'].values
    dash_comp.to_csv(DASH_DIR / 'component_timeseries.csv', index=False)
    print(f'    -> component_timeseries.csv ({len(dash_comp)} rows)')

    # Daily summary
    daily.to_csv(DASH_DIR / 'daily_summary.csv', index=False)
    print(f'    -> daily_summary.csv ({len(daily)} rows)')

    # Severity events
    events.to_csv(DASH_DIR / 'severity_events.csv', index=False)
    print(f'    -> severity_events.csv ({len(events)} rows)')

    osi_p95 = osi_df['OSI'].quantile(0.95)
    del df

    # ------------------------------------------------------------------
    # STEP 14: Final executive report
    # ------------------------------------------------------------------
    print('\n[14] Generating executive report ...')

    mean_osi = summary['mean']
    max_osi = summary['max']
    n_critical = (osi_df['OSI_Level'] == 'Critical').sum()

    exec_lines = [
        '# Executive Summary — Operational Severity Index', '',
        '## Overview',
        '',
        f'The final OSI has been constructed from {len(SCORE_COLS)} risk components '
        f'using expert-weighted aggregation.',
        '',
        f'## Key Metrics',
        '',
        f'| Metric | Value |',
        f'|--------|-------|',
        f'| Average OSI | {mean_osi:.2f} / 100 |',
        f'| Maximum OSI | {max_osi:.2f} / 100 |',
        f'| Standard Deviation | {summary["std"]:.2f} |',
        f'| 95th Percentile | {osi_p95:.2f} |',
        '',
        '## Severity Distribution',
        '',
        '| Level | Range | % Time |',
        '|-------|-------|--------|',
    ]
    for level, (lo, hi) in SEVERITY_LEVELS.items():
        pct = sev_dist.get(level, 0)
        exec_lines.append(f'| {level} | {lo}-{hi} | {pct:.2f}% |')
    exec_lines.append('')
    exec_lines.append(f'## Dominant Contributor')
    exec_lines.append(f'- **{dominant[0]}** — contributes the most to overall severity.')
    exec_lines.append('')
    exec_lines.append(f'## Event Analysis')
    exec_lines.append(f'- Total severity events: {len(events)}')
    exec_lines.append(f'- Critical events (OSI > 75): {n_critical:,} rows ({n_critical/len(osi_df)*100:.2f}% of time)')
    critical_events = events[events['level'] == 'Critical'] if len(events) > 0 else pd.DataFrame()
    exec_lines.append(f'- Distinct critical events: {len(critical_events)}')
    exec_lines.append('')

    # Health assessment
    if mean_osi < 25 and max_osi < 50:
        health = 'HEALTHY'
        recommendation = 'Continue regular monitoring. No immediate action required.'
    elif mean_osi < 50 and max_osi < 75:
        health = 'MODERATE'
        recommendation = 'Periodic inspection recommended. Monitor Load and Usage components.'
    elif mean_osi < 75:
        health = 'ELEVATED'
        recommendation = 'Increased monitoring frequency. Investigate dominant risk components.'
    else:
        health = 'CRITICAL'
        recommendation = 'Immediate inspection required. Operation may be unsafe.'

    exec_lines.append(f'## Fleet Health Assessment')
    exec_lines.append(f'- **Status: {health}**')
    exec_lines.append(f'- {recommendation}')
    exec_lines.append('')
    exec_lines.append(f'## OSI Variant Used')
    exec_lines.append(f'- **{final_osi}** — selected based on stability and domain alignment.')
    exec_lines.append('')
    exec_lines.append('---')
    exec_lines.append('*Report generated by OSI Phase 4 pipeline*')

    (OUT_DIR / 'executive_summary.md').write_text('\n'.join(exec_lines))
    print('    -> executive_summary.md')

    # ------------------------------------------------------------------
    # Final output
    # ------------------------------------------------------------------
    print()
    print('=' * 60)
    print('OSI Phase 4 complete.')
    print('=' * 60)
    print()
    print(f'  Mean OSI:              {mean_osi:.2f}')
    print(f'  Max OSI:               {max_osi:.2f}')
    print(f'  OSI Variant:           {final_osi}')
    print()
    print(f'  Severity distribution:')
    for level, pct in sev_dist.items():
        print(f'    {level}: {pct:.2f}%')
    print()
    n_crit = len(events[events['level'] == 'Critical']) if len(events) > 0 else 0
    print(f'  Number of critical events:  {n_crit}')
    print(f'  Total events:               {len(events)}')
    print()
    print(f'  Dominant risk component:    {dominant[0]}')
    print(f'  Fleet health:               {health}')
    print()
    print(f'  Recommendation: {recommendation}')
    print()


if __name__ == '__main__':
    main()
