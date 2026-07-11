"""
OSI Phase 3: Risk Component Builder
=====================================
Loads telemetry_features.parquet, computes 7 risk component scores
+ anomaly scores, saves telemetry_risk_components.parquet, and
generates Phase 3 reports.

Memory-efficient: processes one row group at a time.
"""

import os, sys, warnings
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

ROOT = Path(r'D:\Tyre_Classification')
FEATURES_PATH = ROOT / 'datasets/telemetry/telemetry_features.parquet'
OUT_DIR = ROOT / 'outputs/osi_phase3'
FIG_DIR = OUT_DIR / 'figures'
TELEMETRY_DIR = ROOT / 'datasets/telemetry'
COMPONENTS_PATH = TELEMETRY_DIR / 'telemetry_risk_components.parquet'
SAMPLE_PATH = TELEMETRY_DIR / 'telemetry_risk_components_sample.csv'

for d in [OUT_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
from src.severity_sensor.risk_scoring import compute_all_risk_scores
from src.severity_sensor.anomaly_detection import AnomalyEnsemble, DERIVED_FEATURES

CHUNK_ROWS = 432_000  # one row group

# ======================================================================
# STEP 1: Feature inventory groups
# ======================================================================
FEATURE_GROUPS = {
    'Pressure': [
        'pressure_mean', 'pressure_std', 'pressure_min', 'pressure_max',
        'pressure_difference_between_tyres', 'pressure_drop_rate',
        'underinflation_flag', 'overinflation_flag', 'pressure_anomaly_flag',
        'pressure_zscore',
    ],
    'Temperature': [
        'temperature_mean', 'temperature_std', 'temperature_max',
        'temperature_difference_between_tyres', 'temperature_rise_rate',
        'overtemperature_flag', 'temperature_zscore',
    ],
    'Load': [
        'payload_mean', 'payload_std', 'payload_max',
        'payload_change_rate', 'payload_percent_of_max',
        'overload_flag', 'cumulative_payload_exposure',
    ],
    'Vibration': [
        'acceleration_magnitude', 'peak_acceleration', 'rms_vibration',
        'crest_factor', 'impact_event_flag', 'high_vibration_flag',
    ],
    'Braking': [
        'brake_frequency', 'hard_braking_flag', 'brake_duration',
        'cumulative_braking_time',
    ],
    'Terrain': [
        'pitch_std', 'roll_std', 'slope_event_flag', 'rough_terrain_flag',
    ],
    'Usage': [
        'distance_travelled', 'cumulative_distance',
        'speed_mean', 'speed_max', 'high_speed_flag',
    ],
}
ALL_DERIVED = sorted({c for cols in FEATURE_GROUPS.values() for c in cols})


# ======================================================================
# STEP 2: Normalisation helpers
# ======================================================================

class Normaliser:
    """Fits min-max, robust, and percentile normalisers on a sample."""

    def __init__(self):
        self.mm_min = {}
        self.mm_max = {}
        self.r_median = {}
        self.r_iqr = {}
        self._fitted = False

    def fit(self, df: pd.DataFrame, columns: Optional[List[str]] = None):
        cols = columns or [c for c in df.columns if c != 'timestamp'
                           and np.issubdtype(df[c].dtype, np.number)]
        for c in cols:
            vals = df[c].dropna().values
            if len(vals) == 0:
                continue
            # Min-max
            self.mm_min[c] = float(np.min(vals))
            self.mm_max[c] = float(np.max(vals))
            # Robust
            self.r_median[c] = float(np.median(vals))
            q1, q3 = np.percentile(vals, [25, 75])
            self.r_iqr[c] = float(q3 - q1) if q3 - q1 > 1e-12 else 1.0
        self._fitted = True
        return self

    def minmax(self, series: pd.Series) -> pd.Series:
        c = series.name
        if c not in self.mm_min:
            return series * 0.0
        r = self.mm_max[c] - self.mm_min[c]
        return (series - self.mm_min[c]) / (r if r > 1e-12 else 1.0)

    def robust(self, series: pd.Series) -> pd.Series:
        c = series.name
        if c not in self.r_median:
            return series * 0.0
        return (series - self.r_median[c]) / self.r_iqr[c]

    def percentile(self, series: pd.Series) -> pd.Series:
        return series.rank(pct=True)

    def transform(self, df: pd.DataFrame,
                  columns: Optional[List[str]] = None) -> pd.DataFrame:
        cols = columns or [c for c in df.columns if c != 'timestamp'
                           and np.issubdtype(df[c].dtype, np.number)]
        result = pd.DataFrame({'timestamp': df['timestamp'].values})
        for c in cols:
            if c not in df.columns:
                continue
            s = df[c].fillna(df[c].median()) if df[c].isna().any() else df[c]
            result[f'{c}_minmax'] = self.minmax(s)
            result[f'{c}_robust'] = self.robust(s)
            result[f'{c}_percentile'] = self.percentile(s)
        return result


# ======================================================================
# Memory-efficient row-group reader
# ======================================================================

def _build_rg_bounds(pf):
    bounds = []
    cum = 0
    for i in range(pf.metadata.num_row_groups):
        cum += pf.metadata.row_group(i).num_rows
        bounds.append(cum)
    return bounds


def _read_rg(pf, rg_idx: int, columns=None) -> pd.DataFrame:
    return pf.read_row_group(rg_idx, columns=columns).to_pandas()


# ======================================================================
# MAIN
# ======================================================================

def main():
    print('=' * 60)
    print('OSI Phase 3 — Risk Component Builder')
    print('=' * 60)

    # ------------------------------------------------------------------
    # Load schema
    # ------------------------------------------------------------------
    print('\n[1] Loading feature dataset ...')
    pf = pq.ParquetFile(FEATURES_PATH)
    rg_bounds = _build_rg_bounds(pf)
    total_rows = rg_bounds[-1]
    all_cols = pf.schema_arrow.names
    print(f'    {total_rows:,} rows, {len(all_cols)} cols, '
          f'{pf.metadata.num_row_groups} row groups')

    # Keep only needed columns for scoring + normalisation
    keep_cols = ['timestamp'] + ALL_DERIVED
    keep_cols = [c for c in keep_cols if c in all_cols]
    n_groups = len(FEATURE_GROUPS)
    print(f'    Feature inventory: {n_groups} groups, '
          f'{len(keep_cols) - 1} derived features')

    # ------------------------------------------------------------------
    # Fit normalisers and anomaly detector on row group 0
    # ------------------------------------------------------------------
    print('\n[2] Fitting normalisers on row group 0 ...')
    rg0 = _read_rg(pf, 0, columns=keep_cols)
    print(f'    Row group 0: {len(rg0):,} rows')

    norm = Normaliser().fit(rg0)
    print(f'    Normalisers fitted on {len(norm.mm_min)} features.')

    print('\n[10] Fitting anomaly detectors on row group 0 ...')
    anomaly = AnomalyEnsemble(random_state=42)
    anomaly.fit(rg0)

    # ------------------------------------------------------------------
    # Process each row group: normalise, score, detect anomalies, write
    # ------------------------------------------------------------------
    print('\n[3-10] Processing row groups ...')
    writer = None
    num_rgs = pf.metadata.num_row_groups

    for rg_idx in range(num_rgs):
        print(f'\n  Row group {rg_idx + 1}/{num_rgs} ...')
        df = _read_rg(pf, rg_idx, columns=keep_cols)
        print(f'    Loaded {len(df):,} rows')

        # STEP 2: Normalise
        norm_df = norm.transform(df, columns=ALL_DERIVED)

        # STEPS 3-9: Risk scores
        scores = compute_all_risk_scores(df)

        # STEP 10: Anomaly scores
        anom_df = anomaly.transform(df)

        # Combine
        out = scores.merge(norm_df, on='timestamp', how='left')
        out = out.merge(anom_df, on='timestamp', how='left')

        # Write
        table = pa.Table.from_pandas(out, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(str(COMPONENTS_PATH), table.schema)
        writer.write_table(table)
        print(f'    Written {out.shape[1]} cols × {len(out):,} rows')

        del df, norm_df, scores, anom_df, out, table

    if writer is not None:
        writer.close()
    pf.close()
    print(f'\n    -> {COMPONENTS_PATH} ({os.path.getsize(COMPONENTS_PATH) / 1e6:.1f} MB)')

    # ------------------------------------------------------------------
    # STEP 11b: Sample CSV
    # ------------------------------------------------------------------
    print('\n[11] Writing sample CSV ...')
    sample = pd.read_parquet(COMPONENTS_PATH)
    sample.head(5000).to_csv(SAMPLE_PATH, index=False)
    print(f'    -> {SAMPLE_PATH}')
    del sample

    # ------------------------------------------------------------------
    # Load component data for reports
    # ------------------------------------------------------------------
    print('\n[12] Loading components for reports ...')
    comp_pf = pq.ParquetFile(COMPONENTS_PATH)
    comp_df = comp_pf.read().to_pandas()
    comp_cols = [c for c in comp_df.columns if c != 'timestamp']
    print(f'    {len(comp_cols)} component columns, {len(comp_df):,} rows')

    # Score columns
    score_cols = [c for c in comp_cols if c.endswith('_Risk_Score') or c == 'Anomaly_Score']
    print(f'    Risk scores: {score_cols}')

    # ------------------------------------------------------------------
    # STEP 12a: Component statistics
    # ------------------------------------------------------------------
    print('\n[12a] Generating component statistics ...')
    stat_lines = ['# Risk Component Statistics', '',
                  f'Total rows: {len(comp_df):,}', '']
    for col in score_cols:
        if col in comp_df.columns:
            desc = comp_df[col].describe()
            stat_lines.append(f'## {col}')
            for k in ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']:
                stat_lines.append(f'- {k}: {desc[k]:.4f}')
            high_pct = (comp_df[col] > 80).mean() * 100
            moderate_pct = ((comp_df[col] > 50) & (comp_df[col] <= 80)).mean() * 100
            low_pct = (comp_df[col] <= 50).mean() * 100
            stat_lines.append(f'- % High risk (>80): {high_pct:.2f}%')
            stat_lines.append(f'- % Moderate risk (50-80): {moderate_pct:.2f}%')
            stat_lines.append(f'- % Low risk (<=50): {low_pct:.2f}%')
            stat_lines.append('')
    Path(OUT_DIR / 'component_statistics.md').write_text('\n'.join(stat_lines))
    print(f'    -> {OUT_DIR}/component_statistics.md')

    # ------------------------------------------------------------------
    # STEP 12b: Component distributions
    # ------------------------------------------------------------------
    print('\n[12b] Generating component distribution plot ...')
    n_scores = len(score_cols)
    fig, axes = plt.subplots(n_scores, 1, figsize=(12, 3 * n_scores))
    if n_scores == 1:
        axes = [axes]
    for ax, col in zip(axes, score_cols):
        vals = comp_df[col].dropna().values
        ax.hist(vals, bins=80, edgecolor='none', alpha=0.7, color='steelblue')
        ax.axvline(50, color='orange', linestyle='--', alpha=0.7, label='Moderate')
        ax.axvline(80, color='red', linestyle='--', alpha=0.7, label='High')
        ax.set_title(col)
        ax.set_ylabel('Frequency')
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'component_distributions.png', dpi=150)
    plt.close(fig)
    print(f'    -> {FIG_DIR}/component_distributions.png')

    # ------------------------------------------------------------------
    # STEP 12c: Correlation between components
    # ------------------------------------------------------------------
    print('\n[12c] Generating component correlation heatmap ...')
    corr = comp_df[score_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1, square=True,
                cbar_kws={'shrink': 0.6}, ax=ax)
    ax.set_title('Correlation Between Risk Components')
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'correlation_between_components.png', dpi=150)
    plt.close(fig)
    print(f'    -> {FIG_DIR}/correlation_between_components.png')

    # ------------------------------------------------------------------
    # STEP 12d: Anomaly summary
    # ------------------------------------------------------------------
    print('\n[12d] Generating anomaly summary ...')
    anom_score = comp_df['Anomaly_Score']
    anom_lines = ['# Anomaly Detection Summary', '',
                  f'Total rows: {len(comp_df):,}', '']
    anom_lines.append(f'## Anomaly_Score Distribution')
    anom_lines.append(f'- Mean: {anom_score.mean():.2f}')
    anom_lines.append(f'- Std:  {anom_score.std():.2f}')
    anom_lines.append(f'- Min:  {anom_score.min():.2f}')
    anom_lines.append(f'- Max:  {anom_score.max():.2f}')
    for thresh, label in [(90, 'Critical'), (80, 'High'), (70, 'Moderate')]:
        count = (anom_score > thresh).sum()
        pct = count / len(comp_df) * 100
        anom_lines.append(f'- {label} anomalies (>{thresh}): {count:,} ({pct:.2f}%)')
    anom_lines.append('')

    # Per-detector summary
    for det in ['Anomaly_IF', 'Anomaly_LOF', 'Anomaly_OCSVM']:
        if det in comp_df.columns:
            vals = comp_df[det]
            anom_lines.append(f'## {det}')
            anom_lines.append(f'- Mean: {vals.mean():.2f}')
            anom_lines.append(f'- % flagged > 80: {(vals > 80).mean() * 100:.2f}%')
            anom_lines.append('')

    # High-risk event analysis
    anom_lines.append('## High-Risk Event Analysis')
    for col in score_cols:
        high = (comp_df[col] > 80).sum()
        pct = high / len(comp_df) * 100
        anom_lines.append(f'- {col}: {high:,} high-risk events ({pct:.2f}%)')

    # Concurrency
    n_high = (comp_df[[c for c in score_cols if c in comp_df.columns]] > 80).sum(axis=1)
    anom_lines.append('')
    anom_lines.append('## Concurrent High Risk')
    for n in range(1, len(score_cols) + 1):
        cnt = (n_high >= n).sum()
        pct = cnt / len(comp_df) * 100
        anom_lines.append(f'- >= {n} components in high risk: {cnt:,} rows ({pct:.2f}%)')

    Path(OUT_DIR / 'anomaly_summary.md').write_text('\n'.join(anom_lines))
    print(f'    -> {OUT_DIR}/anomaly_summary.md')

    # ------------------------------------------------------------------
    # STEP 12e: Risk component summary
    # ------------------------------------------------------------------
    print('\n[12e] Generating risk component summary ...')
    summary_lines = ['# Risk Component Summary', '',
                     '## Component Score Overview', '']
    summary_lines.append('| Component | Mean | Std | Min | Max | % High Risk (>80) |')
    summary_lines.append('|-----------|------|-----|-----|-----|-------------------|')
    for col in score_cols:
        desc = comp_df[col].describe()
        high_pct = (comp_df[col] > 80).mean() * 100
        summary_lines.append(
            f'| {col} | {desc["mean"]:.2f} | {desc["std"]:.2f} | '
            f'{desc["min"]:.2f} | {desc["max"]:.2f} | {high_pct:.2f}% |'
        )
    summary_lines.append('')
    summary_lines.append('## Dominant Risk Components')
    summary_lines.append('')
    mean_scores = {c: comp_df[c].mean() for c in score_cols}
    ranked = sorted(mean_scores.items(), key=lambda x: -x[1])
    for i, (name, val) in enumerate(ranked, 1):
        summary_lines.append(f'{i}. {name}: {val:.2f} (average)')
    summary_lines.append('')
    summary_lines.append(f'Dominant component: **{ranked[0][0]}** '
                         f'(mean = {ranked[0][1]:.2f})')
    summary_lines.append('')
    summary_lines.append('## Data Coverage')
    summary_lines.append(f'- Total rows: {len(comp_df):,}')
    summary_lines.append(f'- Time range: {comp_df["timestamp"].min()} → '
                         f'{comp_df["timestamp"].max()}')
    summary_lines.append(f'- Duration: {comp_df["timestamp"].max() - comp_df["timestamp"].min()}')
    summary_lines.append('')
    Path(OUT_DIR / 'risk_component_summary.md').write_text('\n'.join(summary_lines))
    print(f'    -> {OUT_DIR}/risk_component_summary.md')

    comp_pf.close()
    del comp_df

    # ------------------------------------------------------------------
    # STEP 13: OSI Readiness report
    # ------------------------------------------------------------------
    print('\n[13] Generating OSI readiness report ...')

    # Reload for a fresh look at the data
    comp_df = pd.read_parquet(COMPONENTS_PATH)
    score_df = comp_df[[c for c in score_cols if c in comp_df.columns]]

    # 1. Stability: rolling mean of each component should not jump wildly
    stability = {}
    for col in score_df.columns:
        roll_std = score_df[col].rolling(3600, min_periods=1).std()
        stability[col] = {
            'mean_rolling_std': roll_std.mean(),
            'std_rolling_std': roll_std.std(),
        }

    # 2. Dominant risk
    means = score_df.mean()
    dominant = means.idxmax()
    dominant_val = means.max()

    # 3. High-risk event frequency (per hour, assuming 1 Hz)
    high_risk_counts = {}
    for col in score_df.columns:
        n_high = (score_df[col] > 80).sum()
        freq_per_hour = n_high / (len(comp_df) / 3600)
        high_risk_counts[col] = {'count': n_high, 'freq_per_hour': freq_per_hour}

    # Concurrency: rows where at least 2 components are high-risk
    n_high = (score_df > 80).sum(axis=1)
    concurrent_high = (n_high >= 2).sum()
    concurrent_pct = concurrent_high / len(score_df) * 100

    readiness_lines = ['# OSI Readiness Report', '',
                       '## 1. Are component scores stable?', '']
    stable = True
    for col, st in stability.items():
        flag = 'STABLE' if st['mean_rolling_std'] < 15 else 'VOLATILE'
        if flag == 'VOLATILE':
            stable = False
        readiness_lines.append(
            f'- {col}: mean rolling std = {st["mean_rolling_std"]:.2f} → {flag}'
        )
    readiness_lines.append('')
    stable_label = 'STABLE' if stable else 'PARTIALLY STABLE'
    readiness_lines.append(f'**Overall: {stable_label}**')
    readiness_lines.append('')
    readiness_lines.append('## 2. Which risk dominates this truck?')
    readiness_lines.append('')
    for i, (name, val) in enumerate(ranked, 1):
        readiness_lines.append(f'{i}. {name}: {val:.2f}')
    readiness_lines.append('')
    readiness_lines.append(
        f'**Dominant risk: {dominant} (mean = {dominant_val:.2f}/100)**'
    )
    readiness_lines.append('')
    readiness_lines.append('## 3. How frequently do high-risk events occur?')
    readiness_lines.append('')
    readiness_lines.append('| Component | High-risk events | Frequency (events/hour) |')
    readiness_lines.append('|-----------|-----------------|------------------------|')
    for col in score_cols:
        rc = high_risk_counts[col]
        readiness_lines.append(
            f'| {col} | {rc["count"]:,} | {rc["freq_per_hour"]:.1f} |'
        )
    readiness_lines.append('')
    readiness_lines.append(f'**Concurrent high-risk (>=2 components): '
                           f'{concurrent_high:,} rows ({concurrent_pct:.2f}%)**')
    readiness_lines.append('')
    readiness_lines.append('## 4. Is the dataset ready for final OSI construction?')
    readiness_lines.append('')

    checks = []
    checks.append(('Risk scores computed', len(score_cols) >= 7))
    checks.append(('Anomaly scores computed', 'Anomaly_Score' in comp_df.columns))
    checks.append(('No NaN in risk scores', score_df.isna().sum().sum() == 0))
    checks.append(('Scores in [0, 100] range',
                   all(score_df.min() >= 0) and all(score_df.max() <= 101)))
    checks.append(('Sufficient data rows', len(comp_df) > 10000))
    checks.append(('Component stability', stable))
    checks.append(('Normalised features available',
                   any(c.endswith('_minmax') for c in comp_df.columns)))

    all_pass = all(v for _, v in checks)
    for label, ok in checks:
        readiness_lines.append(f'- [{"x" if ok else " "}] {label}')
    readiness_lines.append('')
    if all_pass:
        readiness_lines.append('**YES — The dataset is ready for final OSI construction.**')
    else:
        readiness_lines.append(
            '**PARTIAL — Review flagged items before OSI construction.**'
        )
    readiness_lines.append('')
    readiness_lines.append('---')
    readiness_lines.append('*Report generated by OSI Phase 3 pipeline*')

    Path(OUT_DIR / 'osi_readiness.md').write_text('\n'.join(readiness_lines))
    print(f'    -> {OUT_DIR}/osi_readiness.md')
    del comp_df, score_df

    # ------------------------------------------------------------------
    # Summary output
    # ------------------------------------------------------------------
    print('\n' + '=' * 60)
    print('OSI Phase 3 complete.')
    print('=' * 60)
    print()

    comp_size_mb = os.path.getsize(COMPONENTS_PATH) / 1e6
    print(f'  Component scores:     {len(score_cols)}')
    print(f'  Component parquet:    {comp_size_mb:.1f} MB')
    print(f'  Rows:                 {total_rows:,}')
    print()

    # Anomaly stats
    print(f'  Anomaly detection:')
    print(f'    - Detectors: IsolationForest, LOF, OneClassSVM')
    print(f'    - Features:  {len(DERIVED_FEATURES)} derived features')
    print()

    # High-risk event counts
    print(f'  High-risk event counts (>80):')
    for col in score_cols:
        rc = high_risk_counts[col]
        print(f'    {col}: {rc["count"]:,} ({rc["freq_per_hour"]:.1f}/hr)')
    print()
    print(f'  Concurrent high-risk (>=2): {concurrent_high:,} ({concurrent_pct:.2f}%)')
    print()
    print(f'  Readiness for OSI construction: {"YES" if all_pass else "PARTIAL"}')
    print()


if __name__ == '__main__':
    main()
