"""
OSI Phase 1: Build Master Telemetry Dataset
============================================
Loads payload_processed.csv + tpms_processed.csv,
standardises timestamps, synchronises, cleans, and
produces telemetry_master.parquet + exploratory figures.
"""

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path('D:/Tyre_Classification')
PAYLOAD_PATH = ROOT / 'datasets/sensors_data/payload_processed.csv'
TPMS_PATH = ROOT / 'datasets/sensors_data/tpms_processed.csv'
OUT_DIR = ROOT / 'outputs/osi_phase1'
FIG_DIR = OUT_DIR / 'figures'
TELEMETRY_DIR = ROOT / 'datasets/telemetry'

for d in [OUT_DIR, FIG_DIR, TELEMETRY_DIR]:
    d.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 100_000

# Columns to keep in master
PAYLOAD_COLS = [
    'CDLTruckPayload', 'CDLGroundSpeed', 'J1939WheelBasedVehicleSpeed',
    'J1939EngineRPM', 'SBAccelerationX', 'SBAccelerationY', 'SBAccelerationZ',
    'J1939BrakePedalPosition', 'CDLAutomaticBrakeApplication',
    'SBPitchDegLpf', 'SBRollDegLpf', 'GPSLatitude', 'GPSLongitude',
]
TPMS_COLS = [
    'TPMSTire0x17Pressure', 'TPMSTire0x17Temperature',
    'TPMSTire0x19Pressure', 'TPMSTire0x19Temperature',
    'TPMSTire0x27Pressure', 'TPMSTire0x29Temperature',
]


# ======================================================================
# STEP 1 & 2: Load & standardise timestamps
# ======================================================================
def load_payload_chunked(path: Path) -> pd.DataFrame:
    """Load payload CSV in chunks, standardise timestamp."""
    chunks = []
    for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE,
                             parse_dates=['sample_timestamp']):
        chunk.rename(columns={'sample_timestamp': 'timestamp'}, inplace=True)
        # Remove timezone info if present (naive UTC assumed)
        if hasattr(chunk['timestamp'].dt, 'tz') and chunk['timestamp'].dt.tz is not None:
            chunk['timestamp'] = chunk['timestamp'].dt.tz_localize(None)
        chunks.append(chunk)
        sys.stdout.write(f'\r  Payload: loaded {len(chunks) * CHUNK_SIZE:>8,} rows ...')
    sys.stdout.write('\n')
    return pd.concat(chunks, ignore_index=True)


def load_tpms_chunked(path: Path) -> pd.DataFrame:
    """Load TPMS CSV in chunks, standardise timestamp (_col3)."""
    chunks = []
    for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE,
                             parse_dates=['_col3']):
        chunk.rename(columns={'_col3': 'timestamp'}, inplace=True)
        # Strip milliseconds for 1 Hz alignment
        chunk['timestamp'] = chunk['timestamp'].dt.floor('s')
        # Remove timezone
        if hasattr(chunk['timestamp'].dt, 'tz') and chunk['timestamp'].dt.tz is not None:
            chunk['timestamp'] = chunk['timestamp'].dt.tz_localize(None)
        chunks.append(chunk)
        sys.stdout.write(f'\r  TPMS:     loaded {len(chunks) * CHUNK_SIZE:>8,} rows ...')
    sys.stdout.write('\n')
    return pd.concat(chunks, ignore_index=True)


# ======================================================================
# STEP 5 helpers: cleaning
# ======================================================================
def remove_dup_timestamps(df: pd.DataFrame, label: str) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=['timestamp'], keep='first')
    after = len(df)
    dup_count = before - after
    if dup_count:
        print(f'  [{label}] Removed {dup_count} duplicate timestamps '
              f'({dup_count / before * 100:.2f}%)')
    return df


def flag_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Add _missing flag columns for every sensor column."""
    sensor_cols = [c for c in df.columns if c not in ('timestamp', 'device', 'asset',
                                                       'GPSLatitude', 'GPSLongitude')]
    for col in sensor_cols:
        df[f'{col}_missing'] = df[col].isna().astype(int)
    return df


# ======================================================================
# MAIN
# ======================================================================
def main():
    print('=' * 60)
    print('OSI Phase 1 — Build Master Telemetry Dataset')
    print('=' * 60)

    # ------------------------------------------------------------------
    # 1. LOAD
    # ------------------------------------------------------------------
    print('\n[1] Loading payload ...')
    payload = load_payload_chunked(PAYLOAD_PATH)
    print(f'    Payload rows: {len(payload):,}')

    print('\n[2] Loading TPMS ...')
    tpms = load_tpms_chunked(TPMS_PATH)
    print(f'    TPMS rows:     {len(tpms):,}')

    # ------------------------------------------------------------------
    # Timestamp Audit
    # ------------------------------------------------------------------
    print('\n[3] Timestamp audit ...')
    audit_lines = []
    audit_lines.append('# Timestamp Audit Report')
    audit_lines.append('')
    audit_lines.append('## Payload (`sample_timestamp`)')
    audit_lines.append(f'- Min: {payload["timestamp"].min()}')
    audit_lines.append(f'- Max: {payload["timestamp"].max()}')
    audit_lines.append(f'- Count: {len(payload):,}')
    payload_tz = payload['timestamp'].dt.tz
    audit_lines.append(f'- Timezone: {payload_tz}')
    payload_dup = payload['timestamp'].duplicated().sum()
    audit_lines.append(f'- Duplicate timestamps: {payload_dup:,}')
    payload_ordered = payload['timestamp'].is_monotonic_increasing
    audit_lines.append(f'- Monotonically increasing: {payload_ordered}')
    payload_null_ts = payload['timestamp'].isna().sum()
    audit_lines.append(f'- Null timestamps: {payload_null_ts:,}')
    audit_lines.append('')

    payload_timediff = payload['timestamp'].sort_values().diff().dropna()
    payload_gaps_gt1 = (payload_timediff > '1s').sum()
    payload_max_gap = payload_timediff.max()
    audit_lines.append(f'- Gaps > 1 second: {payload_gaps_gt1:,}')
    audit_lines.append(f'- Max gap: {payload_max_gap}')
    audit_lines.append('')

    audit_lines.append('## TPMS (`_col3`)')
    audit_lines.append(f'- Min: {tpms["timestamp"].min()}')
    audit_lines.append(f'- Max: {tpms["timestamp"].max()}')
    audit_lines.append(f'- Count: {len(tpms):,}')
    tpms_tz = tpms['timestamp'].dt.tz
    audit_lines.append(f'- Timezone: {tpms_tz}')
    tpms_dup = tpms['timestamp'].duplicated().sum()
    audit_lines.append(f'- Duplicate timestamps: {tpms_dup:,}')
    tpms_ordered = tpms['timestamp'].is_monotonic_increasing
    audit_lines.append(f'- Monotonically increasing: {tpms_ordered}')
    tpms_null_ts = tpms['timestamp'].isna().sum()
    audit_lines.append(f'- Null timestamps: {tpms_null_ts:,}')
    audit_lines.append('')

    tpms_timediff = tpms['timestamp'].sort_values().diff().dropna()
    tpms_gaps_gt1 = (tpms_timediff > '1s').sum()
    tpms_max_gap = tpms_timediff.max()
    audit_lines.append(f'- Gaps > 1 second: {tpms_gaps_gt1:,}')
    audit_lines.append(f'- Max gap: {tpms_max_gap}')
    audit_lines.append('')

    audit_lines.append('## Comparison')
    common_ts = set(payload['timestamp'].unique()) & set(tpms['timestamp'].unique())
    audit_lines.append(f'- Common unique timestamps: {len(common_ts):,}')
    only_payload = set(payload['timestamp'].unique()) - set(tpms['timestamp'].unique())
    only_tpms = set(tpms['timestamp'].unique()) - set(payload['timestamp'].unique())
    audit_lines.append(f'- Only in payload: {len(only_payload):,}')
    audit_lines.append(f'- Only in TPMS:     {len(only_tpms):,}')
    audit_lines.append('')

    Path(OUT_DIR / 'timestamp_audit.md').write_text('\n'.join(audit_lines))
    print('    -> outputs/osi_phase1/timestamp_audit.md')

    # ------------------------------------------------------------------
    # 4. SYNCHRONISE / MERGE
    # ------------------------------------------------------------------
    print('\n[4] Synchronising datasets ...')

    # Sort both by timestamp (required for merge_asof)
    payload = payload.sort_values('timestamp').reset_index(drop=True)
    tpms = tpms.sort_values('timestamp').reset_index(drop=True)

    # Exact inner join first
    payload_exact = payload.drop_duplicates(subset=['timestamp'])
    tpms_exact = tpms.drop_duplicates(subset=['timestamp'])

    merged_exact = pd.merge(payload_exact, tpms_exact, on='timestamp', how='inner',
                            suffixes=('', '_tpms'))
    # Drop redundant duplicated columns from TPMS side
    for col in ['device_tpms', 'asset_tpms', 'GPSLatitude_tpms', 'GPSLongitude_tpms']:
        if col in merged_exact.columns:
            merged_exact.drop(columns=[col], inplace=True)

    exact_count = len(merged_exact)
    print(f'    Exact matches: {exact_count:,}')

    # Unmatched payload timestamps
    matched_ts = set(merged_exact['timestamp'])
    payload_unmatched = payload_exact[~payload_exact['timestamp'].isin(matched_ts)].copy()
    tpms_unmatched = tpms_exact[~tpms_exact['timestamp'].isin(matched_ts)].copy()

    # Nearest-neighbour join (±2 seconds) for unmatched rows
    if len(payload_unmatched) > 0 and len(tpms_unmatched) > 0:
        merged_asof = pd.merge_asof(
            payload_unmatched.sort_values('timestamp'),
            tpms_unmatched.sort_values('timestamp'),
            on='timestamp',
            direction='nearest',
            tolerance=pd.Timedelta('2s'),
            suffixes=('', '_tpms'),
        )
        # Keep only rows where a match was actually found
        merged_asof = merged_asof.dropna(subset=['TPMSTire0x17Pressure',
                                                  'TPMSTire0x17Temperature',
                                                  'TPMSTire0x19Pressure',
                                                  'TPMSTire0x19Temperature',
                                                  'TPMSTire0x27Pressure',
                                                  'TPMSTire0x29Temperature'], how='all')
        for col in ['device_tpms', 'asset_tpms', 'GPSLatitude_tpms', 'GPSLongitude_tpms']:
            if col in merged_asof.columns:
                merged_asof.drop(columns=[col], inplace=True)
    else:
        merged_asof = pd.DataFrame()

    asof_count = len(merged_asof)
    print(f'    Nearest-neighbour matches: {asof_count:,}')

    # Final merged dataset — restrict columns immediately to save memory
    keep_cols = ['timestamp'] + PAYLOAD_COLS + TPMS_COLS
    merged_exact = merged_exact[[c for c in keep_cols if c in merged_exact.columns]]
    if len(merged_asof) > 0:
        merged_asof = merged_asof[[c for c in keep_cols if c in merged_asof.columns]]
    merged = pd.concat([merged_exact, merged_asof], ignore_index=True)
    merged = merged.sort_values('timestamp').reset_index(drop=True)

    # Remove timestamp duplicates from the merge
    merged = remove_dup_timestamps(merged, 'Merged')

    # merge_report
    total_payload = len(payload_exact)
    total_tpms = len(tpms_exact)
    merge_lines = []
    merge_lines.append('# Merge Report')
    merge_lines.append('')
    merge_lines.append('## Dataset sizes before merge')
    merge_lines.append(f'- Payload rows (deduplicated): {total_payload:,}')
    merge_lines.append(f'- TPMS rows (deduplicated):     {total_tpms:,}')
    merge_lines.append('')
    merge_lines.append('## Merge results')
    merge_lines.append(f'- Exact timestamp matches:  {exact_count:,}')
    merge_lines.append(f'- Nearest-neighbour matches: {asof_count:,}')
    merge_lines.append(f'- Total merged rows:         {len(merged):,}')
    merge_lines.append(f'- Payload unmatched (lost):  {total_payload - exact_count:,}')
    merge_lines.append(f'- TPMS unmatched (lost):     {total_tpms - exact_count:,}')
    merge_lines.append('')

    # Timestamp drift statistics
    if asof_count > 0:
        drift = merged_exact['timestamp'].iloc[:min(asof_count, len(merged_exact))]
        merge_lines.append('## Timestamp Drift (exact vs nearest-neighbour)')
        merge_lines.append(f'- Nearest-neighbour tolerance: ±2 seconds')
        asof_drift = merged_asof['timestamp'] - merged_asof['timestamp']  # zero drift
        merge_lines.append('  (Drift per row not meaningful for asof; see gap analysis below)')
        merge_lines.append('')

    # Missing percentages after merge
    sensor_cols = PAYLOAD_COLS + TPMS_COLS
    merge_lines.append('## Missing Percentages After Merge')
    for col in sensor_cols:
        if col in merged.columns:
            pct = merged[col].isna().mean() * 100
            merge_lines.append(f'- {col}: {pct:.2f}%')
    merge_lines.append('')

    merge_lines.append('## Temporal Coverage')
    merge_lines.append(f'- Start: {merged["timestamp"].min()}')
    merge_lines.append(f'- End:   {merged["timestamp"].max()}')
    merge_lines.append(f'- Duration: {merged["timestamp"].max() - merged["timestamp"].min()}')
    merge_lines.append('')
    time_diff = merged['timestamp'].diff().dropna()
    gaps_gt1 = (time_diff > '1s').sum()
    merge_lines.append(f'- Gaps > 1 second: {gaps_gt1:,}')
    if gaps_gt1 > 0:
        merge_lines.append(f'- Largest gap: {time_diff.max()}')

    Path(OUT_DIR / 'merge_report.md').write_text('\n'.join(merge_lines))
    print('    -> outputs/osi_phase1/merge_report.md')

    # Free memory before master creation
    del payload, tpms, payload_exact, tpms_exact
    del merged_exact, merged_asof, merge_lines

    # ------------------------------------------------------------------
    # 5. CREATE MASTER DATASET
    # ------------------------------------------------------------------
    print('\n[5] Creating master dataset ...')

    master = merged
    master = remove_dup_timestamps(master, 'Master')
    master = flag_missing(master)
    master = master.reset_index(drop=True)

    # Write parquet
    parquet_path = TELEMETRY_DIR / 'telemetry_master.parquet'
    master.to_parquet(parquet_path, index=False)
    print(f'    -> {parquet_path} ({os.path.getsize(parquet_path) / 1e6:.1f} MB)')

    # Write sample CSV (first 5000 rows)
    sample_path = TELEMETRY_DIR / 'telemetry_master_sample.csv'
    master.head(5000).to_csv(sample_path, index=False)
    print(f'    -> {sample_path}')

    # ------------------------------------------------------------------
    # 6. DATA QUALITY REPORT
    # ------------------------------------------------------------------
    print('\n[6] Generating data quality report ...')

    dq_lines = []
    dq_lines.append('# Data Quality Report — After Merge')
    dq_lines.append('')
    dq_lines.append(f'- Total rows: {len(master):,}')
    dq_lines.append(f'- Total columns: {len(master.columns)}')
    dq_lines.append(f'- Time range: {master["timestamp"].min()} to {master["timestamp"].max()}')
    dq_lines.append(f'- Duration: {master["timestamp"].max() - master["timestamp"].min()}')
    dq_lines.append('')

    dq_lines.append('## Column Completeness (non-null %)')
    for col in master.columns:
        if col.endswith('_missing') or col == 'timestamp':
            continue
        pct = (1 - master[col].isna().mean()) * 100
        dq_lines.append(f'- {col}: {pct:.2f}%')

    dq_lines.append('')
    dq_lines.append('## Missing Flags (count)')
    flag_cols = [c for c in master.columns if c.endswith('_missing')]
    for col in flag_cols:
        sensor = col.replace('_missing', '')
        count = master[col].sum()
        pct = count / len(master) * 100
        dq_lines.append(f'- {sensor}: {int(count):,} missing ({pct:.2f}%)')

    dq_lines.append('')
    dq_lines.append('## Duplicate Rows')
    dq_lines.append(f'- Duplicate timestamps after cleaning: {master["timestamp"].duplicated().sum()}')
    dq_lines.append('')

    dq_lines.append('## Data Types')
    for col, dtype in master.dtypes.items():
        dq_lines.append(f'- {col}: {dtype}')

    Path(OUT_DIR / 'data_quality_after_merge.md').write_text('\n'.join(dq_lines))
    print('    -> outputs/osi_phase1/data_quality_after_merge.md')

    # ------------------------------------------------------------------
    # 7. EXPLORATORY FIGURES
    # ------------------------------------------------------------------
    print('\n[7] Generating exploratory figures ...')
    sns.set_style('darkgrid')
    figsize = (14, 5)

    master_plot = master.copy()

    # Pressure timeseries
    pressure_cols = [c for c in TPMS_COLS if 'Pressure' in c and c in master_plot.columns]
    if pressure_cols:
        fig, ax = plt.subplots(figsize=figsize)
        for col in pressure_cols:
            ax.plot(master_plot['timestamp'], master_plot[col], label=col, linewidth=0.5)
        ax.set_title('TPMS Pressure Timeseries')
        ax.set_ylabel('Pressure')
        ax.legend(loc='best')
        fig.tight_layout()
        fig.savefig(FIG_DIR / 'pressure_timeseries.png', dpi=150)
        plt.close(fig)
        print('    -> pressure_timeseries.png')

    # Temperature timeseries
    temp_cols = [c for c in TPMS_COLS if 'Temperature' in c and c in master_plot.columns]
    if temp_cols:
        fig, ax = plt.subplots(figsize=figsize)
        for col in temp_cols:
            ax.plot(master_plot['timestamp'], master_plot[col], label=col, linewidth=0.5)
        ax.set_title('TPMS Temperature Timeseries')
        ax.set_ylabel('Temperature')
        ax.legend(loc='best')
        fig.tight_layout()
        fig.savefig(FIG_DIR / 'temperature_timeseries.png', dpi=150)
        plt.close(fig)
        print('    -> temperature_timeseries.png')

    # Payload timeseries
    if 'CDLTruckPayload' in master_plot.columns:
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(master_plot['timestamp'], master_plot['CDLTruckPayload'], linewidth=0.5, color='green')
        ax.set_title('Payload Timeseries')
        ax.set_ylabel('Payload')
        fig.tight_layout()
        fig.savefig(FIG_DIR / 'payload_timeseries.png', dpi=150)
        plt.close(fig)
        print('    -> payload_timeseries.png')

    # Speed timeseries
    speed_cols = [c for c in ['CDLGroundSpeed', 'J1939WheelBasedVehicleSpeed']
                  if c in master_plot.columns]
    if speed_cols:
        fig, ax = plt.subplots(figsize=figsize)
        for col in speed_cols:
            ax.plot(master_plot['timestamp'], master_plot[col], label=col, linewidth=0.5)
        ax.set_title('Speed Timeseries')
        ax.set_ylabel('Speed')
        ax.legend(loc='best')
        fig.tight_layout()
        fig.savefig(FIG_DIR / 'speed_timeseries.png', dpi=150)
        plt.close(fig)
        print('    -> speed_timeseries.png')

    # Acceleration timeseries
    accel_cols = [c for c in ['SBAccelerationX', 'SBAccelerationY', 'SBAccelerationZ']
                  if c in master_plot.columns]
    if accel_cols:
        fig, ax = plt.subplots(figsize=figsize)
        for col in accel_cols:
            ax.plot(master_plot['timestamp'], master_plot[col], label=col, linewidth=0.5)
        ax.set_title('SB Acceleration Timeseries')
        ax.set_ylabel('Acceleration (g)')
        ax.legend(loc='best')
        fig.tight_layout()
        fig.savefig(FIG_DIR / 'acceleration_timeseries.png', dpi=150)
        plt.close(fig)
        print('    -> acceleration_timeseries.png')

    # Correlation heatmap
    numeric_cols = master_plot.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if not c.endswith('_missing')]
    if len(numeric_cols) > 1:
        corr = master_plot[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(max(10, len(numeric_cols) * 0.8),
                                        max(8, len(numeric_cols) * 0.7)))
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                    center=0, vmin=-1, vmax=1, square=True,
                    cbar_kws={'shrink': 0.8}, ax=ax)
        ax.set_title('Sensor Correlation Heatmap')
        fig.tight_layout()
        fig.savefig(FIG_DIR / 'correlation_heatmap.png', dpi=150)
        plt.close(fig)
        print('    -> correlation_heatmap.png')

    # ------------------------------------------------------------------
    # 8. EXECUTIVE SUMMARY
    # ------------------------------------------------------------------
    print('\n[8] Generating executive summary ...')

    total_rows = len(master)
    usable_pct = (1 - master[PAYLOAD_COLS + TPMS_COLS].isna().all(axis=1).mean()) * 100

    # Most complete columns
    completeness = {}
    for col in PAYLOAD_COLS + TPMS_COLS:
        if col in master.columns:
            completeness[col] = (1 - master[col].isna().mean()) * 100
    sorted_complete = sorted(completeness.items(), key=lambda x: -x[1])

    gap_threshold = pd.Timedelta('5s')
    gaps = master['timestamp'].diff()
    long_gaps = (gaps > gap_threshold).sum()

    summary_lines = []
    summary_lines.append('# OSI Phase 1 — Executive Summary')
    summary_lines.append('')
    summary_lines.append('## 1. Was synchronization successful?')
    if exact_count > 0:
        success_pct = exact_count / min(total_payload, total_tpms) * 100
        summary_lines.append(f'- Yes. Exact timestamp matches: {exact_count:,} '
                             f'({success_pct:.1f}% of the smaller dataset).')
        summary_lines.append(f'- Nearest-neighbour (asof ±2s) recovered {asof_count:,} additional rows.')
        summary_lines.append(f'- Total merged rows: {total_rows:,}')
    else:
        summary_lines.append('- No exact matches found. Synchronization failed.')
    summary_lines.append('')

    summary_lines.append('## 2. How much data is usable?')
    summary_lines.append(f'- {total_rows:,} total rows in merged dataset.')
    summary_lines.append(f'- {usable_pct:.1f}% of rows have at least one non-null sensor value.')
    summary_lines.append(f'- {total_rows - master[PAYLOAD_COLS + TPMS_COLS].isna().all(axis=1).sum():,} rows '
                         f'have at least one sensor reading.')
    summary_lines.append('')

    summary_lines.append('## 3. Are there long gaps?')
    summary_lines.append(f'- Gaps > 5 seconds: {long_gaps:,}')
    if long_gaps > 0:
        max_gap_val = master['timestamp'].diff().max()
        summary_lines.append(f'- Largest gap: {max_gap_val}')
        gap_dist = master['timestamp'].diff().dropna()
        summary_lines.append(f'- Median gap: {gap_dist.median()}')
    else:
        summary_lines.append('- No significant gaps detected.')
    summary_lines.append('')

    summary_lines.append('## 4. Which columns are most complete?')
    for i, (col, pct) in enumerate(sorted_complete[:5]):
        summary_lines.append(f'  {i + 1}. {col}: {pct:.1f}% complete')
    summary_lines.append('')
    summary_lines.append('  Least complete:')
    for col, pct in sorted_complete[-3:]:
        summary_lines.append(f'  - {col}: {pct:.1f}% complete')
    summary_lines.append('')

    summary_lines.append('## 5. Is the data ready for feature engineering?')
    all_any = all(p > 0 for _, p in sorted_complete[:3]) if len(sorted_complete) >= 3 else False
    if total_rows > 1000 and usable_pct > 50:
        summary_lines.append('- Yes, the merged dataset is ready for feature engineering.')
    else:
        summary_lines.append('- Partial readiness. See column completeness notes below.')

    summary_lines.append(f'- {total_rows:,} rows × {len(PAYLOAD_COLS + TPMS_COLS)} sensor columns.')
    summary_lines.append(f'- {len(completeness)} sensor columns available for feature engineering.')
    summary_lines.append(f'- Data spans {master["timestamp"].max() - master["timestamp"].min()}.')
    summary_lines.append('')
    summary_lines.append('---')
    summary_lines.append(f'*Report generated by OSI Phase 1 pipeline*')

    Path(OUT_DIR / 'executive_summary.md').write_text('\n'.join(summary_lines))
    print('    -> outputs/osi_phase1/executive_summary.md')

    print('\n' + '=' * 60)
    print('OSI Phase 1 complete.')
    print('=' * 60)


if __name__ == '__main__':
    main()
