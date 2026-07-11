"""
OSI Phase 2: Feature Engineering
=================================
Loads telemetry_master.parquet and engineers features for tyre
severity analysis.  Processes data in daily chunks to manage
memory on constrained hardware.

Saves telemetry_features.parquet and generates quality / relevance
reports.

DO NOT: build severity score, train models, or predict RUL.
"""

import os
import sys
import warnings
import math
from pathlib import Path
from typing import List, Optional, Dict

_PROJ_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from src.severity_sensor.rolling_features import (
    compute_rolling_stats,
    compute_rolling_gradient,
    compute_zscore,
    compute_rms,
    compute_crest_factor,
    WINDOWS,
)
from src.severity_sensor.event_detection import (
    flag_underinflation,
    flag_overinflation,
    flag_overtemperature,
    flag_overload,
    flag_hard_braking,
    flag_impact,
    flag_high_vibration,
    flag_idle,
    flag_high_speed,
    flag_slope_event,
    flag_rough_terrain,
    flag_pressure_anomaly,
    detect_stopped,
    compute_cumulative_counters,
    DEFAULT_THRESHOLDS,
)

warnings.filterwarnings('ignore')

ROOT = Path('D:/Tyre_Classification')
MASTER_PATH = ROOT / 'datasets/telemetry/telemetry_master.parquet'
OUT_DIR = ROOT / 'outputs/osi_phase2'
FIG_DIR = OUT_DIR / 'figures'
DIST_DIR = OUT_DIR / 'feature_distributions'
TELEMETRY_DIR = ROOT / 'datasets/telemetry'
TEMP_DIR = ROOT / 'datasets/telemetry/tmp_features'

for d in [OUT_DIR, FIG_DIR, DIST_DIR, TELEMETRY_DIR, TEMP_DIR]:
    d.mkdir(parents=True, exist_ok=True)

CHUNK_ROWS = 432000  # 5 days at 1 Hz (fewer chunks = faster)
OVERLAP = 3600        # 1 hour overlap for rolling windows

PRESSURE_COLS = [
    'TPMSTire0x17Pressure', 'TPMSTire0x19Pressure', 'TPMSTire0x27Pressure',
]
TEMP_COLS = [
    'TPMSTire0x17Temperature', 'TPMSTire0x19Temperature',
    'TPMSTire0x29Temperature',
]
ACCEL_COLS = ['SBAccelerationX', 'SBAccelerationY', 'SBAccelerationZ']
ROLLING_COLS = [
    'CDLTruckPayload', 'J1939WheelBasedVehicleSpeed', 'J1939EngineRPM',
    'SBAccelerationX', 'SBAccelerationY', 'SBAccelerationZ',
    'J1939BrakePedalPosition', 'SBPitchDegLpf', 'SBRollDegLpf',
    'TPMSTire0x17Pressure', 'TPMSTire0x17Temperature',
    'TPMSTire0x19Pressure', 'TPMSTire0x19Temperature',
    'TPMSTire0x27Pressure', 'TPMSTire0x29Temperature',
]


def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2 +
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) *
         np.sin(dlon / 2) ** 2)
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def features_for_chunk(
    context: pd.DataFrame,
    chunk_start_idx: int,
    chunk_end_idx: int,
) -> pd.DataFrame:
    """Compute all features for ``context`` rows and return only the
    rows in [chunk_start_idx, chunk_end_idx) of the context."""
    n = len(context)

    # --- Rolling features ---
    result = compute_rolling_stats(context, ROLLING_COLS)

    # --- Pressure features ---
    pcol = [c for c in PRESSURE_COLS if c in context.columns]
    if pcol:
        pvals = context[pcol]
        result['pressure_mean'] = pvals.mean(axis=1)
        result['pressure_std'] = pvals.std(axis=1)
        result['pressure_min'] = pvals.min(axis=1)
        result['pressure_max'] = pvals.max(axis=1)

        for i, (a, b) in enumerate([
            ('TPMSTire0x17Pressure', 'TPMSTire0x19Pressure'),
            ('TPMSTire0x17Pressure', 'TPMSTire0x27Pressure'),
            ('TPMSTire0x19Pressure', 'TPMSTire0x27Pressure'),
        ]):
            if a in context.columns and b in context.columns:
                result[f'pressure_diff_{i+1}'] = (context[a] - context[b]).abs()
        dc = [c for c in result.columns if c.startswith('pressure_diff_')]
        result['pressure_difference_between_tyres'] = result[dc].max(axis=1) if dc else 0.0

        for col in pcol:
            result[f'{col}_drop_rate'] = compute_rolling_gradient(
                context[col], window=3600, label=f'{col}_drop_rate'
            )
        drc = [c for c in result.columns if c.endswith('_drop_rate')]
        result['pressure_drop_rate'] = result[drc].min(axis=1) if drc else 0.0

        uf = flag_underinflation(pvals)
        of_ = flag_overinflation(pvals)
        result['underinflation_flag'] = uf['underinflation_flag']
        result['overinflation_flag'] = of_['overinflation_flag']

        zs = []
        for col in pcol:
            zs.append(compute_zscore(context[col], window=3600, label=f'{col}_zscore'))
        if zs:
            zdf = pd.concat(zs, axis=1)
            result['pressure_zscore'] = zdf.mean(axis=1)
            pa = flag_pressure_anomaly(zdf)
            result['pressure_anomaly_flag'] = pa['pressure_anomaly_flag']
    else:
        for c in ['pressure_mean', 'pressure_std', 'pressure_min', 'pressure_max',
                   'pressure_difference_between_tyres', 'pressure_drop_rate',
                   'underinflation_flag', 'overinflation_flag', 'pressure_zscore',
                   'pressure_anomaly_flag']:
            result[c] = np.nan

    # --- Temperature features ---
    tcol = [c for c in TEMP_COLS if c in context.columns]
    if tcol:
        tvals = context[tcol]
        result['temperature_mean'] = tvals.mean(axis=1)
        result['temperature_std'] = tvals.std(axis=1)
        result['temperature_max'] = tvals.max(axis=1)

        for i, (a, b) in enumerate([
            ('TPMSTire0x17Temperature', 'TPMSTire0x19Temperature'),
            ('TPMSTire0x17Temperature', 'TPMSTire0x29Temperature'),
            ('TPMSTire0x19Temperature', 'TPMSTire0x29Temperature'),
        ]):
            if a in context.columns and b in context.columns:
                result[f'temp_diff_{i+1}'] = (context[a] - context[b]).abs()
        td = [c for c in result.columns if c.startswith('temp_diff_')]
        result['temperature_difference_between_tyres'] = result[td].max(axis=1) if td else 0.0

        for col in tcol:
            result[f'{col}_rise_rate'] = compute_rolling_gradient(
                context[col], window=3600, label=f'{col}_rise_rate'
            )
        rr = [c for c in result.columns if c.endswith('_rise_rate')]
        result['temperature_rise_rate'] = result[rr].max(axis=1) if rr else 0.0

        ot = flag_overtemperature(tvals)
        result['overtemperature_flag'] = ot['overtemperature_flag']

        tz = []
        for col in tcol:
            tz.append(compute_zscore(context[col], window=3600, label=f'{col}_zscore'))
        if tz:
            tzdf = pd.concat(tz, axis=1)
            result['temperature_zscore'] = tzdf.mean(axis=1)
    else:
        for c in ['temperature_mean', 'temperature_std', 'temperature_max',
                   'temperature_difference_between_tyres', 'temperature_rise_rate',
                   'overtemperature_flag', 'temperature_zscore']:
            result[c] = np.nan

    # --- Payload features ---
    if 'CDLTruckPayload' in context.columns:
        pl = context['CDLTruckPayload']
        result['payload_mean'] = pl.rolling(3600, min_periods=1).mean()
        result['payload_std'] = pl.rolling(3600, min_periods=1).std(ddof=0)
        result['payload_max'] = pl.rolling(3600, min_periods=1).max()
        result['payload_change_rate'] = compute_rolling_gradient(pl, window=300, label='payload_change_rate')
        max_pl = pl.max()
        result['payload_percent_of_max'] = (pl / max_pl * 100) if max_pl > 0 else 0.0
        ol = flag_overload(pl)
        result['overload_flag'] = ol['overload_flag']
        result['cumulative_payload_exposure'] = pl.fillna(0)
    else:
        for c in ['payload_mean', 'payload_std', 'payload_max', 'payload_change_rate',
                   'payload_percent_of_max', 'overload_flag', 'cumulative_payload_exposure']:
            result[c] = np.nan

    # --- Speed features ---
    speed_col = 'J1939WheelBasedVehicleSpeed'
    if speed_col not in context.columns and 'CDLGroundSpeed' in context.columns:
        speed_col = 'CDLGroundSpeed'
    if speed_col and speed_col in context.columns:
        sp = context[speed_col]
        result['speed_mean'] = sp.rolling(3600, min_periods=1).mean()
        result['speed_std'] = sp.rolling(3600, min_periods=1).std(ddof=0)
        result['speed_max'] = sp.rolling(3600, min_periods=1).max()
        hs = flag_high_speed(sp)
        result['high_speed_flag'] = hs['high_speed_flag']
        id_ = flag_idle(sp)
        result['idle_flag'] = id_['idle_flag']
        result['distance_travelled'] = sp.fillna(0)
    else:
        for c in ['speed_mean', 'speed_std', 'speed_max', 'high_speed_flag',
                   'idle_flag', 'distance_travelled']:
            result[c] = np.nan

    # --- Acceleration features ---
    acl = [c for c in ACCEL_COLS if c in context.columns]
    if len(acl) == 3:
        mag = np.sqrt(context[acl[0]] ** 2 + context[acl[1]] ** 2 + context[acl[2]] ** 2)
        result['acceleration_magnitude'] = mag
        result['rolling_mean_acceleration'] = mag.rolling(60, min_periods=1).mean()
        result['rolling_std_acceleration'] = mag.rolling(60, min_periods=1).std(ddof=0)
        result['peak_acceleration'] = mag.rolling(60, min_periods=1).max()
        imp = flag_impact(mag)
        result['impact_event_flag'] = imp['impact_event_flag']
        result['rms_vibration'] = compute_rms(mag, window=60, label='rms_vibration')
        hv = flag_high_vibration(result['rms_vibration'])
        result['high_vibration_flag'] = hv['high_vibration_flag']
        result['crest_factor'] = compute_crest_factor(mag, result['rms_vibration'], label='crest_factor')
    else:
        for c in ['acceleration_magnitude', 'rolling_mean_acceleration',
                   'rolling_std_acceleration', 'peak_acceleration',
                   'impact_event_flag', 'rms_vibration', 'high_vibration_flag',
                   'crest_factor']:
            result[c] = np.nan

    # --- Braking features ---
    if 'J1939BrakePedalPosition' in context.columns:
        br = context['J1939BrakePedalPosition']
        hb = flag_hard_braking(br)
        result['hard_braking_flag'] = hb['hard_braking_flag']
        result['brake_frequency'] = (br > 0).astype(int).rolling(3600, min_periods=1).sum()
        ba = (br > 0).astype(int)
        result['brake_duration'] = ba.groupby((ba.diff() != 0).cumsum()).cumsum()
        result['cumulative_braking_time'] = ba
    else:
        for c in ['hard_braking_flag', 'brake_frequency', 'brake_duration',
                   'cumulative_braking_time']:
            result[c] = np.nan

    # --- Pitch / Roll ---
    if 'SBPitchDegLpf' in context.columns:
        pitch = context['SBPitchDegLpf']
        result['pitch_std'] = pitch.rolling(300, min_periods=1).std(ddof=0)
        se = flag_slope_event(pitch)
        result['slope_event_flag'] = se['slope_event_flag']
    else:
        result['pitch_std'] = np.nan
        result['slope_event_flag'] = 0

    if 'SBRollDegLpf' in context.columns:
        roll = context['SBRollDegLpf']
        result['roll_std'] = roll.rolling(300, min_periods=1).std(ddof=0)
        rt = flag_rough_terrain(result['roll_std'])
        result['rough_terrain_flag'] = rt['rough_terrain_flag']
    else:
        result['roll_std'] = np.nan
        result['rough_terrain_flag'] = 0

    # --- GPS features ---
    if 'GPSLatitude' in context.columns and 'GPSLongitude' in context.columns:
        lat = context['GPSLatitude'].values
        lon = context['GPSLongitude'].values
        dist = np.zeros(len(lat))
        dist[1:] = haversine(lat[:-1], lon[:-1], lat[1:], lon[1:])
        result['distance_increment'] = dist
        sd = detect_stopped(pd.Series(dist))
        result['stopped_flag'] = sd['stopped_flag']
        if len(lat) > 1:
            time_diff = context['timestamp'].diff().dt.total_seconds().values
            time_diff[0] = 1.0
            result['average_speed_from_gps'] = dist / np.where(time_diff > 0, time_diff, 1.0)
        else:
            result['average_speed_from_gps'] = 0.0
    else:
        result['distance_increment'] = 0.0
        result['stopped_flag'] = 0
        result['average_speed_from_gps'] = 0.0

    # Attach timestamp and slice to the clean chunk region
    result.insert(0, 'timestamp', context['timestamp'].values)
    return result.iloc[chunk_start_idx:chunk_end_idx].copy()


# ======================================================================
# Parquet row-range reader (memory-efficient)
# ======================================================================

def _build_rg_bounds(pf: pq.ParquetFile) -> List[int]:
    """Return exclusive cumulative end indices for each row group."""
    bounds = []
    cum = 0
    for i in range(pf.metadata.num_row_groups):
        cum += pf.metadata.row_group(i).num_rows
        bounds.append(cum)
    return bounds


def _read_rows(pf: pq.ParquetFile, rg_bounds: List[int],
               start: int, length: int,
               columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Read a slice of rows from a ParquetFile without loading the full file.

    Parameters
    ----------
    pf : pq.ParquetFile
    rg_bounds : List[int] — cumulative row-group boundaries.
    start, length : int — row offset and count.

    Returns
    -------
    pd.DataFrame with a reset index.
    """
    if length <= 0:
        return pd.DataFrame()
    end = start + length

    # Find which row groups overlap [start, end)
    first_rg = 0
    while first_rg < len(rg_bounds) and rg_bounds[first_rg] <= start:
        first_rg += 1
    last_rg = first_rg
    while last_rg < len(rg_bounds) and rg_bounds[last_rg] < end:
        last_rg += 1

    # Read relevant row groups (only requested columns)
    read_cols = columns  # None means all columns
    tables = [pf.read_row_group(i, columns=read_cols) for i in range(first_rg, last_rg + 1)]
    if len(tables) == 1:
        table = tables[0]
    else:
        table = pa.concat_tables(tables)

    rg_start = rg_bounds[first_rg - 1] if first_rg > 0 else 0
    offset = start - rg_start
    sliced = table.slice(offset, length)
    df = sliced.to_pandas()
    df.reset_index(drop=True, inplace=True)
    return df


# ======================================================================
# MAIN
# ======================================================================
def main():
    print('=' * 60)
    print('OSI Phase 2 — Feature Engineering (memory-efficient chunked)')
    print('=' * 60)

    # ------------------------------------------------------------------
    # STEP 1: Open master parquet for row-group-level reading
    # ------------------------------------------------------------------
    print('\n[1] Opening master dataset ...')
    pf = pq.ParquetFile(MASTER_PATH)
    rg_bounds = _build_rg_bounds(pf)
    total_rows = rg_bounds[-1] if rg_bounds else 0
    print(f'    Rows: {total_rows:,}, Row groups: {pf.metadata.num_row_groups}, '
          f'Columns: {pf.metadata.num_columns}')

    # Columns to use (skip _missing flags)
    all_cols = pf.schema_arrow.names
    keep_cols = [c for c in all_cols if not c.endswith('_missing')]
    print(f'    Using {len(keep_cols)} columns (dropped {len(all_cols) - len(keep_cols)} _missing flags)')

    # ------------------------------------------------------------------
    # Build chunk descriptors
    # ------------------------------------------------------------------
    chunks = []
    start = 0
    ci = 0
    while start < total_rows:
        chunk_end = min(start + CHUNK_ROWS, total_rows)
        context_start = (start - OVERLAP) if ci > 0 else 0
        context_end = min(chunk_end + OVERLAP, total_rows)
        chunks.append({
            'context_start': context_start,
            'context_end': context_end,
            'chunk_start': start,
            'chunk_end': chunk_end,
        })
        start += CHUNK_ROWS
        ci += 1

    print(f'    Processing {len(chunks)} chunks ...')

    # Running cumulative totals for features that span the full timeline
    cum_dist = 0.0
    cum_payload = 0.0
    cum_braking = 0.0
    cum_dist_gps = 0.0
    cum_counters = {}

    FEATURES_PATH = TELEMETRY_DIR / 'telemetry_features.parquet'

    # ------------------------------------------------------------------
    # Process chunks  (Steps 2-11)
    # ------------------------------------------------------------------
    writer = None
    for ci, ck in enumerate(chunks):
        print(f'\n  Chunk {ci + 1}/{len(chunks)}: '
              f'rows [{ck["chunk_start"]:,}, {ck["chunk_end"]:,})  '
              f'context [{ck["context_start"]:,}, {ck["context_end"]:,})')

        context = _read_rows(pf, rg_bounds, ck['context_start'],
                             ck['context_end'] - ck['context_start'],
                             columns=keep_cols)
        print(f'    Context: {len(context)} rows')

        local_start = ck['chunk_start'] - ck['context_start']
        local_end = ck['chunk_end'] - ck['context_start']

        result = features_for_chunk(context, local_start, local_end)
        n_chunk = len(result)
        print(f'    Features: {result.shape[1]} cols × {n_chunk:,} rows')

        # --- Add cumulative features incrementally ---
        flag_names = ['overload_flag', 'pressure_anomaly_flag', 'overtemperature_flag',
                      'impact_event_flag', 'hard_braking_flag', 'rough_terrain_flag']

        # Cumulative distance from speed (assuming 1 Hz, speed in m/s)
        if 'distance_travelled' in result.columns:
            vals = result['distance_travelled'].fillna(0).values
            result['cumulative_distance'] = np.cumsum(vals) + cum_dist
            cum_dist += vals.sum()
        else:
            result['cumulative_distance'] = cum_dist

        # Cumulative payload exposure
        if 'cumulative_payload_exposure' in result.columns:
            vals = result['cumulative_payload_exposure'].fillna(0).values
            result['cumulative_payload_exposure'] = np.cumsum(vals) + cum_payload
            cum_payload += vals.sum()

        # Cumulative braking time
        if 'cumulative_braking_time' in result.columns:
            vals = result['cumulative_braking_time'].fillna(0).values
            result['cumulative_braking_time'] = np.cumsum(vals) + cum_braking
            cum_braking += vals.sum()

        # Cumulative distance from GPS
        if 'distance_increment' in result.columns:
            vals = result['distance_increment'].fillna(0).values
            result['cumulative_distance_gps'] = np.cumsum(vals) + cum_dist_gps
            cum_dist_gps += vals.sum()

        # Cumulative event counters
        for flag in flag_names:
            if flag in result.columns:
                key = f'num_{flag}'
                vals = result[flag].fillna(0).astype(int).values
                running = cum_counters.get(key, 0)
                result[key] = np.cumsum(vals) + running
                cum_counters[key] = running + vals.sum()

        # --- Write to parquet via persistent writer ---
        table = pa.Table.from_pandas(result, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(str(FEATURES_PATH), table.schema)
        writer.write_table(table)
        del context, result, table

    if writer is not None:
        writer.close()
    pf.close()
    print(f'\n    -> {FEATURES_PATH} ({os.path.getsize(FEATURES_PATH) / 1e6:.1f} MB)')

    # ------------------------------------------------------------------
    # STEP 12b: Sample CSV
    # ------------------------------------------------------------------
    sample_path = TELEMETRY_DIR / 'telemetry_features_sample.csv'
    sample = pd.read_parquet(FEATURES_PATH, engine='pyarrow')
    sample.head(5000).to_csv(sample_path, index=False)
    del sample
    print(f'    -> {sample_path}')

    # ------------------------------------------------------------------
    # Discover feature columns from file schema
    # ------------------------------------------------------------------
    feat_pf = pq.ParquetFile(FEATURES_PATH)
    feature_cols = [c for c in feat_pf.schema_arrow.names if c != 'timestamp']
    total_feat_rows = feat_pf.metadata.num_rows

    groups = {
        'Rolling window stats':
            [c for c in feature_cols if any(f'_{s}_' in c for s in ['mean', 'std', 'min', 'max', 'median'])
             and any(w in c for w in ['1min', '5min', '15min', '1hr'])],
        'Pressure features': [c for c in feature_cols if c.startswith('pressure_') or
                              c in ['underinflation_flag', 'overinflation_flag']],
        'Temperature features': [c for c in feature_cols if c.startswith('temperature_') or
                                  c == 'overtemperature_flag'],
        'Payload features': [c for c in feature_cols if c.startswith('payload_') or
                              c == 'overload_flag'],
        'Speed features': [c for c in feature_cols if c.startswith('speed_') or
                            c in ('high_speed_flag', 'idle_flag', 'distance_travelled', 'cumulative_distance')],
        'Acceleration features': [c for c in feature_cols if c.startswith('acceleration_') or
                                   c.startswith('rolling_') or c.startswith('rms_') or
                                   c in ('peak_acceleration', 'impact_event_flag', 'high_vibration_flag', 'crest_factor')],
        'Braking features': [c for c in feature_cols if c.startswith('brake_') or
                              c.startswith('cumulative_braking') or c == 'hard_braking_flag'],
        'Pitch/Roll features': [c for c in feature_cols if c.startswith('pitch_') or
                                 c.startswith('roll_') or c in ('slope_event_flag', 'rough_terrain_flag')],
        'GPS features': [c for c in feature_cols if c.startswith('distance_') or
                          c.startswith('average_speed') or c == 'stopped_flag' or c == 'cumulative_distance_gps'],
        'Event counters': [c for c in feature_cols if c.startswith('num_')],
    }

    # ------------------------------------------------------------------
    # STEP 13: Feature quality report
    # ------------------------------------------------------------------
    print('\n[13] Generating feature quality report ...')

    # Feature catalog
    cat_lines = ['# Feature Catalog', '',
                 f'Total features: {len(feature_cols)}',
                 f'Total rows: {total_feat_rows:,}', '']
    for gn, gc in groups.items():
        if gc:
            cat_lines.append(f'## {gn} ({len(gc)})')
            for c in gc:
                cat_lines.append(f'  - {c}')
            cat_lines.append('')
    Path(OUT_DIR / 'feature_catalog.md').write_text('\n'.join(cat_lines))
    print('    -> feature_catalog.md')

    # Statistics via chunked reads (never load full feature dataset)
    stat_lines = ['# Feature Statistics', '',
                  f'Total rows: {total_feat_rows:,}',
                  f'Total feature columns: {len(feature_cols)}', '',
                  '## Column Completeness', '']

    # Read one row group of features for statistics (first 100k rows)
    stat_sample = pd.read_parquet(FEATURES_PATH, engine='pyarrow')
    if len(stat_sample) > 100000:
        stat_sample = stat_sample.sample(100000, random_state=42)

    # Completeness
    for col in feature_cols:
        if col in stat_sample.columns:
            non_null = stat_sample[col].notna().sum()
            pct = non_null / len(stat_sample) * 100
            stat_lines.append(f'- {col}: {non_null:,} / {len(stat_sample):,} ({pct:.2f}%)')
    stat_lines.append('')

    # Basic statistics
    stat_lines.append('## Basic Statistics (sample-based)')
    stat_lines.append('')
    for col in feature_cols:
        if col in stat_sample.columns:
            dtype = stat_sample[col].dtype
            if np.issubdtype(dtype, np.number):
                desc = stat_sample[col].describe()
                stat_lines.append(f'### {col} (dtype: {dtype})')
                for k in ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']:
                    stat_lines.append(f'- {k}: {desc[k]:.4f}')
                stat_lines.append('')
            else:
                stat_lines.append(f'### {col} (dtype: {dtype})')
                stat_lines.append('')

    Path(OUT_DIR / 'feature_statistics.md').write_text('\n'.join(stat_lines))
    print('    -> feature_statistics.md')
    del stat_sample

    # Correlation heatmap (from sampled data)
    print('    Computing correlation matrix (sampled) ...')
    corr_sample = pd.read_parquet(FEATURES_PATH, engine='pyarrow')
    if len(corr_sample) > 50000:
        corr_sample = corr_sample.sample(50000, random_state=42)
    numeric_feat = corr_sample[feature_cols].select_dtypes(include=[np.number])
    n_cols = len(numeric_feat.columns)

    if n_cols > 1:
        key_prefixes = ['pressure_', 'temperature_', 'payload_', 'speed_',
                        'acceleration_', 'rms_', 'brake_', 'pitch_', 'roll_',
                        'cumulative_', 'num_', 'idle_', 'high_speed_',
                        'underinflation_', 'overinflation_', 'overload_',
                        'hard_braking_', 'impact_', 'high_vibration_',
                        'slope_', 'rough_terrain_']
        keep = []
        for col in numeric_feat.columns:
            if any(col.startswith(p) for p in key_prefixes) or 'pressure_diff' in col:
                keep.append(col)
        if len(keep) < 2:
            keep = numeric_feat.columns[:50].tolist()
        corr = corr_sample[keep].corr()

        fig, ax = plt.subplots(figsize=(max(20, len(corr.columns) * 0.5),
                                        max(16, len(corr.columns) * 0.45)))
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sns.heatmap(corr, mask=mask, annot=False, cmap='RdBu_r',
                    center=0, vmin=-1, vmax=1, square=True,
                    cbar_kws={'shrink': 0.6}, ax=ax)
        ax.set_title('Feature Correlation Heatmap')
        fig.tight_layout()
        fig.savefig(FIG_DIR / 'correlation_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print('    -> correlation_heatmap.png')

    del corr_sample

    # Feature distributions (from sampled data)
    print('    Generating distribution plots ...')
    dist_sample = pd.read_parquet(FEATURES_PATH, engine='pyarrow')
    if len(dist_sample) > 50000:
        dist_sample = dist_sample.sample(50000, random_state=42)

    key_features = [
        'pressure_mean', 'pressure_std', 'pressure_drop_rate',
        'temperature_mean', 'temperature_max', 'temperature_rise_rate',
        'payload_mean', 'payload_percent_of_max', 'overload_flag',
        'speed_mean', 'speed_max', 'high_speed_flag', 'idle_flag',
        'acceleration_magnitude', 'rms_vibration', 'crest_factor',
        'hard_braking_flag', 'brake_frequency',
        'pitch_std', 'roll_std', 'slope_event_flag', 'rough_terrain_flag',
        'cumulative_distance', 'stopped_flag',
        'average_speed_from_gps',
    ]
    existing_key = [c for c in key_features if c in dist_sample.columns]
    for col in existing_key:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        vals = dist_sample[col].dropna().values
        if len(vals) > 0:
            axes[0].hist(vals, bins=100, edgecolor='none', alpha=0.7)
            axes[0].set_title(f'{col}')
            axes[0].set_ylabel('Frequency')
            axes[1].boxplot(vals, vert=False)
            axes[1].set_title(f'{col}')
            fig.tight_layout()
            fig.savefig(DIST_DIR / f'{col}.png', dpi=100)
            plt.close(fig)
    print(f'    -> {len(existing_key)} distribution plots')
    del dist_sample

    # ------------------------------------------------------------------
    # STEP 14: Severity relevance report
    # ------------------------------------------------------------------
    print('\n[14] Generating severity relevance report ...')

    entries = [
        ('Rolling Window Statistics',
         'Rolling mean/std/min/max/median capture short- and medium-term trends. '
         'Sustained deviations indicate periods of elevated stress.',
         'Operational Severity'),
        ('Pressure Mean / Std / Min / Max',
         'Average pressure levels and variability indicate inflation health.',
         'Load Severity'),
        ('Pressure Difference Between Tyres',
         'Imbalance causes uneven load distribution, increasing wear.',
         'Load Severity'),
        ('Pressure Drop Rate',
         'Rapid pressure loss signals puncture or leak events.',
         'Load Severity'),
        ('Pressure Anomaly Flag',
         'Statistical outliers in pressure that may precede failure.',
         'Load Severity'),
        ('Under/Over Inflation Flags',
         'Direct indicators of incorrect inflation.',
         'Load Severity'),
        ('Temperature Mean / Std / Max',
         'High temperature accelerates rubber degradation.',
         'Thermal Severity'),
        ('Temperature Rise Rate',
         'Rapid increase can precede blowout.',
         'Thermal Severity'),
        ('Temperature Difference Between Tyres',
         'Imbalance suggests uneven loading or mechanical issues.',
         'Thermal Severity'),
        ('Overtemperature Flag',
         'Thermal overload beyond manufacturer limits.',
         'Thermal Severity'),
        ('Payload Mean / Std / Max',
         'Heavier loads increase contact pressure and heat generation.',
         'Load Severity'),
        ('Payload Change Rate',
         'Rapid loading/unloading creates thermal and mechanical shock.',
         'Load Severity'),
        ('Payload Percent of Max / Overload Flag',
         'Near/above rated capacity dramatically increases wear rate.',
         'Load Severity'),
        ('Cumulative Payload Exposure',
         'Total load carried — directly proportional to total wear.',
         'Load Severity'),
        ('Speed Mean / Std / Max',
         'Higher speeds increase tread temperature and wear rate.',
         'Operational Severity'),
        ('High Speed Flag',
         'Sustained high-speed operation increases heat buildup.',
         'Operational Severity'),
        ('Idle Flag',
         'Stationary dwell time periods.',
         'Operational Severity'),
        ('Distance Travelled / Cumulative Distance',
         'Fundamental usage measure — wear accumulates with distance.',
         'Operational Severity'),
        ('Acceleration Magnitude / Peak Acceleration',
         'Lateral and vertical forces deform the tyre and accelerate wear.',
         'Vibration Severity'),
        ('RMS Vibration',
         'Vibration energy indicating rough roads or imbalance.',
         'Vibration Severity'),
        ('Crest Factor',
         'Impulsive events (potholes, kerb strikes) causing localised damage.',
         'Vibration Severity'),
        ('Impact Event Flag',
         'Shock loads causing carcass or sidewall damage.',
         'Vibration Severity'),
        ('High Vibration Flag',
         'Sustained elevated vibration indicating poor road conditions.',
         'Vibration Severity'),
        ('Brake Frequency / Hard Braking Flag',
         'Frequent/aggressive braking generates heat and flat-spot wear.',
         'Thermal Severity'),
        ('Brake Duration / Cumulative Braking Time',
         'Prolonged braking heats brake assembly and adjacent tyre areas.',
         'Thermal Severity'),
        ('Pitch Std / Slope Event Flag',
         'Slope operation shifts load distribution; climbing heats drive axle.',
         'Load Severity'),
        ('Roll Std / Rough Terrain Flag',
         'Uneven terrain causes dynamic load variation and irregular wear.',
         'Vibration Severity'),
        ('GPS Distance / Stopped Flag / Average Speed from GPS',
         'Independent distance and speed measure; dwell period identification.',
         'Operational Severity'),
        ('Event Counters',
         'Cumulative tallies of severity events — rising counts suggest escalating risk.',
         'Operational Severity'),
    ]

    rel_lines = ['# Severity Relevance Report', '',
                 'Each feature\'s relevance to tyre wear and severity category.', '']
    for name, desc, cat in entries:
        rel_lines.append(f'## {name}')
        rel_lines.append(f'- **Relevance:** {desc}')
        rel_lines.append(f'- **Category:** {cat}')
        rel_lines.append('')
    rel_lines.append('---')
    rel_lines.append('*Report generated by OSI Phase 2 pipeline*')
    Path(OUT_DIR / 'severity_relevance.md').write_text('\n'.join(rel_lines))
    print('    -> severity_relevance.md')

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print('\n' + '=' * 60)
    print('OSI Phase 2 complete.')
    print('=' * 60)
    print()
    print(f'  Features engineered: {len(feature_cols):,}')
    print(f'  Feature groups:      {sum(1 for v in groups.values() if v)}')
    # Approximate memory from parquet file size
    feat_size_mb = os.path.getsize(FEATURES_PATH) / 1e6
    print(f'  Feature parquet size: {feat_size_mb:.1f} MB')
    print(f'  Rows:                {total_feat_rows:,}')
    print()
    print('  Feature groups created:')
    for gn, gc in groups.items():
        if gc:
            print(f'    - {gn}: {len(gc)}')
    print()
    ready = 'YES' if len(feature_cols) > 50 else 'PARTIAL'
    print(f'  Ready for OSI scoring: {ready}')
    print()


if __name__ == '__main__':
    main()
