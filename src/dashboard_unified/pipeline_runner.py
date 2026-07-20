"""
Sensor Data Analysis Pipeline Runner
=====================================
Processes uploaded sensor data through the full telemetry analysis
pipeline without modifying any underlying algorithms.

Modes:
  - CSV upload: accepts payload + TPMS CSVs or a single combined CSV
  - Parquet upload: accepts pre-merged telemetry parquet

All outputs are written to a user-specified directory for
dashboard consumption.
"""

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.severity_sensor.build_master_dataset import (
    load_payload_chunked,
    load_tpms_chunked,
    remove_dup_timestamps,
    flag_missing,
    PAYLOAD_COLS,
    TPMS_COLS,
    CHUNK_SIZE,
)
from src.severity_sensor.feature_engineering import features_for_chunk
from src.severity_sensor.component_builder import (
    Normaliser,
    _build_rg_bounds,
    _read_rg,
    ALL_DERIVED,
    FEATURE_GROUPS,
)
from src.severity_sensor.risk_scoring import compute_all_risk_scores
from src.severity_sensor.anomaly_detection import AnomalyEnsemble, DERIVED_FEATURES
from src.severity_sensor.osi_builder import (
    build_weighted_osi,
    WEIGHTS,
    SCORE_COLS,
)
from src.severity_sensor.osi_events import (
    detect_all_events,
    classify_severity,
    classify_severity_code,
    severity_time_distribution,
    SEVERITY_LEVELS,
)
from src.severity_sensor.osi_aggregation import (
    aggregate_hourly,
    aggregate_daily,
    aggregate_weekly,
)

CHUNK_ROWS = 432_000
OVERLAP = 3600


def _ensure_timestamp(df, ts_col='timestamp'):
    if ts_col not in df.columns:
        for c in df.columns:
            if 'timestamp' in c.lower() or 'time' in c.lower():
                ts_col = c
                break
    if ts_col != 'timestamp' and ts_col in df.columns:
        df = df.rename(columns={ts_col: 'timestamp'})
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        if hasattr(df['timestamp'].dt, 'tz') and df['timestamp'].dt.tz is not None:
            df['timestamp'] = df['timestamp'].dt.tz_localize(None)
    return df


def run_full_pipeline(
    payload_path=None,
    tpms_path=None,
    combined_path=None,
    output_dir=None,
    progress_callback=None,
):
    """Run the full telemetry analysis pipeline on uploaded data.

    Parameters
    ----------
    payload_path : str or Path, optional
        Path to payload CSV (with sample_timestamp).
    tpms_path : str or Path, optional
        Path to TPMS CSV (with _col3).
    combined_path : str or Path, optional
        Path to a combined CSV/parquet with all required columns.
    output_dir : str or Path
        Directory to write all pipeline outputs.
    progress_callback : callable, optional
        Called with (stage_name, progress_fraction) for UI updates.

    Returns
    -------
    dict with keys:
        success: bool
        output_dir: Path
        message: str
        stats: dict of key statistics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    telemetry_dir = output_dir / 'telemetry'
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    dashboard_dir = output_dir / 'dashboard_data'
    dashboard_dir.mkdir(parents=True, exist_ok=True)

    def _progress(stage, frac):
        if progress_callback:
            progress_callback(stage, frac)

    stats = {}
    try:
        _progress('Loading sensor data', 0.05)

        if combined_path is not None:
            combined_path = Path(combined_path)
            if combined_path.suffix.lower() == '.parquet':
                master = pd.read_parquet(combined_path)
            else:
                master = pd.read_csv(combined_path, parse_dates=[0])
            master = _ensure_timestamp(master)
        elif payload_path is not None and tpms_path is not None:
            payload = load_payload_chunked(Path(payload_path))
            tpms = load_tpms_chunked(Path(tpms_path))
            payload = payload.sort_values('timestamp').reset_index(drop=True)
            tpms = tpms.sort_values('timestamp').reset_index(drop=True)
            payload_exact = payload.drop_duplicates(subset=['timestamp'])
            tpms_exact = tpms.drop_duplicates(subset=['timestamp'])
            merged_exact = pd.merge(payload_exact, tpms_exact, on='timestamp', how='inner',
                                    suffixes=('', '_tpms'))
            for col in ['device_tpms', 'asset_tpms', 'GPSLatitude_tpms', 'GPSLongitude_tpms']:
                if col in merged_exact.columns:
                    merged_exact.drop(columns=[col], inplace=True)
            matched_ts = set(merged_exact['timestamp'])
            payload_unmatched = payload_exact[~payload_exact['timestamp'].isin(matched_ts)].copy()
            tpms_unmatched = tpms_exact[~tpms_exact['timestamp'].isin(matched_ts)].copy()
            if len(payload_unmatched) > 0 and len(tpms_unmatched) > 0:
                merged_asof = pd.merge_asof(
                    payload_unmatched.sort_values('timestamp'),
                    tpms_unmatched.sort_values('timestamp'),
                    on='timestamp', direction='nearest',
                    tolerance=pd.Timedelta('2s'),
                    suffixes=('', '_tpms'),
                )
                tpms_null_cols = [c for c in TPMS_COLS if c in merged_asof.columns]
                if tpms_null_cols:
                    merged_asof = merged_asof.dropna(subset=tpms_null_cols, how='all')
                for col in ['device_tpms', 'asset_tpms', 'GPSLatitude_tpms', 'GPSLongitude_tpms']:
                    if col in merged_asof.columns:
                        merged_asof.drop(columns=[col], inplace=True)
            else:
                merged_asof = pd.DataFrame()
            keep_cols = ['timestamp'] + PAYLOAD_COLS + TPMS_COLS
            merged_exact = merged_exact[[c for c in keep_cols if c in merged_exact.columns]]
            if len(merged_asof) > 0:
                merged_asof = merged_asof[[c for c in keep_cols if c in merged_asof.columns]]
            master = pd.concat([merged_exact, merged_asof], ignore_index=True)
            master = master.sort_values('timestamp').reset_index(drop=True)
            master = remove_dup_timestamps(master, 'Merged')
            master = flag_missing(master)
        else:
            return {
                'success': False,
                'output_dir': output_dir,
                'message': 'No valid data source provided.',
                'stats': {},
            }

        stats['master_rows'] = len(master)
        stats['master_cols'] = len(master.columns)

        _progress('Writing master dataset', 0.10)
        parquet_path = telemetry_dir / 'telemetry_master.parquet'
        master.to_parquet(parquet_path, index=False)

        _progress('Computing features', 0.15)
        pf = pq.ParquetFile(parquet_path)
        rg_bounds = _build_rg_bounds(pf)
        total_rows = rg_bounds[-1] if rg_bounds else 0
        all_cols = pf.schema_arrow.names
        keep_cols_feat = [c for c in all_cols if not c.endswith('_missing')]

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

        features_path = telemetry_dir / 'telemetry_features.parquet'
        cum_dist = 0.0
        cum_payload = 0.0
        cum_braking = 0.0
        cum_dist_gps = 0.0
        cum_counters = {}
        flag_names = ['overload_flag', 'pressure_anomaly_flag', 'overtemperature_flag',
                      'impact_event_flag', 'hard_braking_flag', 'rough_terrain_flag']

        writer = None
        for ci_idx, ck in enumerate(chunks):
            _progress(f'Features chunk {ci_idx + 1}/{len(chunks)}', 0.15 + 0.25 * (ci_idx / max(1, len(chunks))))
            context = _read_rg_slice(pf, rg_bounds, ck['context_start'],
                                     ck['context_end'] - ck['context_start'],
                                     columns=keep_cols_feat)
            local_start = ck['chunk_start'] - ck['context_start']
            local_end = ck['chunk_end'] - ck['context_start']
            result = features_for_chunk(context, local_start, local_end)

            if 'distance_travelled' in result.columns:
                vals = result['distance_travelled'].fillna(0).values
                result['cumulative_distance'] = np.cumsum(vals) + cum_dist
                cum_dist += vals.sum()
            if 'cumulative_payload_exposure' in result.columns:
                vals = result['cumulative_payload_exposure'].fillna(0).values
                result['cumulative_payload_exposure'] = np.cumsum(vals) + cum_payload
                cum_payload += vals.sum()
            if 'cumulative_braking_time' in result.columns:
                vals = result['cumulative_braking_time'].fillna(0).values
                result['cumulative_braking_time'] = np.cumsum(vals) + cum_braking
                cum_braking += vals.sum()
            if 'distance_increment' in result.columns:
                vals = result['distance_increment'].fillna(0).values
                result['cumulative_distance_gps'] = np.cumsum(vals) + cum_dist_gps
                cum_dist_gps += vals.sum()
            for flag in flag_names:
                if flag in result.columns:
                    key = f'num_{flag}'
                    vals = result[flag].fillna(0).astype(int).values
                    running = cum_counters.get(key, 0)
                    result[key] = np.cumsum(vals) + running
                    cum_counters[key] = running + vals.sum()

            table = pa.Table.from_pandas(result, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(str(features_path), table.schema)
            writer.write_table(table)
            del context, result, table

        if writer is not None:
            writer.close()
        pf.close()

        stats['feature_rows'] = total_rows
        _progress('Computing risk scores', 0.45)
        feat_pf = pq.ParquetFile(features_path)
        feat_keep = ['timestamp'] + ALL_DERIVED
        feat_keep = [c for c in feat_keep if c in feat_pf.schema_arrow.names]
        rg0 = feat_pf.read_row_group(0, columns=feat_keep).to_pandas()
        norm = Normaliser().fit(rg0)
        anomaly = AnomalyEnsemble(random_state=42)
        anomaly.fit(rg0)

        components_path = telemetry_dir / 'telemetry_risk_components.parquet'
        writer = None
        num_rgs = feat_pf.metadata.num_row_groups
        for rg_idx in range(num_rgs):
            _progress(f'Risk scores {rg_idx + 1}/{num_rgs}', 0.45 + 0.25 * (rg_idx / max(1, num_rgs)))
            df_chunk = feat_pf.read_row_group(rg_idx, columns=feat_keep).to_pandas()
            scores = compute_all_risk_scores(df_chunk)
            norm_df = norm.transform(df_chunk, columns=ALL_DERIVED)
            anom_df = anomaly.transform(df_chunk)
            out = scores.merge(norm_df, on='timestamp', how='left')
            out = out.merge(anom_df, on='timestamp', how='left')
            table = pa.Table.from_pandas(out, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(str(components_path), table.schema)
            writer.write_table(table)
            del df_chunk, scores, norm_df, anom_df, out, table
        if writer is not None:
            writer.close()
        feat_pf.close()

        stats['component_cols'] = len(SCORE_COLS)
        _progress('Building operational severity index', 0.75)
        comp_pf = pq.ParquetFile(components_path)
        needed = ['timestamp'] + SCORE_COLS
        read_cols = [c for c in needed if c in comp_pf.schema_arrow.names]
        osi_df = comp_pf.read(columns=read_cols).to_pandas()
        comp_pf.close()

        osi_df['OSI_Base'] = build_weighted_osi(osi_df, WEIGHTS)
        osi_df['OSI'] = osi_df['OSI_Base']
        osi_df['OSI_Level'] = classify_severity(osi_df['OSI'])
        osi_df['OSI_Level_Code'] = classify_severity_code(osi_df['OSI'])

        osi_out_cols = ['timestamp', 'OSI', 'OSI_Base', 'OSI_Level', 'OSI_Level_Code']
        osi_out_cols = [c for c in osi_out_cols if c in osi_df.columns]
        osi_path = telemetry_dir / 'telemetry_osi.parquet'
        osi_df[osi_out_cols].to_parquet(osi_path, index=False)

        stats['mean_osi'] = float(osi_df['OSI'].mean())
        stats['max_osi'] = float(osi_df['OSI'].max())

        _progress('Detecting events', 0.85)
        events = detect_all_events(osi_df['OSI'], osi_df['timestamp'], min_duration=1)
        events.to_csv(dashboard_dir / 'severity_events.csv', index=False)
        stats['event_count'] = len(events)

        _progress('Aggregating temporal data', 0.90)
        osi_cols_for_agg = ['OSI', 'OSI_Base']
        osi_cols_for_agg = [c for c in osi_cols_for_agg if c in osi_df.columns]
        hourly = aggregate_hourly(osi_df, osi_cols_for_agg)
        daily = aggregate_daily(osi_df, osi_cols_for_agg)
        weekly = aggregate_weekly(osi_df, osi_cols_for_agg)

        hourly.to_csv(telemetry_dir / 'hourly_osi.csv', index=False)
        daily.to_csv(telemetry_dir / 'daily_osi.csv', index=False)
        weekly.to_csv(telemetry_dir / 'weekly_osi.csv', index=False)
        daily.to_csv(dashboard_dir / 'daily_summary.csv', index=False)

        comp_timeseries = osi_df[['timestamp'] + SCORE_COLS].copy() if 'timestamp' in osi_df.columns else pd.DataFrame()
        if len(comp_timeseries) > 0:
            comp_timeseries.to_csv(dashboard_dir / 'component_timeseries.csv', index=False)

        _progress('Writing dashboard metrics', 0.95)
        metrics = {
            'start': str(osi_df['timestamp'].min()),
            'end': str(osi_df['timestamp'].max()),
            'duration': str(osi_df['timestamp'].max() - osi_df['timestamp'].min()),
            'records': len(osi_df),
            'variant': 'Expert-Weighted Risk Assessment',
            'truck_id': 'N/A',
            'mean_osi': float(osi_df['OSI'].mean()),
            'max_osi': float(osi_df['OSI'].max()),
            'event_count': len(events),
        }
        with open(dashboard_dir / 'dashboard_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2, default=str)

        _progress('Complete', 1.0)
        stats['output_dir'] = str(output_dir)
        return {
            'success': True,
            'output_dir': output_dir,
            'message': f'Pipeline completed. {stats["master_rows"]:,} records analysed.',
            'stats': stats,
        }

    except Exception as e:
        return {
            'success': False,
            'output_dir': output_dir,
            'message': f'Pipeline failed: {str(e)}',
            'stats': stats,
        }


def _read_rg_slice(pf, rg_bounds, start, length, columns=None):
    if length <= 0:
        return pd.DataFrame()
    end = start + length
    first_rg = 0
    while first_rg < len(rg_bounds) and rg_bounds[first_rg] <= start:
        first_rg += 1
    last_rg = first_rg
    while last_rg < len(rg_bounds) and rg_bounds[last_rg] < end:
        last_rg += 1
    tables = [pf.read_row_group(i, columns=columns) for i in range(first_rg, last_rg + 1)]
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
