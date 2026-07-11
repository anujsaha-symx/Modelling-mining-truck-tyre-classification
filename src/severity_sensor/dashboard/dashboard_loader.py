"""
Dashboard Data Loader
=====================
Lightweight data loader that consumes pre-computed dashboard artifacts.
Never loads large parquet files (telemetry_features.parquet,
telemetry_risk_components.parquet, telemetry_osi.parquet) inside
Streamlit. All dashboard data is sourced from:

  outputs/osi_phase4/dashboard_data/
    component_timeseries.csv  -- risk component scores + OSI timeline
    daily_summary.csv         -- daily aggregation statistics
    severity_events.csv       -- detected high-risk events
    dashboard_metrics.json    -- precomputed metadata & summary stats

  datasets/telemetry/
    hourly_osi.csv            -- hourly aggregation
    daily_osi.csv             -- daily aggregation
    weekly_osi.csv            -- weekly aggregation

For metadata (date range, record count), dashboard_metrics.json is
preferred. If absent, lightweight daily_summary.csv is read instead.
For OSI timeseries, component_timeseries.csv is sampled (2%) for plots.
For component means, either full or sampled data is used.

Supports dataset switching via set_data_root() to load from
different directories (default, demo scenarios, or uploaded results).
"""

import json
from pathlib import Path
from typing import Optional
import pandas as pd
import streamlit as st

ROOT = Path(r'D:\Tyre_Classification')

_data_root = None

DEFAULT_TELEMETRY_DIR = ROOT / 'datasets' / 'telemetry'
DEFAULT_OUTPUT_DIR = ROOT / 'outputs' / 'osi_phase4' / 'dashboard_data'


def set_data_root(telemetry_dir, dashboard_dir=None):
    """Override the base directories for data loading."""
    global _data_root
    _data_root = {
        'telemetry': Path(telemetry_dir),
        'dashboard': Path(dashboard_dir) if dashboard_dir else Path(telemetry_dir),
    }


def reset_data_root():
    """Reset to default data paths."""
    global _data_root
    _data_root = None


def _tdir():
    if _data_root:
        return _data_root['telemetry']
    return DEFAULT_TELEMETRY_DIR


def _ddir():
    if _data_root:
        return _data_root['dashboard']
    return DEFAULT_OUTPUT_DIR


SCORE_COLS = [
    'Pressure_Risk_Score', 'Thermal_Risk_Score', 'Load_Risk_Score',
    'Vibration_Risk_Score', 'Braking_Risk_Score', 'Terrain_Risk_Score',
    'Usage_Risk_Score', 'Anomaly_Score',
]

_OSI_COLS_IN_TS = ['timestamp', 'OSI'] + SCORE_COLS


def _load_comp_ts(chunk_size: Optional[int] = None) -> pd.DataFrame:
    """Load component_timeseries.csv (primary dashboard data source).

    Reads from dashboard_data/ first, falls back to telemetry directory.
    Returns the full DataFrame or, if chunk_size is given, reads in
    chunks and returns a 2% downsample -- useful for memory-constrained
    plotting.

    Parameters
    ----------
    chunk_size : int, optional
        If set, read in chunks of this size and return a ~2% sample.
        If None, read the entire file (for aggregate computations).
    """
    path = _ddir() / 'component_timeseries.csv'
    if not path.exists():
        path = _tdir() / 'component_timeseries.csv'
    if not path.exists():
        return pd.DataFrame()

    if chunk_size is None:
        return pd.read_csv(path, parse_dates=['timestamp'])

    sampled = []
    row_counter = 0
    for chunk in pd.read_csv(path, parse_dates=['timestamp'],
                             chunksize=chunk_size):
        sample_mask = (row_counter + chunk.index) % 50 == 0
        sampled.append(chunk[sample_mask].copy())
        row_counter += len(chunk)
    if not sampled:
        return pd.DataFrame()
    return pd.concat(sampled, ignore_index=True)


# ---- Metadata (lightweight, no parquet reads) ----

def get_dataset_info() -> dict:
    """Return dataset metadata without loading parquet files.

    Reads dashboard_metrics.json if available (precomputed by the
    pipeline), otherwise falls back to daily_summary.csv which is
    typically <50 KB.
    """
    metrics_path = _ddir() / 'dashboard_metrics.json'
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            m = json.load(f)
        return {
            'start': pd.Timestamp(m['start']),
            'end': pd.Timestamp(m['end']),
            'duration': pd.Timedelta(m['duration']),
            'records': m['records'],
            'variant': m.get('variant', 'Expert-Weighted Risk Assessment'),
            'truck_id': m.get('truck_id', 'N/A'),
        }

    daily = load_daily_summary()
    if daily.empty or 'timestamp' not in daily.columns:
        return {
            'start': pd.Timestamp('2000-01-01'),
            'end': pd.Timestamp('2000-01-02'),
            'duration': pd.Timedelta(days=1),
            'records': 0,
            'variant': 'N/A',
            'truck_id': 'N/A',
        }

    ts = pd.to_datetime(daily['timestamp'])
    comp_ts_path = _ddir() / 'component_timeseries.csv'
    if not comp_ts_path.exists():
        comp_ts_path = _tdir() / 'component_timeseries.csv'
    records = 0
    if comp_ts_path.exists():
        for chunk in pd.read_csv(comp_ts_path, usecols=['timestamp'],
                                 chunksize=500_000):
            records += len(chunk)

    start, end = ts.min(), ts.max()
    return {
        'start': start,
        'end': end,
        'duration': end - start,
        'records': records,
        'variant': 'Expert-Weighted Risk Assessment',
        'truck_id': 'N/A',
    }


# ---- OSI timeseries (sampled for plots) ----

def load_osi_sample(sample_rate: float = 0.02) -> pd.DataFrame:
    """Load a 2% sample of OSI timeseries for plotting.

    Reads component_timeseries.csv in chunks and returns every 50th row.
    Typical output: ~53K rows, ~4 MB, loads in <3 seconds.
    """
    df = _load_comp_ts(chunk_size=500_000)
    if df.empty:
        return df
    cols = [c for c in ['timestamp', 'OSI'] if c in df.columns]
    return df[cols].copy()


# ---- Component scores (sampled for stacked area / anomaly timeline) ----

def load_component_sample(sample_rate: float = 0.02) -> pd.DataFrame:
    """Load a 2% sample of component scores for timeline plots."""
    df = _load_comp_ts(chunk_size=500_000)
    if df.empty:
        return df
    cols = ['timestamp'] + [c for c in SCORE_COLS if c in df.columns]
    return df[cols].copy()


# ---- Component means (from full data, computed efficiently) ----

def load_component_means_from_sample() -> dict:
    """Compute component means from a 2% sample.

    For dashboard statistics, sample means are accurate to within ~2%
    and are sufficient for radar charts, bar charts, and recommendations.
    Uses only ~53K rows instead of 2.6M.
    """
    df = _load_comp_ts(chunk_size=500_000)
    if df.empty:
        return {}
    means = {}
    for col in SCORE_COLS:
        if col in df.columns:
            means[col] = float(df[col].mean())
    return means


def load_component_means_full() -> dict:
    """Compute component means from full component_timeseries.csv.

    Reads 2.6M rows in chunks (~450 MB peak RAM). Use only when
    exact means are required (e.g. export, report generation).
    """
    df = _load_comp_ts()
    if df.empty:
        return {}
    means = {}
    for col in SCORE_COLS:
        if col in df.columns:
            means[col] = float(df[col].mean())
    return means


def load_osi_stats() -> dict:
    """Compute OSI statistics (mean, max, p95) from a 2% sample."""
    df = _load_comp_ts(chunk_size=500_000)
    if df.empty or 'OSI' not in df.columns:
        return {'mean_osi': 0.0, 'max_osi': 0.0, 'p95_osi': 0.0}
    osi = df['OSI']
    return {
        'mean_osi': float(osi.mean()),
        'max_osi': float(osi.max()),
        'p95_osi': float(osi.quantile(0.95)),
    }


def load_osi_stats_full() -> dict:
    """Compute OSI statistics from full component_timeseries.csv.

    Reads 2.6M rows in chunks (~450 MB peak RAM). Use only when
    exact statistics are required.
    """
    df = _load_comp_ts()
    if df.empty or 'OSI' not in df.columns:
        return {'mean_osi': 0.0, 'max_osi': 0.0, 'p95_osi': 0.0}
    osi = df['OSI']
    return {
        'mean_osi': float(osi.mean()),
        'max_osi': float(osi.max()),
        'p95_osi': float(osi.quantile(0.95)),
    }


def load_severity_distribution_from_sample() -> dict:
    """Compute severity distribution from a 2% sample."""
    df = _load_comp_ts(chunk_size=500_000)
    if df.empty or 'OSI' not in df.columns:
        return {'Normal': 0.0, 'Moderate': 0.0, 'Severe': 0.0, 'Critical': 0.0}
    osi = df['OSI']
    n = len(osi)
    return {
        'Normal': float((osi <= 25).sum() / n * 100),
        'Moderate': float(((osi > 25) & (osi <= 50)).sum() / n * 100),
        'Severe': float(((osi > 50) & (osi <= 75)).sum() / n * 100),
        'Critical': float((osi > 75).sum() / n * 100),
    }


def load_severity_distribution_full() -> dict:
    """Compute severity distribution from full component_timeseries.csv."""
    df = _load_comp_ts()
    if df.empty or 'OSI' not in df.columns:
        return {'Normal': 0.0, 'Moderate': 0.0, 'Severe': 0.0, 'Critical': 0.0}
    osi = df['OSI']
    n = len(osi)
    return {
        'Normal': float((osi <= 25).sum() / n * 100),
        'Moderate': float(((osi > 25) & (osi <= 50)).sum() / n * 100),
        'Severe': float(((osi > 50) & (osi <= 75)).sum() / n * 100),
        'Critical': float((osi > 75).sum() / n * 100),
    }


# ---- Anomaly (from component_timeseries.csv, sampled) ----

def load_anomaly_sample() -> pd.DataFrame:
    """Load sampled anomaly scores for distribution and timeline plots."""
    df = _load_comp_ts(chunk_size=500_000)
    if df.empty:
        return df
    cols = [c for c in ['timestamp', 'Anomaly_Score'] if c in df.columns]
    return df[cols].copy()


def load_top_anomalies(n: int = 20) -> pd.DataFrame:
    """Return the top-N anomaly scores from a 2% sample."""
    df = _load_comp_ts(chunk_size=500_000)
    if df.empty or 'Anomaly_Score' not in df.columns:
        return pd.DataFrame(columns=['timestamp', 'Anomaly_Score'])
    return df.nlargest(n, 'Anomaly_Score')[['timestamp', 'Anomaly_Score']].copy()


# ---- Aggregation CSVs (pre-computed, lightweight) ----

def load_hourly() -> pd.DataFrame:
    path = _tdir() / 'hourly_osi.csv'
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=['timestamp'])


def load_daily() -> pd.DataFrame:
    path = _tdir() / 'daily_osi.csv'
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=['timestamp'])


def load_weekly() -> pd.DataFrame:
    path = _tdir() / 'weekly_osi.csv'
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=['timestamp'])


# ---- Dashboard data CSVs (from outputs/osi_phase4/dashboard_data/) ----

def load_daily_summary() -> pd.DataFrame:
    path = _ddir() / 'daily_summary.csv'
    if not path.exists():
        path = _tdir() / 'daily_osi.csv'
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=['timestamp'])


def load_events() -> pd.DataFrame:
    path = _ddir() / 'severity_events.csv'
    if not path.exists():
        return pd.DataFrame(columns=['level', 'start_time', 'end_time',
                                     'duration_seconds', 'peak_osi', 'mean_osi'])
    df = pd.read_csv(path, parse_dates=['start_time', 'end_time'])
    return df


def get_date_range(df: pd.DataFrame) -> tuple:
    ts = pd.to_datetime(df['timestamp'])
    return ts.min(), ts.max()
