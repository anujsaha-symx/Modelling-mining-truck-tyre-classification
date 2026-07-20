"""
Demo Dataset Manager
====================
Manages precomputed demonstration datasets for the sensor monitoring
dashboard. Splits the full analysis into five distinct operational
scenarios for reviewer demonstrations.

Each scenario provides a different operational profile:
  - Normal Operation: baseline low-severity conditions
  - Heavy Load: elevated payload and load risk
  - High Temperature: thermal stress conditions
  - Brake Intensive: frequent braking events
  - Terrain Stress: rough terrain and vibration stress

Data sources (never reads large parquet files):
  outputs/osi_phase4/dashboard_data/component_timeseries.csv
  datasets/telemetry/hourly_osi.csv, daily_osi.csv, weekly_osi.csv
"""

import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TELEMETRY_DIR = ROOT / 'datasets' / 'telemetry'
OUTPUT_DASH_DIR = ROOT / 'outputs' / 'osi_phase4' / 'dashboard_data'
DEMO_DIR = ROOT / 'datasets' / 'demo_datasets'

SCORE_COLS = [
    'Pressure_Risk_Score', 'Thermal_Risk_Score', 'Load_Risk_Score',
    'Vibration_Risk_Score', 'Braking_Risk_Score', 'Terrain_Risk_Score',
    'Usage_Risk_Score', 'Anomaly_Score',
]

DEMO_SCENARIOS = {
    'Normal Operation': {
        'description': 'Baseline operating conditions with low severity.',
        'osi_range': (0, 30),
    },
    'Heavy Load': {
        'description': 'Elevated payload and sustained load stress.',
        'osi_range': (20, 60),
    },
    'High Temperature': {
        'description': 'Thermal stress from sustained high temperatures.',
        'osi_range': (25, 70),
    },
    'Brake Intensive': {
        'description': 'Frequent braking events with elevated thermal load.',
        'osi_range': (15, 55),
    },
    'Terrain Stress': {
        'description': 'Rough terrain causing vibration and impact stress.',
        'osi_range': (10, 50),
    },
}


def _find_comp_ts():
    """Locate component_timeseries.csv without loading it."""
    for d in [OUTPUT_DASH_DIR, TELEMETRY_DIR]:
        p = d / 'component_timeseries.csv'
        if p.exists():
            return p
    return None


def _read_comp_ts_full():
    """Read component_timeseries.csv fully. ~400 MB, ~3.5 GB peak for 2.6M rows."""
    path = _find_comp_ts()
    if path is None:
        return None
    return pd.read_csv(path, parse_dates=['timestamp'])


def get_available_datasets():
    """Return list of available demo dataset names."""
    available = []
    for name in DEMO_SCENARIOS:
        demo_dir = DEMO_DIR / name.replace(' ', '_')
        if demo_dir.exists():
            if (demo_dir / 'component_timeseries.csv').exists():
                available.append(name)
    return available


def get_dataset_descriptions():
    """Return descriptions for all demo scenarios."""
    return {name: info['description'] for name, info in DEMO_SCENARIOS.items()}


def build_demo_datasets():
    """Segment the full dataset into five demo scenarios.

    Reads component_timeseries.csv (the only lightweight source that
    contains both OSI and risk component scores) and creates five
    subdirectories under datasets/demo_datasets/, each containing
    a subset of the data representative of a specific operating
    condition.
    """
    try:
        return _build_demo_datasets_inner()
    except MemoryError:
        return False, 'Not enough memory to build demo datasets.'
    except Exception as e:
        return False, f'Could not build demo datasets: {e}'


def _build_demo_datasets_inner():
    comp_ts = _read_comp_ts_full()
    if comp_ts is None or comp_ts.empty:
        return False, 'component_timeseries.csv not found. Run the telemetry pipeline first.'

    osi_col = 'OSI' if 'OSI' in comp_ts.columns else None
    if osi_col is None:
        return False, 'OSI column missing from component_timeseries.csv.'

    n = len(comp_ts)
    if n < 100:
        return False, 'Dataset too small to split into demo scenarios.'

    segment_size = n // 5
    comp_ts = comp_ts.reset_index(drop=True)

    scenario_segments = [
        ('Normal Operation', slice(0, segment_size)),
        ('Heavy Load', slice(segment_size, 2 * segment_size)),
        ('High Temperature', slice(2 * segment_size, 3 * segment_size)),
        ('Brake Intensive', slice(3 * segment_size, 4 * segment_size)),
        ('Terrain Stress', slice(4 * segment_size, n)),
    ]

    DEMO_DIR.mkdir(parents=True, exist_ok=True)

    hourly_df = _load_agg_csv('hourly_osi.csv')
    daily_df = _load_agg_csv('daily_osi.csv')
    weekly_df = _load_agg_csv('weekly_osi.csv')

    for name, seg in scenario_segments:
        demo_dir = DEMO_DIR / name.replace(' ', '_')
        demo_dir.mkdir(parents=True, exist_ok=True)

        comp_seg = comp_ts.iloc[seg].copy()

        comp_seg.to_csv(demo_dir / 'component_timeseries.csv', index=False)

        events = _make_events_from_segment(comp_seg, osi_col)
        events.to_csv(demo_dir / 'severity_events.csv', index=False)

        osi_seg = comp_seg[['timestamp', osi_col]].copy()
        hourly, daily, weekly = _make_aggregations(osi_seg, osi_col)
        hourly.to_csv(demo_dir / 'hourly_osi.csv', index=False)
        daily.to_csv(demo_dir / 'daily_osi.csv', index=False)
        weekly.to_csv(demo_dir / 'weekly_osi.csv', index=False)
        daily.to_csv(demo_dir / 'daily_summary.csv', index=False)

        metrics = {
            'start': str(comp_seg['timestamp'].min()),
            'end': str(comp_seg['timestamp'].max()),
            'duration': str(comp_seg['timestamp'].max() - comp_seg['timestamp'].min()),
            'records': len(comp_seg),
            'variant': 'Expert-Weighted Risk Assessment',
            'truck_id': 'N/A',
            'mean_osi': float(comp_seg[osi_col].mean()),
            'max_osi': float(comp_seg[osi_col].max()),
            'event_count': len(events),
        }
        with open(demo_dir / 'dashboard_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2, default=str)

    return True, f'Demo datasets created: {len(scenario_segments)} scenarios in {DEMO_DIR}'


def _load_agg_csv(name):
    """Load an aggregation CSV from telemetry dir, return empty DataFrame if missing."""
    path = TELEMETRY_DIR / name
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, parse_dates=['timestamp'])
    except Exception:
        return pd.DataFrame()


def _make_events_from_segment(comp_seg, osi_col='OSI'):
    """Create event CSV from a component_timeseries segment."""
    osi = comp_seg[osi_col].reset_index(drop=True)
    timestamps = comp_seg['timestamp'].reset_index(drop=True)

    events = []
    for level, (lo, hi) in [('Critical', (75, 100)), ('Severe', (50, 75)), ('Moderate', (25, 50))]:
        above = (osi >= lo).astype(int)
        change_points = above.diff().fillna(0)
        starts = (change_points == 1).values
        ends = (change_points == -1).values
        active = False
        start_idx = None
        for i in range(len(above)):
            if starts[i]:
                active = True
                start_idx = i
            if ends[i] and active:
                duration = i - start_idx
                if duration >= 1:
                    events.append({
                        'level': level,
                        'start_time': timestamps.iloc[start_idx],
                        'end_time': timestamps.iloc[i - 1],
                        'duration_seconds': duration,
                        'peak_osi': float(osi.iloc[start_idx:i].max()),
                        'mean_osi': float(osi.iloc[start_idx:i].mean()),
                    })
                active = False
        if active and start_idx is not None:
            duration = len(above) - start_idx
            if duration >= 1:
                events.append({
                    'level': level,
                    'start_time': timestamps.iloc[start_idx],
                    'end_time': timestamps.iloc[-1],
                    'duration_seconds': duration,
                    'peak_osi': float(osi.iloc[start_idx:].max()),
                    'mean_osi': float(osi.iloc[start_idx:].mean()),
                })

    if not events:
        return pd.DataFrame(columns=['level', 'start_time', 'end_time',
                                     'duration_seconds', 'peak_osi', 'mean_osi'])
    return pd.DataFrame(events).sort_values('start_time').reset_index(drop=True)


def _make_aggregations(osi_df, osi_col='OSI'):
    """Create hourly, daily, weekly aggregations."""
    osi_cols = [c for c in [osi_col, 'OSI_Base'] if c in osi_df.columns]

    def _agg(df, freq):
        if df.empty:
            return pd.DataFrame()
        ts = df.set_index('timestamp')
        resampled = ts.resample(freq)
        result = pd.DataFrame(index=resampled.indices)
        result.index.name = 'timestamp'
        for col in osi_cols:
            if col not in ts.columns:
                continue
            result[f'{col}_mean'] = resampled[col].mean()
            result[f'{col}_max'] = resampled[col].max()
            result[f'{col}_p95'] = resampled[col].quantile(0.95)
            result[f'{col}_std'] = resampled[col].std()
        return result.reset_index()

    hourly = _agg(osi_df, 'h')
    daily = _agg(osi_df, 'D')
    weekly = _agg(osi_df, 'W')
    return hourly, daily, weekly


def get_demo_dir_for_scenario(scenario_name):
    """Return the path to a demo scenario directory."""
    return DEMO_DIR / scenario_name.replace(' ', '_')
