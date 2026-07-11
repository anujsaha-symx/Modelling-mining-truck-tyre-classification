"""
OSI Phase 4: OSI Aggregation
==============================
Aggregates OSI values at hourly, daily, and weekly levels.
"""

import pandas as pd
import numpy as np
from src.severity_sensor.osi_events import severity_time_distribution


def _resample_osi(df: pd.DataFrame, osi_cols: list, freq: str) -> pd.DataFrame:
    """Resample OSI columns to a given frequency.

    Parameters
    ----------
    df : pd.DataFrame
        Must have a 'timestamp' column (datetime).
    osi_cols : list
        Columns to aggregate.
    freq : str
        Pandas offset string ('h', 'D', 'W').

    Returns
    -------
    pd.DataFrame with resampled statistics.
    """
    ts = df.set_index('timestamp')
    resampled = ts.resample(freq)
    result = pd.DataFrame(index=resampled.indices)
    result.index.name = 'timestamp'

    for col in osi_cols:
        if col not in ts.columns:
            continue
        group = ts[col]
        result[f'{col}_mean'] = resampled[col].mean()
        result[f'{col}_max'] = resampled[col].max()
        result[f'{col}_p95'] = resampled[col].quantile(0.95)
        result[f'{col}_std'] = resampled[col].std()

    return result.reset_index()


def aggregate_hourly(df: pd.DataFrame, osi_cols: list) -> pd.DataFrame:
    """Aggregate OSI to hourly level with severity distributions."""
    result = _resample_osi(df, osi_cols, 'h')

    # Add severity distribution per hour
    ts = df.set_index('timestamp')
    for col in osi_cols:
        if col not in ts.columns:
            continue
        hourly_sev = ts[col].resample('h').agg(severity_time_distribution)
        sev_df = pd.DataFrame(hourly_sev.tolist(), index=hourly_sev.index)
        for level in ['Normal', 'Moderate', 'Severe', 'Critical']:
            if level in sev_df.columns:
                result[f'{col}_pct_{level}'] = sev_df[level].values

    return result


def aggregate_daily(df: pd.DataFrame, osi_cols: list) -> pd.DataFrame:
    """Aggregate OSI to daily level."""
    result = _resample_osi(df, osi_cols, 'D')

    ts = df.set_index('timestamp')
    for col in osi_cols:
        if col not in ts.columns:
            continue
        daily_sev = ts[col].resample('D').agg(severity_time_distribution)
        sev_df = pd.DataFrame(daily_sev.tolist(), index=daily_sev.index)
        for level in ['Normal', 'Moderate', 'Severe', 'Critical']:
            if level in sev_df.columns:
                result[f'{col}_pct_{level}'] = sev_df[level].values

    return result


def aggregate_weekly(df: pd.DataFrame, osi_cols: list) -> pd.DataFrame:
    """Aggregate OSI to weekly level."""
    result = _resample_osi(df, osi_cols, 'W')

    ts = df.set_index('timestamp')
    for col in osi_cols:
        if col not in ts.columns:
            continue
        weekly_sev = ts[col].resample('W').agg(severity_time_distribution)
        sev_df = pd.DataFrame(weekly_sev.tolist(), index=weekly_sev.index)
        for level in ['Normal', 'Moderate', 'Severe', 'Critical']:
            if level in sev_df.columns:
                result[f'{col}_pct_{level}'] = sev_df[level].values

    return result
