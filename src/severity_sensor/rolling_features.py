"""
Rolling window feature computations for OSI Phase 2.

Provides reusable functions to compute rolling statistics
for telemetry sensor data at multiple window sizes.
"""

from typing import List, Dict, Optional
import pandas as pd
import numpy as np


WINDOWS = {
    '1min': 60,
    '5min': 300,
    '15min': 900,
    '1hr': 3600,
}


def compute_rolling_stats(
    df: pd.DataFrame,
    columns: List[str],
    windows: Optional[Dict[str, int]] = None,
    stats: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Compute rolling statistics for given columns at multiple window sizes.

    Processes one column at a time and accumulates into a result DataFrame
    to keep peak memory usage low.

    .. note::
       ``median`` is only computed for windows <= 300 (5 min) because
       rolling median is O(n × w × log w) and dominates runtime for
       large windows.

    Parameters
    ----------
    df : pd.DataFrame
        Must have a datetime index (for rolling windows).
    columns : List[str]
        Numeric columns to compute rolling stats for.
    windows : dict, optional
        Mapping of window label -> window size in rows. Defaults to WINDOWS.
    stats : list, optional
        Statistics to compute. Defaults to ['mean', 'std', 'min', 'max', 'median'].

    Returns
    -------
    pd.DataFrame
        DataFrame indexed like ``df`` with columns ``{col}_{stat}_{window}``.
    """
    if windows is None:
        windows = WINDOWS
    if stats is None:
        stats = ['mean', 'std', 'min', 'max']

    result = pd.DataFrame(index=df.index)
    for col in columns:
        for window_label, window_size in windows.items():
            roll = df[col].rolling(window=window_size, min_periods=1)
            for stat in stats:
                col_name = f'{col}_{stat}_{window_label}'
                if stat == 'mean':
                    result[col_name] = roll.mean()
                elif stat == 'std':
                    result[col_name] = roll.std(ddof=0)
                elif stat == 'min':
                    result[col_name] = roll.min()
                elif stat == 'max':
                    result[col_name] = roll.max()
                elif stat == 'median':
                    result[col_name] = roll.median()
    return result


def compute_rolling_gradient(
    series: pd.Series,
    window: int = 60,
    label: str = '',
) -> pd.Series:
    """Compute rate of change over a rolling window.

    Uses linear regression slope over the window as an approximation
    of the gradient.  For 1 Hz data *window* rows ≈ *window* seconds.
    """
    def _slope(x):
        if len(x) < 2:
            return np.nan
        y = x.values.astype(float)
        n = len(y)
        xs = np.arange(n)
        return np.polyfit(xs, y, 1)[0]

    result = series.rolling(window=window, min_periods=2).apply(_slope, raw=False)
    if label:
        result.name = label
    return result


def compute_zscore(
    series: pd.Series,
    window: int = 3600,
    label: str = '',
) -> pd.Series:
    """Compute rolling z-score over a given window."""
    roll = series.rolling(window=window, min_periods=10)
    mean = roll.mean()
    std = roll.std(ddof=0)
    result = (series - mean) / std.replace(0, np.nan)
    if label:
        result.name = label
    return result


def compute_rms(series: pd.Series, window: int = 60, label: str = '') -> pd.Series:
    """Compute rolling RMS over a window."""
    result = (series ** 2).rolling(window=window, min_periods=1).mean().apply(np.sqrt)
    if label:
        result.name = label
    return result


def compute_crest_factor(
    signal: pd.Series,
    rms: pd.Series,
    label: str = '',
) -> pd.Series:
    """Compute crest factor = peak(absolute) / RMS."""
    peak = signal.abs().rolling(window=60, min_periods=1).max()
    result = peak / rms.replace(0, np.nan)
    if label:
        result.name = label
    return result
