"""
OSI Phase 4: Severity Event Detection
======================================
Detects contiguous periods where OSI exceeds severity thresholds.
"""

import pandas as pd
import numpy as np


SEVERITY_LEVELS = {
    'Normal':   (0, 25),
    'Moderate': (25, 50),
    'Severe':   (50, 75),
    'Critical': (75, 100),
}


def classify_severity(osi: pd.Series) -> pd.Series:
    """Map OSI values to severity level labels."""
    result = pd.Series('Normal', index=osi.index)
    result[osi > 75] = 'Critical'
    result[(osi > 50) & (osi <= 75)] = 'Severe'
    result[(osi > 25) & (osi <= 50)] = 'Moderate'
    return result


def classify_severity_code(osi: pd.Series) -> pd.Series:
    """Map OSI to numeric severity codes: 0=Normal, 1=Moderate, 2=Severe, 3=Critical."""
    codes = np.zeros(len(osi), dtype=int)
    codes[osi > 75] = 3
    codes[(osi > 50) & (osi <= 75)] = 2
    codes[(osi > 25) & (osi <= 50)] = 1
    return pd.Series(codes, index=osi.index)


def detect_events(osi: pd.Series, timestamps: pd.Series,
                  threshold: float = 50,
                  min_duration: int = 1) -> pd.DataFrame:
    """Detect contiguous events where OSI >= threshold.

    Parameters
    ----------
    osi : pd.Series
        OSI values.
    timestamps : pd.Series
        Corresponding timestamps.
    threshold : float
        Minimum OSI value to consider.
    min_duration : int
        Minimum number of consecutive rows to form an event.

    Returns
    -------
    pd.DataFrame with start_time, end_time, duration, peak_osi, mean_osi.
    """
    above = (osi >= threshold).astype(int)
    change_points = above.diff().fillna(0)
    starts = (change_points == 1).values
    ends = (change_points == -1).values

    events = []
    active = False
    start_idx = None
    for i in range(len(above)):
        if starts[i]:
            active = True
            start_idx = i
        if ends[i] and active:
            duration = i - start_idx
            if duration >= min_duration:
                events.append({
                    'start_time': timestamps.iloc[start_idx],
                    'end_time': timestamps.iloc[i - 1],
                    'duration_seconds': duration,
                    'peak_osi': osi.iloc[start_idx:i].max(),
                    'mean_osi': osi.iloc[start_idx:i].mean(),
                })
            active = False
    # Handle event that extends to the end
    if active and start_idx is not None:
        duration = len(above) - start_idx
        if duration >= min_duration:
            events.append({
                'start_time': timestamps.iloc[start_idx],
                'end_time': timestamps.iloc[-1],
                'duration_seconds': duration,
                'peak_osi': osi.iloc[start_idx:].max(),
                'mean_osi': osi.iloc[start_idx:].mean(),
            })

    if not events:
        return pd.DataFrame(columns=['start_time', 'end_time', 'duration_seconds',
                                      'peak_osi', 'mean_osi'])
    return pd.DataFrame(events)


def detect_all_events(osi: pd.Series, timestamps: pd.Series,
                      min_duration: int = 1) -> pd.DataFrame:
    """Detect events for all severity levels.

    Returns
    -------
    pd.DataFrame with level, start_time, end_time, duration, peak, mean.
    """
    all_events = []
    for level, (lo, hi) in SEVERITY_LEVELS.items():
        if level == 'Normal':
            continue
        threshold = lo  # bottom of the range
        events = detect_events(osi, timestamps, threshold=lo, min_duration=min_duration)
        if len(events) > 0:
            events.insert(0, 'level', level)
            all_events.append(events)
    if not all_events:
        return pd.DataFrame(columns=['level', 'start_time', 'end_time',
                                      'duration_seconds', 'peak_osi', 'mean_osi'])
    return pd.concat(all_events, ignore_index=True).sort_values('start_time').reset_index(drop=True)


def severity_time_distribution(osi: pd.Series) -> dict:
    """Return percentage of time spent in each severity level."""
    n = len(osi)
    dist = {}
    for level, (lo, hi) in SEVERITY_LEVELS.items():
        if level == 'Normal':
            cnt = (osi <= 25).sum()
        elif level == 'Critical':
            cnt = (osi > 75).sum()
        else:
            cnt = ((osi > lo) & (osi <= hi)).sum()
        dist[level] = cnt / n * 100
    return dist
