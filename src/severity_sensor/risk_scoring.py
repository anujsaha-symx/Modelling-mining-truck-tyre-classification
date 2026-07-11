"""
OSI Phase 3: Risk Component Scoring
====================================
Computes individual risk scores for each severity dimension.
Each score is normalised to [0, 100].
"""

import numpy as np
import pandas as pd


def _normalise_to_100(series: pd.Series, cap: float = 100.0) -> pd.Series:
    """Min-max normalise a series to [0, cap] and clip outliers."""
    lo, hi = series.min(), series.max()
    if hi - lo < 1e-12:
        return pd.Series(0.0, index=series.index)
    return (series - lo) / (hi - lo) * cap


def _composite_score(features: pd.DataFrame, weights: dict,
                     missing_ok: bool = True) -> pd.Series:
    """Weighted average of selected features, normalised to [0, 100].

    Parameters
    ----------
    features : pd.DataFrame
        Input features (will be normalised internally).
    weights : dict {col: weight}
        Column names and their relative weights.
    missing_ok : bool
        If True, missing columns are silently ignored.

    Returns
    -------
    pd.Series  in [0, 100].
    """
    score = pd.Series(0.0, index=features.index)
    total_w = 0.0
    for col, w in weights.items():
        if col not in features.columns:
            if not missing_ok:
                raise KeyError(f'Required feature {col} not found')
            continue
        vals = features[col].fillna(features[col].median())
        score += w * _normalise_to_100(vals)
        total_w += w
    if total_w > 0:
        score /= total_w
    return score.clip(0, 100)


# ======================================================================
# Pressure Risk Score
# ======================================================================

def pressure_risk_score(df: pd.DataFrame) -> pd.Series:
    """Pressure_Risk_Score ∈ [0, 100]."""
    cols = {
        'pressure_difference_between_tyres': 1.5,
        'pressure_drop_rate':                1.5,
        'underinflation_flag':               2.0,
        'overinflation_flag':                1.5,
        'pressure_anomaly_flag':             1.5,
        'pressure_zscore':                   2.0,
    }
    return _composite_score(df, cols)


# ======================================================================
# Thermal Risk Score
# ======================================================================

def thermal_risk_score(df: pd.DataFrame) -> pd.Series:
    """Thermal_Risk_Score ∈ [0, 100]."""
    cols = {
        'temperature_mean':                   1.5,
        'temperature_max':                    1.5,
        'temperature_rise_rate':              2.0,
        'temperature_difference_between_tyres': 1.5,
        'overtemperature_flag':               2.0,
        'temperature_zscore':                 1.5,
    }
    return _composite_score(df, cols)


# ======================================================================
# Load Risk Score
# ======================================================================

def load_risk_score(df: pd.DataFrame) -> pd.Series:
    """Load_Risk_Score ∈ [0, 100]."""
    cols = {
        'payload_mean':                1.5,
        'payload_std':                 1.0,
        'payload_max':                 1.5,
        'payload_change_rate':         1.5,
        'payload_percent_of_max':      1.5,
        'overload_flag':               2.0,
        'cumulative_payload_exposure': 1.0,
    }
    return _composite_score(df, cols)


# ======================================================================
# Vibration Risk Score
# ======================================================================

def vibration_risk_score(df: pd.DataFrame) -> pd.Series:
    """Vibration_Risk_Score ∈ [0, 100]."""
    cols = {
        'acceleration_magnitude': 1.5,
        'peak_acceleration':      1.5,
        'rms_vibration':          2.0,
        'crest_factor':           1.5,
        'impact_event_flag':      2.0,
        'high_vibration_flag':    1.5,
    }
    return _composite_score(df, cols)


# ======================================================================
# Braking Risk Score
# ======================================================================

def braking_risk_score(df: pd.DataFrame) -> pd.Series:
    """Braking_Risk_Score ∈ [0, 100]."""
    cols = {
        'brake_frequency':        2.0,
        'hard_braking_flag':      2.0,
        'brake_duration':         1.5,
        'cumulative_braking_time': 1.5,
    }
    return _composite_score(df, cols)


# ======================================================================
# Terrain Risk Score
# ======================================================================

def terrain_risk_score(df: pd.DataFrame) -> pd.Series:
    """Terrain_Risk_Score ∈ [0, 100]."""
    cols = {
        'pitch_std':          2.0,
        'roll_std':           2.0,
        'slope_event_flag':   2.0,
        'rough_terrain_flag': 2.0,
    }
    return _composite_score(df, cols)


# ======================================================================
# Usage Risk Score
# ======================================================================

def usage_risk_score(df: pd.DataFrame) -> pd.Series:
    """Usage_Risk_Score ∈ [0, 100]."""
    cols = {
        'distance_travelled':  1.5,
        'cumulative_distance': 1.5,
        'speed_mean':          1.5,
        'speed_max':           1.5,
        'high_speed_flag':     2.0,
    }
    return _composite_score(df, cols)


# ======================================================================
# Convenience: compute all scores at once
# ======================================================================

ALL_SCORE_FUNCS = {
    'Pressure_Risk_Score':    pressure_risk_score,
    'Thermal_Risk_Score':     thermal_risk_score,
    'Load_Risk_Score':        load_risk_score,
    'Vibration_Risk_Score':   vibration_risk_score,
    'Braking_Risk_Score':     braking_risk_score,
    'Terrain_Risk_Score':     terrain_risk_score,
    'Usage_Risk_Score':       usage_risk_score,
}


def compute_all_risk_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with all 7 risk component scores (plus timestamp)."""
    scores = pd.DataFrame({'timestamp': df['timestamp'].values})
    for name, func in ALL_SCORE_FUNCS.items():
        scores[name] = func(df)
    return scores
