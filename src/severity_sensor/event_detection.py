"""
Event detection and counters for OSI Phase 2.

Flags events based on domain-specific thresholds and accumulates
cumulative counters over the full dataset timeline.
"""

from typing import List, Optional, Dict, Tuple
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Default thresholds (can be overridden per call)
# ---------------------------------------------------------------------------
DEFAULT_THRESHOLDS: Dict[str, float] = {
    'underinflation_bar': 500.0,
    'overinflation_bar': 800.0,
    'overtemperature_bar': 50.0,
    'overload_bar': 90.0,
    'hard_brake_bar': 50.0,
    'impact_bar': 0.5,
    'high_vibration_bar': 0.15,
    'slope_bar': 10.0,
    'rough_terrain_bar': 2.0,
    'idle_speed_bar': 1.0,
}


# ======================================================================
# Flag helpers
# ======================================================================

def flag_underinflation(
    pressures: pd.DataFrame,
    threshold: float = DEFAULT_THRESHOLDS['underinflation_bar'],
) -> pd.DataFrame:
    """Flag rows where any tyre pressure < threshold."""
    result = (pressures < threshold).any(axis=1).astype(int)
    return result.to_frame('underinflation_flag')


def flag_overinflation(
    pressures: pd.DataFrame,
    threshold: float = DEFAULT_THRESHOLDS['overinflation_bar'],
) -> pd.DataFrame:
    """Flag rows where any tyre pressure > threshold."""
    result = (pressures > threshold).any(axis=1).astype(int)
    return result.to_frame('overinflation_flag')


def flag_overtemperature(
    temperatures: pd.DataFrame,
    threshold: float = DEFAULT_THRESHOLDS['overtemperature_bar'],
) -> pd.DataFrame:
    """Flag rows where any tyre temperature > threshold."""
    result = (temperatures > threshold).any(axis=1).astype(int)
    return result.to_frame('overtemperature_flag')


def flag_overload(
    payload: pd.Series,
    threshold: float = DEFAULT_THRESHOLDS['overload_bar'],
) -> pd.DataFrame:
    """Flag rows where payload exceeds threshold."""
    result = (payload > threshold).astype(int)
    return result.to_frame('overload_flag')


def flag_hard_braking(
    brake_position: pd.Series,
    threshold: float = DEFAULT_THRESHOLDS['hard_brake_bar'],
) -> pd.DataFrame:
    """Flag rows where brake pedal position exceeds threshold."""
    result = (brake_position > threshold).astype(int)
    return result.to_frame('hard_braking_flag')


def flag_impact(
    accel_magnitude: pd.Series,
    threshold: float = DEFAULT_THRESHOLDS['impact_bar'],
) -> pd.DataFrame:
    """Flag rows where acceleration magnitude exceeds threshold."""
    result = (accel_magnitude > threshold).astype(int)
    return result.to_frame('impact_event_flag')


def flag_high_vibration(
    rms_vibration: pd.Series,
    threshold: float = DEFAULT_THRESHOLDS['high_vibration_bar'],
) -> pd.DataFrame:
    """Flag rows where RMS vibration exceeds threshold."""
    result = (rms_vibration > threshold).astype(int)
    return result.to_frame('high_vibration_flag')


def flag_idle(
    speed: pd.Series,
    threshold: float = DEFAULT_THRESHOLDS['idle_speed_bar'],
) -> pd.DataFrame:
    """Flag rows where vehicle speed indicates idle."""
    result = (speed < threshold).astype(int)
    return result.to_frame('idle_flag')


def flag_high_speed(
    speed: pd.Series,
    threshold: float = 30.0,
) -> pd.DataFrame:
    """Flag rows where speed exceeds threshold (m/s → km/h ≈ *3.6)."""
    result = (speed > threshold).astype(int)
    return result.to_frame('high_speed_flag')


def flag_slope_event(
    pitch: pd.Series,
    threshold: float = DEFAULT_THRESHOLDS['slope_bar'],
) -> pd.DataFrame:
    """Flag rows where pitch angle exceeds threshold."""
    result = (pitch.abs() > threshold).astype(int)
    return result.to_frame('slope_event_flag')


def flag_rough_terrain(
    roll_std: pd.Series,
    threshold: float = DEFAULT_THRESHOLDS['rough_terrain_bar'],
) -> pd.DataFrame:
    """Flag rows where rolling std of roll angle exceeds threshold."""
    result = (roll_std > threshold).astype(int)
    return result.to_frame('rough_terrain_flag')


def flag_pressure_anomaly(
    zscores: pd.DataFrame,
    threshold: float = 3.0,
) -> pd.DataFrame:
    """Flag rows where any tyre pressure z-score > threshold."""
    result = (zscores.abs() > threshold).any(axis=1).astype(int)
    return result.to_frame('pressure_anomaly_flag')


# ======================================================================
# Stopped detection from GPS
# ======================================================================

def detect_stopped(
    distance_increment: pd.Series,
    threshold: float = 0.0,
) -> pd.DataFrame:
    """Flag rows where GPS distance increment is ~0 (vehicle stopped)."""
    result = (distance_increment <= threshold).astype(int)
    return result.to_frame('stopped_flag')


# ======================================================================
# Cumulative event counters
# ======================================================================

def compute_cumulative_counters(
    df: pd.DataFrame,
    flag_columns: List[str],
) -> pd.DataFrame:
    """Compute cumulative sums for event flag columns.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain the flag columns (integers 0/1).
    flag_columns : List[str]
        Column names for which to compute cumulative counters.

    Returns
    -------
    pd.DataFrame
        New columns named ``num_{flag_name}`` for each flag column.
    """
    result = {}
    for col in flag_columns:
        counter_name = f'num_{col}'
        result[counter_name] = df[col].fillna(0).astype(int).cumsum()
    return pd.DataFrame(result, index=df.index)
