"""
Telemetry Loader — reusable module for OSI pipeline phases.

Provides convenience functions to load the master telemetry
dataset and the raw sensor CSVs.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path('D:/Tyre_Classification')
TELEMETRY_DIR = ROOT / 'datasets/telemetry'
SENSORS_DIR = ROOT / 'datasets/sensors_data'
CHUNK_SIZE = 100_000


def load_master_parquet(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the master telemetry parquet file."""
    path = path or TELEMETRY_DIR / 'telemetry_master.parquet'
    return pd.read_parquet(path)


def load_master_sample(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the sample CSV (first 5000 rows)."""
    path = path or TELEMETRY_DIR / 'telemetry_master_sample.csv'
    return pd.read_csv(path, parse_dates=['timestamp'])


def load_payload(path: Optional[Path] = None,
                 chunksize: Optional[int] = None) -> pd.DataFrame:
    """Load payload CSV with timestamp parsing."""
    path = path or SENSORS_DIR / 'payload_processed.csv'
    return pd.read_csv(path,
                       parse_dates=['sample_timestamp'],
                       chunksize=chunksize)


def load_tpms(path: Optional[Path] = None,
              chunksize: Optional[int] = None) -> pd.DataFrame:
    """Load TPMS CSV with timestamp parsing.

    The TPMS timestamp column is ``_col3`` and is standardised to
    second-precision (milliseconds stripped) with column renamed to
    ``timestamp`` when using chunked mode.
    """
    path = path or SENSORS_DIR / 'tpms_processed.csv'
    if chunksize:
        chunks = []
        for chunk in pd.read_csv(path, chunksize=chunksize,
                                 parse_dates=['_col3']):
            chunk.rename(columns={'_col3': 'timestamp'}, inplace=True)
            chunk['timestamp'] = chunk['timestamp'].dt.floor('s')
            chunks.append(chunk)
        return pd.concat(chunks, ignore_index=True)
    df = pd.read_csv(path, parse_dates=['_col3'])
    df.rename(columns={'_col3': 'timestamp'}, inplace=True)
    df['timestamp'] = df['timestamp'].dt.floor('s')
    return df


# Columns of interest
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
