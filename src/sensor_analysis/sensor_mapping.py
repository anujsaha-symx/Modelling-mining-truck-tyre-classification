"""
sensor_analysis/sensor_mapping.py
Map columns to physical sensors and assess usability for tyre severity / RUL.
Usage: python -m src.sensor_analysis.sensor_mapping --data-dir datasets/sensors_data --output-dir outputs/sensor_audit
"""

import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict


SENSOR_PATTERNS = {
    '3-axis accelerometer': ['acceleration', 'accel'],
    'pressure sensor (TPMS)': ['pressure'],
    'temperature sensor (TPMS)': ['temperature'],
    'speed sensor': ['speed', 'groundspeed', 'vehiclespeed'],
    'RPM sensor': ['rpm'],
    'load/payload sensor': ['payload', 'load'],
    'GPS receiver': ['gps', 'latitude', 'longitude'],
    'brake sensor': ['brake', 'braking'],
    'gear sensor': ['gear'],
    'torque sensor': ['torque'],
    'fuel rate sensor': ['fuel'],
    'pitch/roll sensor': ['pitch', 'roll'],
}


def map_columns(files, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    sensor_map = defaultdict(list)

    for fname, fpath in files:
        df = pd.read_csv(fpath, low_memory=False)
        for col in df.columns:
            c = col.lower()
            mapped = False
            for sensor_type, patterns in SENSOR_PATTERNS.items():
                if any(p in c for p in patterns):
                    stats = {
                        'column': col,
                        'source_file': fname,
                        'dtype': str(df[col].dtype),
                        'non_null': int(df[col].count()),
                        'missing_pct': round(df[col].isna().mean() * 100, 2),
                    }
                    if pd.api.types.is_numeric_dtype(df[col]):
                        stats['min'] = round(float(df[col].min()), 6)
                        stats['max'] = round(float(df[col].max()), 6)
                        stats['mean'] = round(float(df[col].mean()), 6)
                        stats['std'] = round(float(df[col].std()), 6)
                    else:
                        stats['sample'] = [str(s) for s in df[col].dropna().unique()[:5]]
                    sensor_map[sensor_type].append(stats)
                    mapped = True
            if not mapped:
                # unknown columns
                pass

    lines = [
        '# Sensor Mapping Report',
        '',
        f'*Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*',
        '',
        '## Summary of Detected Sensors',
        '',
        '| Sensor Type | Columns Found | Source File |',
        '|---|---|---|',
    ]
    for s_type, cols in sorted(sensor_map.items()):
        fnames = set(c['source_file'] for c in cols)
        lines.append(f"| {s_type} | {len(cols)} | {', '.join(sorted(fnames))} |")

    lines.append('')
    lines.append('---')
    lines.append('')

    for s_type, cols in sorted(sensor_map.items()):
        lines.append(f'## {s_type}')
        lines.append('')

        for c in cols:
            lines.append(f'### `{c["column"]}` (in {c["source_file"]})')
            lines.append('')
            lines.append(f'- **Dtype:** {c["dtype"]}')
            lines.append(f'- **Non-null:** {c["non_null"]:,}')
            lines.append(f'- **Missing:** {c["missing_pct"]}%')
            if 'min' in c:
                lines.append(f'- **Range:** [{c["min"]}, {c["max"]}]')
                lines.append(f'- **Mean ± Std:** {c["mean"]} ± {c["std"]}')
            if 'sample' in c:
                lines.append(f'- **Sample values:** {c["sample"]}')

            # per-sensor reasoning
            lines.append('')
            if 'acceleration' in s_type.lower() or 'accel' in c['column'].lower():
                lines.append(f'- **Reasoning:** Column name contains "Acceleration" or similar. Likely G-force (±{c.get("max", "?")} range).')
                lines.append(f'- **Inferred units:** G (9.81 m/s²)')
                lines.append(f'- **Use in tyre severity:** Measures vibration/impact; high-freq events correlate with road hazards, potholes, curb strikes.')
            elif 'pressure' in s_type.lower():
                lines.append(f'- **Reasoning:** Column name explicitly references tyre pressure.')
                lines.append(f'- **Inferred units:** psi or bar (check TPMS standard)')
                lines.append(f'- **Use in tyre severity:** Under-inflation causes overheating, faster wear, blowout risk. Key severity indicator.')
            elif 'temperature' in s_type.lower():
                lines.append(f'- **Reasoning:** Column name explicitly references tyre temperature.')
                lines.append(f'- **Inferred units:** °C or °F')
                lines.append(f'- **Use in tyre severity:** Overheating indicates friction, under-inflation, overloading. Key severity indicator.')
            elif 'speed' in s_type.lower():
                lines.append(f'- **Reasoning:** Column name references vehicle speed (ground speed or wheel-based).')
                lines.append(f'- **Inferred units:** km/h')
                lines.append(f'- **Use in tyre severity:** Context for load/impact analysis; high-speed events more damaging.')
            elif 'rpm' in s_type.lower():
                lines.append(f'- **Reasoning:** Column name references engine RPM.')
                lines.append(f'- **Inferred units:** RPM')
                lines.append(f'- **Use in tyre severity:** Correlated with vehicle speed and gear; useful context.')
            elif 'payload' in s_type.lower() or 'load' in s_type.lower():
                lines.append(f'- **Reasoning:** Column name references truck payload/load.')
                lines.append(f'- **Inferred units:** kg or tons')
                lines.append(f'- **Use in tyre severity:** Overloading directly causes tyre stress, overheating, blowout risk. Critical for severity.')
            elif 'gps' in s_type.lower() or 'latitude' in c['column'].lower():
                lines.append(f'- **Reasoning:** Column name references GPS coordinates.')
                lines.append(f'- **Use in tyre severity:** Location context; road type, terrain, route conditions.')
            elif 'brake' in s_type.lower():
                lines.append(f'- **Reasoning:** Column name references brake system.')
                lines.append(f'- **Use in tyre severity:** Harsh braking events contribute to flat-spotting and uneven wear.')
            elif 'gear' in s_type.lower():
                lines.append(f'- **Reasoning:** Column name references gear state.')
                lines.append(f'- **Use in tyre severity:** Driving context; low gear + high RPM indicates loaded/heavy operation.')
            elif 'torque' in s_type.lower():
                lines.append(f'- **Reasoning:** Column name references engine/retarder torque.')
                lines.append(f'- **Inferred units:** % or Nm')
                lines.append(f'- **Use in tyre severity:** High torque = high stress on drive tyres.')
            elif 'fuel' in s_type.lower():
                lines.append(f'- **Reasoning:** Column name references fuel rate.')
                lines.append(f'- **Inferred units:** L/h or similar')
                lines.append(f'- **Use in tyre severity:** Correlated with engine load.')
            elif 'pitch' in s_type.lower() or 'roll' in c['column'].lower():
                lines.append(f'- **Reasoning:** Column name references pitch/roll angle.')
                lines.append(f'- **Inferred units:** degrees')
                lines.append(f'- **Use in tyre severity:** Vehicle attitude; roll indicates cornering severity, pitch indicates braking/acceleration.')
            lines.append('')

    # also note unmapped columns
    lines.append('## Unidentified / Non-Sensor Columns')
    lines.append('')
    for fname, fpath in files:
        df = pd.read_csv(fpath, low_memory=False)
        all_mapped_cols = set()
        for cols in sensor_map.values():
            for c in cols:
                all_mapped_cols.add(c['column'])
        unidentified = [c for c in df.columns if c not in all_mapped_cols]
        if unidentified:
            lines.append(f'**{fname}:** {", ".join(unidentified)}')
            lines.append('')
        for c in unidentified:
            lines.append(f'- `{c}` appears to be a device/asset identifier, timestamp, or administrative column (not a sensor reading).')
        lines.append('')

    report = '\n'.join(lines)
    with open(os.path.join(output_dir, 'sensor_mapping.md'), 'w', encoding='utf-8') as f:
        f.write(report)
    print(f'[sensor_mapping] Wrote sensor_mapping.md')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='datasets/sensors_data')
    parser.add_argument('--output-dir', default='outputs/sensor_audit')
    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir

    if not os.path.isabs(data_dir):
        data_dir = os.path.join(os.getcwd(), data_dir)
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(os.getcwd(), output_dir)

    os.makedirs(output_dir, exist_ok=True)
    csv_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])
    files = [(f, os.path.join(data_dir, f)) for f in csv_files]

    map_columns(files, output_dir)


if __name__ == '__main__':
    main()
