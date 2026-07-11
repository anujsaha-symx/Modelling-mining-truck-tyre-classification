"""
sensor_analysis/time_series_analysis.py
Analyze timestamp columns, sampling frequency, regularity, duration, number of devices.
Usage: python -m src.sensor_analysis.time_series_analysis --data-dir datasets/sensors_data --output-dir outputs/sensor_audit
"""

import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict


def find_timestamp_column(df):
    for col in df.columns:
        c = col.lower()
        if 'sample_timestamp' in c or '_col3' in c:
            return col
        if ('timestamp' in c or 'time' in c) and 'sample' in c:
            return col
    for col in df.columns:
        c = col.lower()
        if 'timestamp' in c or 'time' in c:
            return col
    return None


def analyze_time_series(files, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    lines = [
        '# Time Series Analysis Report',
        '',
        f'*Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*',
        '',
    ]

    for fname, fpath in files:
        lines.append(f'## {fname}')
        lines.append('')

        df = pd.read_csv(fpath, low_memory=False)
        ts_col = find_timestamp_column(df)

        if ts_col is None:
            lines.append('**No timestamp column detected.**')
            lines.append('')
            continue

        lines.append(f'### Timestamp Column')
        lines.append(f'- Detected column: `{ts_col}`')
        lines.append(f'- Dtype: {df[ts_col].dtype}')
        lines.append('')

        # Parse timestamps
        sample = df[ts_col].dropna().astype(str)
        try:
            parsed = pd.to_datetime(sample, errors='coerce')
            parsed = parsed.dropna()
            n_parsed = len(parsed)
            n_total = len(sample)
            parse_rate = n_parsed / n_total * 100
            lines.append(f'### Parsing')
            lines.append(f'- Successfully parsed: {n_parsed:,} / {n_total:,} ({parse_rate:.2f}%)')
            lines.append('')

            if n_parsed > 1:
                lines.append('### Time Range')
                lines.append(f'- Start: {parsed.min()}')
                lines.append(f'- End: {parsed.max()}')
                duration = parsed.max() - parsed.min()
                lines.append(f'- Duration: {duration}')
                lines.append(f'- Duration (hours): {duration.total_seconds() / 3600:.2f}')
                lines.append(f'- Duration (days): {duration.total_seconds() / 86400:.2f}')
                lines.append('')

                # Sampling frequency
                sorted_ts = parsed.sort_values()
                diffs = sorted_ts.diff().dropna()
                median_diff = diffs.median()
                mean_diff = diffs.mean()
                std_diff = diffs.std()

                lines.append('### Sampling Frequency')
                lines.append(f'- Median interval: {median_diff}')
                lines.append(f'- Mean interval: {mean_diff}')
                lines.append(f'- Std interval: {std_diff}')
                lines.append(f'- Median freq (Hz): {1 / median_diff.total_seconds():.4f}' if median_diff.total_seconds() > 0 else '-')

                # Regular vs irregular
                cv = std_diff.total_seconds() / mean_diff.total_seconds() if mean_diff.total_seconds() > 0 else 0
                regularity = 'Regular' if cv < 0.1 else 'Irregular'
                lines.append(f'- Coefficient of variation: {cv:.4f}')
                lines.append(f'- **Sampling: {regularity}**')
                lines.append('')

                # Gaps
                expected = median_diff
                gaps = diffs[diffs > expected * 2]
                lines.append('### Gaps')
                lines.append(f'- Total intervals: {len(diffs):,}')
                lines.append(f'- Gaps (>2x median): {len(gaps):,} ({len(gaps)/len(diffs)*100:.2f}%)')
                if len(gaps) > 0:
                    lines.append(f'- Max gap: {gaps.max()}')
                    lines.append(f'- Min gap: {gaps.min()}')
                    lines.append(f'- Mean gap duration: {gaps.mean()}')
                lines.append('')
        except Exception as e:
            lines.append(f'**Error parsing timestamps:** {e}')
            lines.append('')

        # Device / truck / sensor identification
        id_cols = [c for c in df.columns if any(x in c.lower() for x in ['device', 'asset', 'truck', 'sensor', 'id'])]
        if id_cols:
            lines.append('### Devices / Entities')
            for c in id_cols:
                unique_vals = df[c].nunique()
                sample_vals = df[c].dropna().unique()[:5]
                lines.append(f'- `{c}`: {unique_vals} unique values')
                if len(sample_vals) > 0:
                    lines.append(f'  - Sample: {[str(s) for s in sample_vals]}')
            lines.append('')

        # Per-device analysis if device col exists
        device_cols = [c for c in df.columns if c.lower() == 'device']
        if device_cols:
            dev_col = device_cols[0]
            devices = df[dev_col].dropna().unique()
            lines.append(f'### Per-Device Summary')
            lines.append(f'| Device | Rows | Start | End |')
            lines.append('|---|---|---|---|')
            for dev in devices:
                sub = df[df[dev_col] == dev]
                if ts_col and ts_col in sub.columns:
                    ts_sub = pd.to_datetime(sub[ts_col].dropna().astype(str), errors='coerce').dropna()
                    start = ts_sub.min() if len(ts_sub) > 0 else 'N/A'
                    end = ts_sub.max() if len(ts_sub) > 0 else 'N/A'
                else:
                    start, end = 'N/A', 'N/A'
                lines.append(f'| {dev} | {len(sub):,} | {start} | {end} |')
            lines.append('')

        lines.append('---')
        lines.append('')

    report = '\n'.join(lines)
    with open(os.path.join(output_dir, 'time_series_report.md'), 'w', encoding='utf-8') as f:
        f.write(report)
    print(f'[time_series] Wrote time_series_report.md')


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

    analyze_time_series(files, output_dir)


if __name__ == '__main__':
    main()
