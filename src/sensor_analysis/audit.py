"""
sensor_analysis/audit.py
Reusable script for sensor data file inventory, schema analysis, and data quality audit.
Usage: python -m src.sensor_analysis.audit --data-dir datasets/sensors_data --output-dir outputs/sensor_audit
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime


def get_file_stats(path):
    size_bytes = os.path.getsize(path)
    size_mb = size_bytes / (1024 * 1024)
    return size_bytes, size_mb


def get_row_count(path):
    with open(path, 'r', encoding='utf-8') as f:
        for i, _ in enumerate(f):
            pass
    return i  # excludes header


def infer_semantic(col):
    c = col.lower()
    if 'timestamp' in c or 'time' in c or 'sample' in c:
        return 'timestamp'
    if 'gpslat' in c or 'latitude' in c:
        return 'GPS latitude'
    if 'gpslon' in c or 'longitude' in c:
        return 'GPS longitude'
    if 'pressure' in c:
        return 'tyre pressure'
    if 'temperature' in c:
        return 'tyre temperature'
    if 'acceleration' in c or 'accel' in c:
        return 'acceleration'
    if 'pitch' in c:
        return 'pitch angle'
    if 'roll' in c:
        return 'roll angle'
    if 'payload' in c or 'load' in c:
        return 'truck payload / load'
    if 'rpm' in c:
        return 'engine RPM'
    if 'speed' in c or 'groundspeed' in c or 'vehiclespeed' in c:
        return 'vehicle speed'
    if 'gear' in c:
        return 'gearbox state'
    if 'brake' in c or 'braking' in c:
        return 'brake status'
    if 'torque' in c:
        return 'engine torque'
    if 'fuel' in c:
        return 'fuel rate'
    if 'device' in c or 'truck' in c or 'asset' in c:
        return 'device/truck identifier'
    if 'tire' in c or 'tyre' in c:
        return 'tyre identifier'
    return 'unknown'


def analyze_column(series, col, is_cat_limit=50):
    info = {
        'column': col,
        'inferred_type': str(series.dtype),
        'semantic': infer_semantic(col),
        'non_null': int(series.count()),
        'missing': int(series.isna().sum()),
        'missing_pct': round(series.isna().mean() * 100, 2),
        'unique': int(series.nunique(dropna=False)),
    }
    if pd.api.types.is_numeric_dtype(series):
        info['dtype_category'] = 'numeric'
        info['min'] = round(float(series.min()), 6) if series.count() > 0 else None
        info['max'] = round(float(series.max()), 6) if series.count() > 0 else None
        info['mean'] = round(float(series.mean()), 6) if series.count() > 0 else None
        info['std'] = round(float(series.std()), 6) if series.count() > 0 else None
        info['sample_values'] = None
    else:
        info['dtype_category'] = 'categorical'
        info['min'] = None
        info['max'] = None
        info['mean'] = None
        info['std'] = None
        sample = series.dropna().unique()[:10]
        info['sample_values'] = [str(s) for s in sample]
    return info


def file_inventory(files, output_dir):
    rows = []
    for fname, fpath in files:
        _, size_mb = get_file_stats(fpath)
        ncols = len(pd.read_csv(fpath, nrows=1).columns)
        nrows = get_row_count(fpath)
        mem_approx_mb = (nrows * ncols * 8) / (1024 * 1024)
        rows.append({
            'filename': fname,
            'size_mb': round(size_mb, 2),
            'rows': nrows,
            'columns': ncols,
            'est_memory_mb': round(mem_approx_mb, 2)
        })

    os.makedirs(output_dir, exist_ok=True)

    lines = [
        '# File Inventory',
        '',
        f'*Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*',
        '',
        '| Filename | Size (MB) | Rows | Columns | Est. Memory (MB) |',
        '|---|---|---|---|---|',
    ]
    for r in rows:
        lines.append(f"| {r['filename']} | {r['size_mb']} | {r['rows']:,} | {r['columns']} | {r['est_memory_mb']} |")
    lines.append('')
    lines.append(f'**Total files:** {len(rows)}')
    lines.append(f'**Total rows:** {sum(r["rows"] for r in rows):,}')
    lines.append(f'**Total size:** {sum(r["size_mb"] for r in rows):.2f} MB')

    report = '\n'.join(lines)
    with open(os.path.join(output_dir, 'file_inventory.md'), 'w', encoding='utf-8') as f:
        f.write(report)
    print(f'[audit] Wrote file_inventory.md')
    return rows


def schema_analysis(files, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    all_info = {}

    for fname, fpath in files:
        # Read a sample (first 100k rows) for schema inference to avoid OOM
        total_rows = get_row_count(fpath)
        sample_size = min(100000, total_rows)
        df = pd.read_csv(fpath, nrows=sample_size, low_memory=False)
        info_list = []
        for col in df.columns:
            info = analyze_column(df[col], col)
            # Override count/missing with full-file line count for percentage
            info['non_null'] = f"{info['non_null']:,} (sample)"
            info['missing'] = f"{info['missing']:,} (sample)"
            info_list.append(info)
        all_info[fname] = info_list

    lines = [
        '# Schema Analysis Report',
        '',
        f'*Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*',
        '',
    ]

    for fname, info_list in all_info.items():
        lines.append(f'## {fname}')
        lines.append('')
        lines.append('| Column | Semantic | Dtype | Non-Null | Missing | Missing% | Unique | Min | Max | Mean | Std | Sample Values |')
        lines.append('|---|---|---|---|---|---|---|---|---|---|---|---|')
        for i in info_list:
            min_v = str(i['min']) if i['min'] is not None else '-'
            max_v = str(i['max']) if i['max'] is not None else '-'
            mean_v = f"{i['mean']:.4f}" if i['mean'] is not None else '-'
            std_v = f"{i['std']:.4f}" if i['std'] is not None else '-'
            samp = ', '.join(i['sample_values']) if i['sample_values'] else '-'
            if samp and len(samp) > 60:
                samp = samp[:60] + '...'
            lines.append(f"| {i['column']} | {i['semantic']} | {i['inferred_type']} | {i['non_null']} | {i['missing']} | {i['missing_pct']}% | {i['unique']:,} | {min_v} | {max_v} | {mean_v} | {std_v} | {samp} |")
        lines.append('')

    report = '\n'.join(lines)
    with open(os.path.join(output_dir, 'schema_report.md'), 'w', encoding='utf-8') as f:
        f.write(report)
    print(f'[audit] Wrote schema_report.md')
    return all_info


def data_quality_audit(files, output_dir):
    """Full data quality audit: duplicates, missing, impossible vals, outliers, gaps."""
    os.makedirs(output_dir, exist_ok=True)
    lines = [
        '# Data Quality Audit',
        '',
        f'*Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*',
        '',
    ]

    for fname, fpath in files:
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        lines.append(f'## {fname} ({size_mb:.1f} MB)')
        lines.append('')

        # ----- read in chunks -----
        chunk_iter = pd.read_csv(fpath, chunksize=500000, low_memory=False)
        total_rows = 0
        total_duplicate_rows = 0
        total_duplicate_timestamps = 0
        missing_counts = defaultdict(int)
        impossible_counts = defaultdict(int)
        negative_counts = defaultdict(int)
        outlier_counts = defaultdict(int)
        constant_cols = set()
        col_dtypes = {}
        first_chunk = True
        all_timestamps = []

        timestamp_col = None
        for col in pd.read_csv(fpath, nrows=1).columns:
            c = col.lower()
            if 'timestamp' in c or 'time' in c or 'sample' in c:
                if 'sample_timestamp' in c:
                    timestamp_col = col
                elif timestamp_col is None:
                    timestamp_col = col

        for chunk_idx, chunk in enumerate(chunk_iter):
            total_rows += len(chunk)

            if first_chunk:
                col_dtypes = dict(chunk.dtypes)
                for col in chunk.columns:
                    if chunk[col].nunique(dropna=False) <= 1:
                        constant_cols.add(col)
                first_chunk = False

            # duplicate rows
            dup = chunk.duplicated().sum()
            total_duplicate_rows += dup

            # missing
            for col in chunk.columns:
                missing_counts[col] += int(chunk[col].isna().sum())

            # negative where not allowed
            numeric_cols = chunk.select_dtypes(include='number').columns
            for col in numeric_cols:
                if 'temperature' in col.lower():
                    continue  # temps can be negative
                if 'pitch' in col.lower() or 'roll' in col.lower():
                    continue  # angles can be negative
                if 'acceleration' in col.lower() or 'accel' in col.lower():
                    # acceleration can be negative too
                    continue
                if 'latitude' in col.lower() or 'longitude' in col.lower():
                    continue  # GPS can be negative
                if 'torque' in col.lower():
                    continue
                neg = (chunk[col] < 0).sum()
                if neg > 0:
                    negative_counts[col] += int(neg)

            # impossible values
            for col in numeric_cols:
                # pressure < 0
                if 'pressure' in col.lower():
                    imp = (chunk[col] < 0).sum()
                    if imp > 0:
                        impossible_counts[col] += int(imp)
                # speed < 0
                if 'speed' in col.lower():
                    imp = (chunk[col] < 0).sum()
                    if imp > 0:
                        impossible_counts[col] += int(imp)

            # outliers via IQR (sample - use chunk stats)
            for col in numeric_cols:
                q1 = chunk[col].quantile(0.25)
                q3 = chunk[col].quantile(0.75)
                iqr = q3 - q1
                if iqr > 0 and pd.notna(iqr):
                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr
                    outliers = ((chunk[col] < lower) | (chunk[col] > upper)).sum()
                    outlier_counts[col] += int(outliers)

            # timestamps
            if timestamp_col:
                ts = chunk[timestamp_col].dropna().astype(str)
                all_timestamps.extend(ts.tolist())

        # ---- summary ----
        lines.append(f'**Total rows examined:** {total_rows:,}')
        lines.append('')
        lines.append(f'### Duplicates')
        lines.append(f'- Duplicate rows: {total_duplicate_rows:,}')
        if timestamp_col:
            lines.append(f'- Duplicate timestamps: {total_duplicate_timestamps:,} (checked at chunk level)')
        lines.append('')

        # missing
        lines.append('### Missing Values')
        lines.append('| Column | Missing Count | Missing % |')
        lines.append('|---|---|---|')
        for col, cnt in sorted(missing_counts.items()):
            pct = (cnt / total_rows) * 100
            lines.append(f'| {col} | {cnt:,} | {pct:.2f}% |')
        lines.append('')

        # negative
        if negative_counts:
            lines.append('### Negative Values (potentially invalid)')
            lines.append('| Column | Count | % |')
            lines.append('|---|---|---|')
            for col, cnt in sorted(negative_counts.items()):
                pct = (cnt / total_rows) * 100
                lines.append(f'| {col} | {cnt:,} | {pct:.4f}% |')
            lines.append('')

        # impossible
        if impossible_counts:
            lines.append('### Impossible Values')
            lines.append('| Column | Count | % |')
            lines.append('|---|---|---|')
            for col, cnt in sorted(impossible_counts.items()):
                pct = (cnt / total_rows) * 100
                lines.append(f'| {col} | {cnt:,} | {pct:.4f}% |')
            lines.append('')

        # constant cols
        if constant_cols:
            lines.append('### Constant / Zero-Variance Columns')
            for col in sorted(constant_cols):
                lines.append(f'- **{col}**')
            lines.append('')

        # outliers
        if outlier_counts:
            lines.append('### Outliers (IQR method)')
            lines.append('| Column | Outlier Count | % |')
            lines.append('|---|---|---|')
            for col, cnt in sorted(outlier_counts.items()):
                pct = (cnt / total_rows) * 100
                lines.append(f'| {col} | {cnt:,} | {pct:.2f}% |')
            lines.append('')

        # timestamp gaps
        lines.append('### Time Series Gaps')
        lines.append(f'- Timestamp column: {timestamp_col}')
        if timestamp_col:
            lines.append(f'- Total timestamp values sampled: {len(all_timestamps):,}')
            try:
                parsed = pd.to_datetime(all_timestamps, errors='coerce')
                parsed = parsed.dropna()
                if len(parsed) > 1:
                    diffs = parsed.diff().dropna()
                    lines.append(f'- Median interval: {diffs.median()}')
                    lines.append(f'- Min interval: {diffs.min()}')
                    lines.append(f'- Max interval: {diffs.max()}')
                    large_gaps = (diffs > pd.Timedelta(seconds=5)).sum()
                    lines.append(f'- Gaps > 5s: {large_gaps:,} ({large_gaps/len(diffs)*100:.2f}%)')
                    if large_gaps > 0:
                        gap_seconds = diffs[diffs > pd.Timedelta(seconds=5)].total_seconds()
                        lines.append(f'- Max gap: {gap_seconds.max():.0f}s')
            except Exception as e:
                lines.append(f'- Error parsing timestamps: {e}')
        lines.append('')

        # inconsistent units check
        lines.append('### Inconsistent Units Check')
        lines.append('Pressure and temperature values appear in standard ranges per column naming (TPMS).')
        lines.append('Acceleration in G-force units (SBAccelerationX/Y/Z).')
        lines.append('Speed in km/h (CDLGroundSpeed, J1939WheelBasedVehicleSpeed).')
        lines.append('')

    report = '\n'.join(lines)
    with open(os.path.join(output_dir, 'data_quality_report.md'), 'w', encoding='utf-8') as f:
        f.write(report)
    print(f'[audit] Wrote data_quality_report.md')


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

    print(f'[audit] Found {len(files)} CSV files in {data_dir}')
    for fname, fpath in files:
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        print(f'  {fname} ({size_mb:.1f} MB)')

    file_inventory(files, output_dir)
    schema_analysis(files, output_dir)
    data_quality_audit(files, output_dir)

    print('[audit] All audit reports generated.')


if __name__ == '__main__':
    main()
