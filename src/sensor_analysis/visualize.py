"""
sensor_analysis/visualize.py
Generate visual exploration figures: missing value heatmap, correlation heatmap,
histograms, boxplots, time-series plots, distribution plots, pair plots.
Usage: python -m src.sensor_analysis.visualize --data-dir datasets/sensors_data --output-dir outputs/sensor_audit
"""

import os
import argparse
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime

warnings.filterwarnings('ignore')

plt.rcParams.update({
    'figure.max_open_warning': 0,
    'font.size': 8,
    'axes.titlesize': 10,
    'axes.labelsize': 8,
})


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


def plot_missing_heatmap(df, fname, out_dir):
    plt.figure(figsize=(12, 2.5))
    sns.heatmap(df.isna(), cbar=False, yticklabels=False, cmap='viridis')
    plt.title(f'Missing Values Heatmap - {fname}')
    plt.tight_layout()
    path = os.path.join(out_dir, f'missing_heatmap_{fname.replace(".csv","")}.png')
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f'  [viz] Saved {path}')


def plot_correlation_heatmap(df, fname, out_dir):
    numeric = df.select_dtypes(include='number')
    if numeric.shape[1] < 2:
        return
    plt.figure(figsize=(max(8, numeric.shape[1] * 0.6), max(6, numeric.shape[1] * 0.5)))
    corr = numeric.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True if numeric.shape[1] < 15 else False,
                cmap='RdBu_r', center=0, fmt='.2f', linewidths=0.5,
                annot_kws={'size': 6})
    plt.title(f'Correlation Heatmap - {fname}')
    plt.tight_layout()
    path = os.path.join(out_dir, f'correlation_heatmap_{fname.replace(".csv","")}.png')
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f'  [viz] Saved {path}')


def plot_histograms(df, fname, out_dir):
    numeric = df.select_dtypes(include='number')
    if numeric.shape[1] == 0:
        return
    ncols = min(4, numeric.shape[1])
    nrows = int(np.ceil(numeric.shape[1] / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = axes.flatten() if nrows * ncols > 1 else [axes]
    for i, col in enumerate(numeric.columns):
        if i < len(axes):
            ax = axes[i]
            data = numeric[col].dropna()
            if len(data) > 0:
                ax.hist(data, bins=50, alpha=0.7, edgecolor='black', linewidth=0.3)
                ax.set_title(col, fontsize=8)
                ax.tick_params(axis='both', labelsize=6)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle(f'Histograms - {fname}', fontsize=12)
    plt.tight_layout()
    path = os.path.join(out_dir, f'histograms_{fname.replace(".csv","")}.png')
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f'  [viz] Saved {path}')


def plot_boxplots(df, fname, out_dir):
    numeric = df.select_dtypes(include='number')
    if numeric.shape[1] == 0:
        return
    ncols = min(4, numeric.shape[1])
    nrows = int(np.ceil(numeric.shape[1] / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = axes.flatten() if nrows * ncols > 1 else [axes]
    for i, col in enumerate(numeric.columns):
        if i < len(axes):
            ax = axes[i]
            data = numeric[col].dropna()
            if len(data) > 0:
                ax.boxplot(data, vert=False, patch_artist=True)
                ax.set_title(col, fontsize=8)
                ax.tick_params(axis='both', labelsize=6)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle(f'Boxplots - {fname}', fontsize=12)
    plt.tight_layout()
    path = os.path.join(out_dir, f'boxplots_{fname.replace(".csv","")}.png')
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f'  [viz] Saved {path}')


def plot_time_series(df, fname, out_dir):
    ts_col = find_timestamp_column(df)
    if ts_col is None:
        return
    numeric = df.select_dtypes(include='number').columns
    if len(numeric) == 0:
        return

    # Sample if too large
    if len(df) > 50000:
        plot_df = df.sample(n=50000, random_state=42).sort_values(ts_col)
    else:
        plot_df = df.sort_values(ts_col)

    try:
        time = pd.to_datetime(plot_df[ts_col], errors='coerce')
        plot_df = plot_df.assign(_time=time).dropna(subset=['_time'])
    except:
        return

    n_plots = min(len(numeric), 9)
    ncols = 3
    nrows = int(np.ceil(n_plots / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows))
    axes = axes.flatten() if nrows * ncols > 1 else [axes]

    for i, col in enumerate(numeric[:n_plots]):
        ax = axes[i]
        data = plot_df[['_time', col]].dropna()
        if len(data) > 0:
            ax.plot(data['_time'], data[col], linewidth=0.3, alpha=0.7)
            ax.set_title(col, fontsize=8)
            ax.tick_params(axis='x', rotation=45, labelsize=5)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle(f'Time Series (sampled {len(plot_df):,} pts) - {fname}', fontsize=12)
    plt.tight_layout()
    path = os.path.join(out_dir, f'timeseries_{fname.replace(".csv","")}.png')
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f'  [viz] Saved {path}')


def plot_distributions(df, fname, out_dir):
    numeric = df.select_dtypes(include='number')
    if numeric.shape[1] == 0:
        return
    ncols = min(4, numeric.shape[1])
    nrows = int(np.ceil(numeric.shape[1] / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = axes.flatten() if nrows * ncols > 1 else [axes]
    for i, col in enumerate(numeric.columns):
        if i < len(axes):
            ax = axes[i]
            data = numeric[col].dropna()
            if len(data) > 0:
                sns.kdeplot(data, ax=ax, fill=True, alpha=0.6)
                ax.set_title(col, fontsize=8)
                ax.tick_params(axis='both', labelsize=6)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle(f'Distribution (KDE) Plots - {fname}', fontsize=12)
    plt.tight_layout()
    path = os.path.join(out_dir, f'distributions_{fname.replace(".csv","")}.png')
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f'  [viz] Saved {path}')


def plot_pairplot(df, fname, out_dir):
    numeric = df.select_dtypes(include='number')
    if numeric.shape[1] < 2 or numeric.shape[1] > 8:
        return
    # Sample if too large
    if len(df) > 10000:
        sample_df = df.sample(n=10000, random_state=42)
    else:
        sample_df = df
    numeric_sample = sample_df.select_dtypes(include='number')
    try:
        g = sns.pairplot(numeric_sample, diag_kind='kde', plot_kws={'alpha': 0.3, 's': 3})
        g.fig.suptitle(f'Pairplot - {fname}', y=1.02, fontsize=12)
        path = os.path.join(out_dir, f'pairplot_{fname.replace(".csv","")}.png')
        g.savefig(path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f'  [viz] Saved {path}')
    except Exception as e:
        print(f'  [viz] Pairplot skipped: {e}')


def visualize_all(files, output_dir):
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    for fname, fpath in files:
        print(f'[viz] Processing {fname}...')
        df = pd.read_csv(fpath, low_memory=False)

        plot_missing_heatmap(df, fname, figures_dir)
        plot_correlation_heatmap(df, fname, figures_dir)
        plot_histograms(df, fname, figures_dir)
        plot_boxplots(df, fname, figures_dir)
        plot_time_series(df, fname, figures_dir)
        plot_distributions(df, fname, figures_dir)
        plot_pairplot(df, fname, figures_dir)

    print('[viz] All visualizations generated.')


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

    csv_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])
    files = [(f, os.path.join(data_dir, f)) for f in csv_files]
    visualize_all(files, output_dir)


if __name__ == '__main__':
    main()
