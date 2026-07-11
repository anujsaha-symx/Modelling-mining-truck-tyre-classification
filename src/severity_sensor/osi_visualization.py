"""
OSI Phase 4: OSI Visualizations
=================================
Generates all figures for the OSI Phase 4 reports.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def plot_osi_distribution(df: pd.DataFrame, osi_cols: list,
                          save_path: Path, dpi: int = 150):
    """Distribution histogram of each OSI variant."""
    n = len(osi_cols)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n))
    if n == 1:
        axes = [axes]
    for ax, col in zip(axes, osi_cols):
        if col not in df.columns:
            continue
        vals = df[col].dropna().values
        ax.hist(vals, bins=80, edgecolor='none', alpha=0.7, color='steelblue')
        ax.axvline(25, color='green', linestyle='--', alpha=0.5, label='Normal')
        ax.axvline(50, color='orange', linestyle='--', alpha=0.5, label='Moderate')
        ax.axvline(75, color='red', linestyle='--', alpha=0.5, label='Severe/Critical')
        ax.set_title(f'{col} Distribution')
        ax.set_ylabel('Frequency')
        ax.legend()
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=dpi)
    plt.close(fig)


def plot_osi_timeseries(df: pd.DataFrame, osi_col: str,
                        save_path: Path, dpi: int = 150):
    """Full OSI timeseries with severity bands."""
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(df['timestamp'], df[osi_col], linewidth=0.3, color='steelblue', alpha=0.7)
    ax.axhspan(75, 100, alpha=0.08, color='red', label='Critical')
    ax.axhspan(50, 75, alpha=0.08, color='orange', label='Severe')
    ax.axhspan(25, 50, alpha=0.08, color='yellow', label='Moderate')
    ax.set_title(f'{osi_col} Over Time')
    ax.set_ylabel('OSI')
    ax.set_xlabel('Time')
    ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=dpi)
    plt.close(fig)


def plot_daily_osi(agg_df: pd.DataFrame, osi_cols: list,
                   save_path: Path, dpi: int = 150):
    """Daily OSI mean + max with severity bands."""
    mean_cols = [f'{c}_mean' for c in osi_cols if f'{c}_mean' in agg_df.columns]
    if not mean_cols:
        return
    fig, ax = plt.subplots(figsize=(16, 5))
    for col in mean_cols:
        ax.plot(agg_df['timestamp'], agg_df[col],
                label=col.replace('_mean', ''), linewidth=0.8)
    ax.axhspan(75, 100, alpha=0.05, color='red')
    ax.axhspan(50, 75, alpha=0.05, color='orange')
    ax.axhspan(25, 50, alpha=0.05, color='yellow')
    ax.set_title('Daily Mean OSI')
    ax.set_ylabel('OSI')
    ax.set_xlabel('Date')
    ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=dpi)
    plt.close(fig)


def plot_component_contributions(contrib_df: pd.DataFrame,
                                 save_path: Path, dpi: int = 150):
    """Stacked bar of component contributions over time."""
    if contrib_df.empty:
        return
    fig, ax = plt.subplots(figsize=(16, 6))
    cols = [c for c in contrib_df.columns if c != 'timestamp']
    contrib_df.set_index('timestamp')[cols].plot.area(
        ax=ax, linewidth=0, alpha=0.7, colormap='tab10'
    )
    ax.set_title('Component Contributions to OSI')
    ax.set_ylabel('Weighted Contribution')
    ax.set_xlabel('Time')
    ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=dpi)
    plt.close(fig)


def plot_severity_timeline(event_df: pd.DataFrame,
                           save_path: Path, dpi: int = 150):
    """Gantt-like timeline of severity events."""
    if event_df.empty:
        return
    colors = {'Moderate': 'yellow', 'Severe': 'orange', 'Critical': 'red'}
    fig, ax = plt.subplots(figsize=(16, 3))
    for i, row in event_df.iterrows():
        c = colors.get(row['level'], 'gray')
        ax.barh(0, row['duration_seconds'] / 3600,
                left=row['start_time'], height=0.6,
                color=c, alpha=0.7, edgecolor='black', linewidth=0.3)
    ax.set_title('Severity Event Timeline')
    ax.set_xlabel('Time')
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=dpi)
    plt.close(fig)


def plot_hourly_heatmap(hourly_df: pd.DataFrame, osi_col: str,
                        save_path: Path, dpi: int = 150):
    """Hourly OSI heatmap (hour of day x date)."""
    if osi_col not in hourly_df.columns:
        return
    df = hourly_df[['timestamp', osi_col]].copy()
    df['date'] = df['timestamp'].dt.date
    df['hour'] = df['timestamp'].dt.hour
    pivot = df.pivot_table(index='date', columns='hour', values=osi_col, aggfunc='mean')
    fig, ax = plt.subplots(figsize=(14, max(6, len(pivot) * 0.3)))
    sns.heatmap(pivot, ax=ax, cmap='RdYlGn_r', cbar_kws={'label': 'OSI'},
                linewidths=0, xticklabels=True)
    ax.set_title(f'Hourly {osi_col} Heatmap')
    ax.set_ylabel('Date')
    ax.set_xlabel('Hour of Day')
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=dpi)
    plt.close(fig)
