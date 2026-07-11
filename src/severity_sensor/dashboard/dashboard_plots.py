"""
Dashboard Plotting Utilities
=============================
Reusable plotting functions for the Streamlit OSI dashboard.
Uses Plotly for interactive charts and Matplotlib for static figures.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

SEVERITY_COLORS = {'Normal': '#2ecc71', 'Moderate': '#f1c40f',
                   'Severe': '#e67e22', 'Critical': '#e74c3c'}


def plot_osi_gauge(osi_value: float) -> go.Figure:
    """Gauge chart showing current operational severity index."""
    color = '#2ecc71' if osi_value <= 25 else '#f1c40f' if osi_value <= 50 else '#e67e22' if osi_value <= 75 else '#e74c3c'
    fig = go.Figure(go.Indicator(
        mode='gauge+number',
        value=osi_value,
        number={'suffix': '/100', 'font': {'size': 36, 'color': color}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': 'darkgray'},
            'bar': {'color': color, 'thickness': 0.3},
            'bgcolor': 'white',
            'borderwidth': 2,
            'bordercolor': 'gray',
            'steps': [
                {'range': [0, 25], 'color': 'rgba(46, 204, 113, 0.13)'},
                {'range': [25, 50], 'color': 'rgba(241, 196, 15, 0.13)'},
                {'range': [50, 75], 'color': 'rgba(230, 126, 34, 0.13)'},
                {'range': [75, 100], 'color': 'rgba(231, 76, 60, 0.13)'},
            ],
            'threshold': {
                'line': {'color': 'red', 'width': 4},
                'thickness': 0.75,
                'value': osi_value,
            },
        },
        title={'text': 'Current Operational Severity', 'font': {'size': 20}},
    ))
    fig.update_layout(height=280, margin=dict(l=30, r=30, t=60, b=30))
    return fig


def plot_osi_timeseries(df: pd.DataFrame, osi_col: str = 'OSI',
                        rolling_window: int = 3600) -> go.Figure:
    """Interactive operational severity timeseries with rolling average."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df[osi_col],
        mode='lines', name='Operational Severity',
        line=dict(color='steelblue', width=1),
        opacity=0.5,
    ))
    if rolling_window > 0 and len(df) > rolling_window:
        roll = df[osi_col].rolling(rolling_window, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=df['timestamp'], y=roll,
            mode='lines', name=f'{rolling_window//60}min Rolling Avg',
            line=dict(color='darkorange', width=2),
        ))
    fig.add_hrect(y0=75, y1=100, fillcolor='red', opacity=0.05, line_width=0)
    fig.add_hrect(y0=50, y1=75, fillcolor='orange', opacity=0.05, line_width=0)
    fig.add_hrect(y0=25, y1=50, fillcolor='yellow', opacity=0.05, line_width=0)
    fig.update_layout(
        title='Operational Severity Over Time', xaxis_title='Time',
        yaxis_title='Severity Index',
        hovermode='x unified', height=400,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    fig.update_xaxes(rangeslider_visible=True)
    return fig


def plot_severity_pie(sev_dist: dict) -> go.Figure:
    """Pie chart of severity distribution."""
    labels = list(sev_dist.keys())
    values = list(sev_dist.values())
    colors = [SEVERITY_COLORS.get(l, '#999') for l in labels]
    fig = go.Figure(data=[go.Pie(
        labels=labels, values=values, marker=dict(colors=colors),
        textinfo='label+percent', hole=0.4,
    )])
    fig.update_layout(title='Severity Distribution', height=350)
    return fig


def plot_severity_bar(sev_dist: dict) -> go.Figure:
    """Bar chart of severity distribution."""
    labels = list(sev_dist.keys())
    values = list(sev_dist.values())
    colors = [SEVERITY_COLORS.get(l, '#999') for l in labels]
    fig = go.Figure(data=[go.Bar(
        x=labels, y=values, marker_color=colors,
        text=[f'{v:.1f}%' for v in values], textposition='auto',
    )])
    fig.update_layout(
        title='Time Spent in Each Severity Zone',
        xaxis_title='Severity Level', yaxis_title='% of Time',
        height=350, yaxis=dict(range=[0, 100]),
    )
    return fig


def plot_radar_chart(component_means: dict) -> go.Figure:
    """Radar chart of mean risk component scores."""
    categories = list(component_means.keys())
    values = list(component_means.values())
    fig = go.Figure(data=go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill='toself', name='Mean Score',
        line=dict(color='steelblue', width=2),
        marker=dict(color='steelblue', size=6),
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100]),
        ),
        title='Risk Component Profile', height=400,
    )
    return fig


def plot_component_contributions(component_means: dict, weights: dict) -> go.Figure:
    """Bar chart showing weighted component contributions to OSI."""
    contribs = {}
    for col, mean_val in component_means.items():
        w = weights.get(col, 0)
        contribs[col] = w * mean_val
    contribs = dict(sorted(contribs.items(), key=lambda x: -x[1]))
    colors = px.colors.qualitative.Set2[:len(contribs)]
    fig = go.Figure(data=[go.Bar(
        x=list(contribs.keys()), y=list(contribs.values()),
        marker_color=colors,
        text=[f'{v:.2f}' for v in contribs.values()],
        textposition='auto',
    )])
    fig.update_layout(
        title='Weighted Risk Contributions',
        xaxis_title='Risk Component', yaxis_title='Weighted Contribution',
        height=400, xaxis_tickangle=-30,
    )
    return fig


def plot_stacked_area(df: pd.DataFrame, score_cols: list) -> go.Figure:
    """Stacked area chart of component scores over time."""
    fig = go.Figure()
    colors = px.colors.qualitative.Set2
    for i, col in enumerate(score_cols):
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df['timestamp'], y=df[col],
                mode='lines', name=col.replace('_Risk_Score', ''),
                line=dict(width=0.5),
                stackgroup='one',
                fillcolor=colors[i % len(colors)],
            ))
    fig.update_layout(
        title='Risk Components Over Time (Stacked)',
        xaxis_title='Time', yaxis_title='Score',
        hovermode='x unified', height=400,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    return fig


def plot_daily_trends(daily_df: pd.DataFrame) -> go.Figure:
    """Daily mean + max severity trend."""
    fig = go.Figure()
    if 'OSI_mean' in daily_df.columns:
        fig.add_trace(go.Scatter(
            x=daily_df['timestamp'], y=daily_df['OSI_mean'],
            mode='lines+markers', name='Daily Mean',
            line=dict(color='steelblue', width=2),
            marker=dict(size=6),
        ))
    if 'OSI_max' in daily_df.columns:
        fig.add_trace(go.Scatter(
            x=daily_df['timestamp'], y=daily_df['OSI_max'],
            mode='lines+markers', name='Daily Max',
            line=dict(color='coral', width=2, dash='dot'),
            marker=dict(size=6),
        ))
    fig.add_hrect(y0=75, y1=100, fillcolor='red', opacity=0.05, line_width=0)
    fig.add_hrect(y0=50, y1=75, fillcolor='orange', opacity=0.05, line_width=0)
    fig.add_hrect(y0=25, y1=50, fillcolor='yellow', opacity=0.05, line_width=0)
    fig.update_layout(
        title='Daily Severity Trends', xaxis_title='Date', yaxis_title='Severity Index',
        hovermode='x unified', height=400,
    )
    return fig


def plot_weekly_trends(weekly_df: pd.DataFrame) -> go.Figure:
    """Weekly severity trend."""
    fig = go.Figure()
    if 'OSI_mean' in weekly_df.columns:
        fig.add_trace(go.Scatter(
            x=weekly_df['timestamp'], y=weekly_df['OSI_mean'],
            mode='lines+markers', name='Weekly Mean',
            line=dict(color='steelblue', width=2),
            marker=dict(size=8),
        ))
    fig.add_hrect(y0=75, y1=100, fillcolor='red', opacity=0.05, line_width=0)
    fig.add_hrect(y0=50, y1=75, fillcolor='orange', opacity=0.05, line_width=0)
    fig.add_hrect(y0=25, y1=50, fillcolor='yellow', opacity=0.05, line_width=0)
    fig.update_layout(
        title='Weekly Severity Trend', xaxis_title='Week', yaxis_title='Severity Index',
        hovermode='x unified', height=350,
    )
    return fig


def plot_anomaly_distribution(anomaly_scores: pd.Series) -> go.Figure:
    """Distribution of anomaly scores."""
    fig = go.Figure(data=[go.Histogram(
        x=anomaly_scores, nbinsx=60,
        marker_color='crimson', opacity=0.7,
    )])
    fig.add_vline(x=80, line_dash='dash', line_color='red',
                  annotation_text='High', annotation_position='top right')
    fig.add_vline(x=70, line_dash='dash', line_color='orange',
                  annotation_text='Moderate', annotation_position='top right')
    fig.update_layout(
        title='Abnormal Behaviour Score Distribution',
        xaxis_title='Abnormality Score', yaxis_title='Frequency',
        height=350,
    )
    return fig


def plot_anomaly_timeline(df: pd.DataFrame) -> go.Figure:
    """Timeline of abnormal behaviour scores over time."""
    if 'Anomaly_Score' not in df.columns or 'timestamp' not in df.columns:
        return go.Figure()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['Anomaly_Score'],
        mode='markers', name='Abnormality Score',
        marker=dict(color='crimson', size=3, opacity=0.5),
    ))
    fig.add_hline(y=80, line_dash='dash', line_color='red',
                  annotation_text='High Threshold')
    fig.update_layout(
        title='Abnormal Behaviour Timeline', xaxis_title='Time',
        yaxis_title='Abnormality Score', height=300,
        hovermode='x unified',
    )
    return fig
