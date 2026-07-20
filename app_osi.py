"""
Mining Truck Tyre Operational Severity Dashboard
=================================================
Standalone Streamlit dashboard consuming pre-computed OSI data.
Run with: streamlit run app_osi.py
"""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title='Mining Truck Tyre Operational Severity Dashboard',
    layout='wide',
    initial_sidebar_state='expanded',
)

from src.severity_sensor.dashboard.dashboard_loader import (
    load_osi_sample, load_hourly, load_daily,
    load_weekly, load_component_sample,
    load_events, load_daily_summary, get_dataset_info,
    load_osi_stats, load_severity_distribution_from_sample,
    load_component_means_from_sample,
    SCORE_COLS,
)
from src.severity_sensor.dashboard.dashboard_plots import (
    plot_osi_gauge, plot_osi_timeseries, plot_severity_pie,
    plot_severity_bar, plot_radar_chart, plot_component_contributions,
    plot_stacked_area, plot_daily_trends, plot_weekly_trends,
    plot_anomaly_distribution, plot_anomaly_timeline, SEVERITY_COLORS,
)
from src.severity_sensor.dashboard.recommendations import (
    get_health_status, get_component_recommendations,
    get_general_recommendations, SEVERITY_LEVELS,
)

WEIGHTS = {
    'Pressure_Risk_Score':   0.15,
    'Thermal_Risk_Score':    0.15,
    'Load_Risk_Score':       0.20,
    'Vibration_Risk_Score':  0.15,
    'Braking_Risk_Score':    0.05,
    'Terrain_Risk_Score':    0.05,
    'Usage_Risk_Score':      0.15,
    'Anomaly_Score':         0.10,
}

# ======================================================================
# Cache helpers
# ======================================================================

@st.cache_resource
def _load_data():
    """Load lightweight dashboard data -- no parquet files are read.

    Returns a dict with all dashboard data, or a dict of empty
    defaults if any loading step fails.
    """
    empty = {
        'info': {'start': pd.Timestamp('2000-01-01'), 'end': pd.Timestamp('2000-01-02'),
                 'duration': pd.Timedelta(days=1), 'records': 0, 'variant': 'N/A', 'truck_id': 'N/A'},
        'osi_sample': pd.DataFrame(columns=['timestamp', 'OSI']),
        'hourly': pd.DataFrame(), 'daily': pd.DataFrame(), 'weekly': pd.DataFrame(),
        'comp_sample': pd.DataFrame(),
        'events': pd.DataFrame(columns=['level', 'start_time', 'end_time',
                                        'duration_seconds', 'peak_osi', 'mean_osi']),
        'daily_sum': pd.DataFrame(),
        'stats': {'mean_osi': 0.0, 'max_osi': 0.0, 'p95_osi': 0.0},
        'sev_dist': {'Normal': 0.0, 'Moderate': 0.0, 'Severe': 0.0, 'Critical': 0.0},
        'comp_means': {},
    }
    try:
        info = get_dataset_info()
        osi_sample = load_osi_sample(0.02)
        hourly = load_hourly()
        daily = load_daily()
        weekly = load_weekly()
        comp_sample = load_component_sample(0.02)
        events = load_events()
        daily_sum = load_daily_summary()
        osi_stats = load_osi_stats()
        sev_dist = load_severity_distribution_from_sample()
        comp_means = load_component_means_from_sample()
        return {
            'info': info,
            'osi_sample': osi_sample,
            'hourly': hourly, 'daily': daily, 'weekly': weekly,
            'comp_sample': comp_sample,
            'events': events, 'daily_sum': daily_sum,
            'stats': osi_stats,
            'sev_dist': sev_dist,
            'comp_means': comp_means,
        }
    except MemoryError:
        st.error('Not enough memory to load telemetry data.')
        return empty
    except Exception as e:
        st.error(f'Telemetry data could not be loaded: {e}')
        return empty


@st.cache_data
def _compute_stats(data):
    stats = data['stats'].copy()
    events = data['events']
    stats['severe_count'] = int((events['level'] == 'Severe').sum()) if len(events) > 0 else 0
    stats['critical_count'] = int((events['level'] == 'Critical').sum()) if len(events) > 0 else 0
    return stats


@st.cache_data
def _compute_severity_dist(data):
    return data['sev_dist']


@st.cache_data
def _compute_component_means(data):
    return data['comp_means']


# ======================================================================
# Sidebar
# ======================================================================

def render_sidebar(data):
    info = data['info']
    st.sidebar.image(
        'https://cdn-icons-png.flaticon.com/512/1995/1995515.png',
        width=60,
    )
    st.sidebar.title('OSI Dashboard')
    st.sidebar.markdown('---')

    st.sidebar.subheader('Dataset Info')
    st.sidebar.write(f'**Duration:** {info["duration"]}')
    st.sidebar.write(f'**Start:** {info["start"].date()}')
    st.sidebar.write(f'**End:** {info["end"].date()}')
    st.sidebar.write(f'**Records:** {info["records"]:,}')
    st.sidebar.write(f'**OSI Variant:** {info["variant"]}')
    st.sidebar.write(f'**Truck ID:** {info["truck_id"]}')

    st.sidebar.markdown('---')
    st.sidebar.subheader('Filters')

    date_range = st.sidebar.date_input(
        'Date Range',
        value=(info['start'].date(), info['end'].date()),
        min_value=info['start'].date(),
        max_value=info['end'].date(),
    )
    severity_filter = st.sidebar.multiselect(
        'Severity Level',
        ['Normal', 'Moderate', 'Severe', 'Critical'],
        default=['Moderate', 'Severe', 'Critical'],
    )
    component_filter = st.sidebar.multiselect(
        'Risk Component',
        [c.replace('_Risk_Score', '') for c in SCORE_COLS],
        default=[c.replace('_Risk_Score', '') for c in SCORE_COLS],
    )

    st.sidebar.markdown('---')
    st.sidebar.caption('Phase 5A — OSI Dashboard')
    st.sidebar.caption('Data: Pre-computed OSI pipeline')

    return date_range, severity_filter, component_filter


# ======================================================================
# SECTION 1 — Top Metrics
# ======================================================================

def render_metrics(stats):
    cols = st.columns(6)
    with cols[0]:
        m = stats['mean_osi']
        color = '#2ecc71' if m <= 25 else '#f1c40f' if m <= 50 else '#e67e22' if m <= 75 else '#e74c3c'
        st.markdown(
            f'<div style="background:{color}20;padding:10px;border-radius:8px;'
            f'border-left:4px solid {color}">'
            f'<h6 style="margin:0">Average OSI</h6>'
            f'<h3 style="margin:0;color:{color}">{m:.1f}</h3></div>',
            unsafe_allow_html=True,
        )
    with cols[1]:
        mx = stats['max_osi']
        color = '#2ecc71' if mx <= 25 else '#f1c40f' if mx <= 50 else '#e67e22' if mx <= 75 else '#e74c3c'
        st.markdown(
            f'<div style="background:{color}20;padding:10px;border-radius:8px;'
            f'border-left:4px solid {color}">'
            f'<h6 style="margin:0">Max OSI</h6>'
            f'<h3 style="margin:0;color:{color}">{mx:.1f}</h3></div>',
            unsafe_allow_html=True,
        )
    with cols[2]:
        p95 = stats['p95_osi']
        color = '#2ecc71' if p95 <= 25 else '#f1c40f' if p95 <= 50 else '#e67e22' if p95 <= 75 else '#e74c3c'
        st.markdown(
            f'<div style="background:{color}20;padding:10px;border-radius:8px;'
            f'border-left:4px solid {color}">'
            f'<h6 style="margin:0">95th Percentile</h6>'
            f'<h3 style="margin:0;color:{color}">{p95:.1f}</h3></div>',
            unsafe_allow_html=True,
        )
    with cols[3]:
        health, _ = get_health_status(stats['mean_osi'], stats['max_osi'])
        hc = {'HEALTHY': '#2ecc71', 'MODERATE': '#f1c40f',
              'ELEVATED': '#e67e22', 'CRITICAL': '#e74c3c'}
        st.markdown(
            f'<div style="background:{hc.get(health, "#999")}20;padding:10px;'
            f'border-radius:8px;border-left:4px solid {hc.get(health, "#999")}">'
            f'<h6 style="margin:0">Vehicle Health</h6>'
            f'<h3 style="margin:0;color:{hc.get(health, "#999")}">{health}</h3></div>',
            unsafe_allow_html=True,
        )
    with cols[4]:
        sc = stats['severe_count']
        st.markdown(
            f'<div style="background:#e67e2220;padding:10px;border-radius:8px;'
            f'border-left:4px solid #e67e22">'
            f'<h6 style="margin:0">Severe Events</h6>'
            f'<h3 style="margin:0;color:#e67e22">{sc}</h3></div>',
            unsafe_allow_html=True,
        )
    with cols[5]:
        cc = stats['critical_count']
        st.markdown(
            f'<div style="background:#e74c3c20;padding:10px;border-radius:8px;'
            f'border-left:4px solid #e74c3c">'
            f'<h6 style="margin:0">Critical Events</h6>'
            f'<h3 style="margin:0;color:#e74c3c">{cc}</h3></div>',
            unsafe_allow_html=True,
        )


# ======================================================================
# SECTION 2 — Current Health Status
# ======================================================================

def render_health(data):
    osi_sample = data['osi_sample']
    stats = data['stats']
    current_osi = float(osi_sample['OSI'].iloc[-1]) if len(osi_sample) > 0 else stats.get('max_osi', 0.0)
    col1, col2 = st.columns([1, 2])
    with col1:
        st.plotly_chart(plot_osi_gauge(current_osi), use_container_width=True)
    with col2:
        st.subheader('Severity Level Definitions')
        levels_data = {
            'Level': ['Normal', 'Moderate', 'Severe', 'Critical'],
            'Range': ['0 — 25', '25 — 50', '50 — 75', '75 — 100'],
            'Color': ['🟢 Green', '🟡 Yellow', '🟠 Orange', '🔴 Red'],
        }
        st.table(pd.DataFrame(levels_data))


# ======================================================================
# SECTION 3 — OSI Timeseries
# ======================================================================

def render_timeseries(data):
    st.subheader('OSI Over Time')
    df = data['osi_sample']
    fig = plot_osi_timeseries(df)
    st.plotly_chart(fig, use_container_width=True)


# ======================================================================
# SECTION 4 — Severity Distribution
# ======================================================================

def render_severity_dist(sev_dist):
    st.subheader('Severity Distribution')
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_severity_pie(sev_dist), use_container_width=True)
    with col2:
        st.plotly_chart(plot_severity_bar(sev_dist), use_container_width=True)
    dist_df = pd.DataFrame([
        {'Level': k, '% Time': f'{v:.2f}%'} for k, v in sev_dist.items()
    ])
    st.dataframe(dist_df, use_container_width=True, hide_index=True)


# ======================================================================
# SECTION 5 — Risk Component Breakdown
# ======================================================================

def render_components(data, component_filter):
    comp_means = _compute_component_means(data)
    
    # Map full names to short names
    full_to_short = {c: c.replace('_Risk_Score', '') for c in SCORE_COLS}
    short_to_full = {v: k for k, v in full_to_short.items()}
    
    # Apply filter
    filtered_means = {}
    for short in component_filter:
        full = short_to_full.get(short)
        if full and full in comp_means:
            filtered_means[full] = comp_means[full]
    
    if not filtered_means:
        st.info('Select at least one risk component.')
        return
    
    dominant = max(filtered_means, key=filtered_means.get)
    
    st.subheader(f'Risk Component Breakdown')
    st.markdown(f'**Dominant risk:** {dominant.replace("_Risk_Score", "")} '
                f'(mean = {filtered_means[dominant]:.2f})')
    
    tab1, tab2, tab3 = st.tabs(['Radar Chart', 'Contributions', 'Stacked Area'])
    
    with tab1:
        st.plotly_chart(plot_radar_chart(filtered_means), use_container_width=True)
    
    with tab2:
        w = {k: WEIGHTS.get(k, 0) for k in filtered_means}
        st.plotly_chart(plot_component_contributions(filtered_means, w),
                       use_container_width=True)
    
    with tab3:
        comp_sample = data['comp_sample']
        filtered_full = [c for c in SCORE_COLS if c in filtered_means]
        if 'timestamp' in comp_sample.columns:
            st.plotly_chart(
                plot_stacked_area(comp_sample, filtered_full),
                use_container_width=True,
            )


# ======================================================================
# SECTION 6 — Event Analysis
# ======================================================================

def render_events(data):
    st.subheader('Severity Events')
    events = data['events']
    if len(events) == 0:
        st.info('No severity events detected.')
        return
    
    display = events.copy()
    if 'duration_seconds' in display.columns:
        display['duration_min'] = display['duration_seconds'] / 60
    display['peak_osi'] = display['peak_osi'].round(1)
    display['mean_osi'] = display['mean_osi'].round(1)
    
    # Color rows by severity
    def color_level(val):
        colors = {'Critical': '#e74c3c', 'Severe': '#e67e22', 'Moderate': '#f1c40f'}
        return f'background-color: {colors.get(val, "white")}30'
    
    styled = display.style.applymap(
        color_level, subset=['level']
    )
    
    st.dataframe(styled, use_container_width=True, hide_index=True)
    
    col1, col2 = st.columns(2)
    with col1:
        csv = events.to_csv(index=False).encode('utf-8')
        st.download_button(
            '📥 Download Events CSV',
            data=csv,
            file_name='severity_events.csv',
            mime='text/csv',
        )
    with col2:
        st.write(f'**Total events:** {len(events)}')
        for level in ['Critical', 'Severe', 'Moderate']:
            cnt = len(events[events['level'] == level])
            st.write(f'**{level}:** {cnt}')


# ======================================================================
# SECTION 7 — Daily & Weekly Trends
# ======================================================================

def render_trends(data):
    st.subheader('Daily & Weekly Trends')
    tab1, tab2 = st.tabs(['Daily', 'Weekly'])
    with tab1:
        daily = data['daily']
        if len(daily) > 0:
            st.plotly_chart(plot_daily_trends(daily), use_container_width=True)
            with st.expander('View Daily Data'):
                st.dataframe(daily, use_container_width=True, hide_index=True)
    with tab2:
        weekly = data['weekly']
        if len(weekly) > 0:
            st.plotly_chart(plot_weekly_trends(weekly), use_container_width=True)
            with st.expander('View Weekly Data'):
                st.dataframe(weekly, use_container_width=True, hide_index=True)


# ======================================================================
# SECTION 8 — Anomaly Analysis
# ======================================================================

def render_anomaly(data):
    st.subheader('Anomaly Analysis')
    from src.severity_sensor.dashboard.dashboard_loader import load_anomaly_sample, load_top_anomalies
    anomaly_data = load_anomaly_sample()
    if len(anomaly_data) == 0 or 'Anomaly_Score' not in anomaly_data.columns:
        st.info('Anomaly scores not available.')
        return

    comp_sample = data['comp_sample']

    tab1, tab2, tab3 = st.tabs(['Distribution', 'Top Anomalies', 'Timeline'])

    with tab1:
        st.plotly_chart(
            plot_anomaly_distribution(anomaly_data['Anomaly_Score']),
            use_container_width=True,
        )

    with tab2:
        top = load_top_anomalies(20)
        st.dataframe(top, use_container_width=True, hide_index=True)

    with tab3:
        if 'timestamp' in comp_sample.columns:
            st.plotly_chart(
                plot_anomaly_timeline(comp_sample),
                use_container_width=True,
            )


# ======================================================================
# SECTION 9 — Health Recommendations
# ======================================================================

def render_recommendations(data, stats, sev_dist):
    st.subheader('Health Recommendations')
    comp_means = _compute_component_means(data)
    ordered = sorted(comp_means.items(), key=lambda x: -x[1])
    dominant = ordered[0][0] if ordered else 'N/A'
    
    status, desc = get_health_status(stats['mean_osi'], stats['max_osi'])
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown(f'**Current Status:** `{status}`')
        st.markdown(f'*{desc}*')
        st.markdown('')
        
        recs = get_component_recommendations(dominant, ordered)
        for r in recs:
            st.markdown(r)
    
    with col2:
        general = get_general_recommendations(
            stats['mean_osi'], stats['max_osi'],
            stats['severe_count'], stats['critical_count'],
        )
        for r in general:
            st.markdown(r)
        
        if not general and not any('**' in r for r in recs):
            st.success('No urgent recommendations. Continue regular monitoring.')


# ======================================================================
# SECTION 10 — Exports
# ======================================================================

def render_exports():
    st.subheader('Data Exports')
    st.markdown('Download summary data for offline analysis. Large raw datasets are not included.')

    export_files = {
        'Daily Summary': 'outputs/osi_phase4/dashboard_data/daily_summary.csv',
        'Severity Events': 'outputs/osi_phase4/dashboard_data/severity_events.csv',
    }

    cols = st.columns(2)
    for i, (label, rel_path) in enumerate(export_files.items()):
        with cols[i]:
            path = ROOT / rel_path
            if path.exists():
                df_export = pd.read_csv(path)
                if label == 'Daily Summary' and len(df_export) > 5000:
                    df_export = df_export.head(5000)
                elif label == 'Severity Events' and len(df_export) > 2000:
                    df_export = df_export.head(2000)
                csv_data = df_export.to_csv(index=False).encode('utf-8')
                st.download_button(
                    f'{label}',
                    data=csv_data,
                    file_name=rel_path.split('/')[-1],
                    mime='text/csv',
                )
            else:
                st.warning(f'{label} not found')


# ======================================================================
# SECTION 11 — Dashboard Reporting (auto-save)
# ======================================================================

def generate_dashboard_report(data, stats, sev_dist):
    """Generate dashboard_summary.md."""
    report_dir = ROOT / 'outputs/osi_dashboard'
    report_dir.mkdir(parents=True, exist_ok=True)
    
    comp_means = _compute_component_means(data)
    ordered = sorted(comp_means.items(), key=lambda x: -x[1])
    dominant = ordered[0][0].replace('_Risk_Score', '') if ordered else 'N/A'
    status, _ = get_health_status(stats['mean_osi'], stats['max_osi'])
    
    recs = get_component_recommendations(ordered[0][0] if ordered else '', ordered)
    rec_text = '\n'.join(recs)
    
    lines = [
        '# OSI Dashboard Summary',
        '',
        f'*Generated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}*',
        '',
        '## Current Health',
        f'- Status: **{status}**',
        f'- Mean OSI: {stats["mean_osi"]:.2f}',
        f'- Max OSI: {stats["max_osi"]:.2f}',
        '',
        '## Severity Distribution',
    ]
    for level, pct in sev_dist.items():
        lines.append(f'- {level}: {pct:.2f}%')
    lines += [
        '',
        '## Dominant Risk',
        f'- {dominant}',
        '',
        '## Events',
        f'- Severe events: {stats["severe_count"]}',
        f'- Critical events: {stats["critical_count"]}',
        '',
        '## Recommendations',
        rec_text,
        '',
        '---',
        '*Generated by OSI Phase 5A Dashboard*',
    ]
    
    path = report_dir / 'dashboard_summary.md'
    path.write_text('\n'.join(lines))
    return path


# ======================================================================
# Main App
# ======================================================================

def main():
    st.title('🚛 Mining Truck Tyre Operational Severity Dashboard')
    st.markdown('---')
    
    # Load all data
    with st.spinner('Loading OSI data...'):
        data = _load_data()
    
    # Sidebar
    date_range, severity_filter, component_filter = render_sidebar(data)
    
    # Compute stats
    stats = _compute_stats(data)
    sev_dist = _compute_severity_dist(data)
    
    # ---- SECTIONS ----
    render_metrics(stats)
    st.markdown('---')
    
    render_health(data)
    st.markdown('---')
    
    render_timeseries(data)
    st.markdown('---')
    
    render_severity_dist(sev_dist)
    st.markdown('---')
    
    render_components(data, component_filter)
    st.markdown('---')
    
    render_events(data)
    st.markdown('---')
    
    render_trends(data)
    st.markdown('---')
    
    render_anomaly(data)
    st.markdown('---')
    
    render_recommendations(data, stats, sev_dist)
    st.markdown('---')
    
    render_exports()
    st.markdown('---')
    
    # Generate report
    report_path = generate_dashboard_report(data, stats, sev_dist)
    st.caption(f'📄 Dashboard report saved: `{report_path}`')

    # clear cache button
    st.sidebar.markdown('---')
    if st.sidebar.button('🔄 Clear Cache & Reload'):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()


if __name__ == '__main__':
    main()
