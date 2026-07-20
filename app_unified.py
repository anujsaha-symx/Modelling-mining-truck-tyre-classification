"""
Mining Truck Tyre Health Monitoring System
==========================================
Combines computer-vision-based tyre inspection with telemetry-driven
operational severity monitoring into a single Streamlit application.

Tabs:
  1. Computer Vision Inspection
  2. Telemetry Analytics
  3. Vehicle Health Report

Run with: streamlit run app_unified.py
"""

import json
import sys
import tempfile
import shutil
from io import BytesIO
from pathlib import Path

import streamlit as st
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ---- Computer vision imports ----
from src.demo import config
from src.demo.gatekeeper_service import GatekeeperService
from src.demo.wear_service import WearService
from src.demo.visualization import draw_boxes
from src.demo.utils import (
    GATEKEEPER_METRICS,
    WEAR_METRICS_V2,
    GATEKEEPER_V3_CKPT,
    WEAR_V2_CKPT,
    load_image,
    check_checkpoint,
)

# ---- Telemetry dashboard imports ----
from src.severity_sensor.dashboard.dashboard_loader import (
    load_osi_sample,
    load_hourly,
    load_daily,
    load_weekly,
    load_component_sample,
    load_events,
    load_daily_summary,
    get_dataset_info,
    load_osi_stats,
    load_severity_distribution_from_sample,
    load_component_means_from_sample,
    set_data_root,
    reset_data_root,
    SCORE_COLS,
)
from src.severity_sensor.dashboard.dashboard_plots import (
    plot_osi_gauge,
    plot_osi_timeseries,
    plot_severity_pie,
    plot_severity_bar,
    plot_radar_chart,
    plot_component_contributions,
    plot_stacked_area,
    plot_daily_trends,
    plot_weekly_trends,
    plot_anomaly_distribution,
    plot_anomaly_timeline,
    SEVERITY_COLORS,
)
from src.severity_sensor.dashboard.recommendations import (
    get_health_status,
    get_component_recommendations,
    get_general_recommendations,
    SEVERITY_LEVELS,
)

# ---- Unified modules ----
from src.dashboard_unified.health_logic import (
    classify_image_health,
    classify_sensor_health,
    classify_overall_health,
    compute_health_score,
    generate_combined_recommendations,
    HEALTH_COLORS,
)
from src.dashboard_unified.report_generator import (
    generate_combined_report,
)
from src.dashboard_unified.demo_datasets import (
    get_available_datasets,
    get_dataset_descriptions,
    build_demo_datasets,
    get_demo_dir_for_scenario,
)
from src.dashboard_unified.pipeline_runner import run_full_pipeline

st.set_page_config(
    page_title='Tyre Health Monitoring System',
    layout='wide',
    initial_sidebar_state='expanded',
)

OSI_WEIGHTS = {
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
# Reusable helpers
# ======================================================================

@st.cache_resource
def _init_gatekeeper():
    return GatekeeperService()


@st.cache_resource
def _init_wear():
    return WearService()


EXAMPLE_DIR = ROOT / 'datasets' / 'annotated'
EXAMPLE_GOOD = sorted((EXAMPLE_DIR / 'good').glob('*.jpg'))[:3]
EXAMPLE_BAD = sorted((EXAMPLE_DIR / 'bad').glob('*.jpg'))[:3]
EXAMPLE_NEG = sorted((EXAMPLE_DIR / 'negative').glob('*.jpg'))[:3]


def _load_example_image(category):
    files = {'Good Tyre': EXAMPLE_GOOD, 'Bad Tyre': EXAMPLE_BAD, 'Non Tyre': EXAMPLE_NEG}.get(category)
    if files:
        return Image.open(files[0]).convert('RGB')
    return None


def _resize_display_image(image, max_width=800, max_height=450):
    w, h = image.size
    ratio = min(max_width / w, max_height / h, 1.0)
    if ratio < 1.0:
        return image.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    return image


def _process_image(image):
    gatekeeper = _init_gatekeeper()
    wear = _init_wear()
    gate_dets = gatekeeper.predict(image)
    gate_decision = gatekeeper.decide(gate_dets)
    gate_vis = image.copy()
    if gate_dets:
        gate_vis = draw_boxes(gate_vis, gate_dets)
    wear_result = None
    wear_vis = None
    final_class = 'Non-Tire'
    confidence = 0.0
    if gate_decision['is_tire']:
        boxes, scores, labels = wear.predict(image)
        wear_result = wear.classify_output(boxes, scores, labels)
        final_class = wear_result['final_class']
        confidence = max(d['confidence'] for d in wear_result['detections']) if wear_result['detections'] else 0.0
        wear_vis = image.copy()
        wear_vis = draw_boxes(wear_vis, wear_result['detections'])
    return {
        'gatekeeper_result': 'Tyre' if gate_decision['is_tire'] else 'Non-Tyre',
        'gatekeeper_confidence': gate_decision.get('confidence', 0.0),
        'gatekeeper_reason': gate_decision.get('reason', ''),
        'gatekeeper_detections': gate_dets,
        'gatekeeper_vis': gate_vis,
        'wear_result': wear_result,
        'wear_vis': wear_vis,
        'final_class': final_class,
        'confidence': confidence,
    }


def _image_to_bytes(img, fmt='PNG'):
    buf = BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def _build_metric_card(label, value, color):
    return (
        '<div style="background:' + color + '20;padding:10px;border-radius:8px;'
        'border-left:4px solid ' + color + '">'
        '<h6 style="margin:0">' + label + '</h6>'
        '<h3 style="margin:0;color:' + color + '">' + str(value) + '</h3></div>'
    )


# ======================================================================
# Telemetry cached helpers
# ======================================================================

@st.cache_resource
def _load_osi_data():
    """Load lightweight dashboard data -- no parquet files are read.

    Returns a dict with all dashboard data, or a dict of empty
    defaults if any loading step fails (avoids crashing the UI
    with tracebacks or ArrowMemoryError).
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
        st.error('Not enough memory to load telemetry data. Try selecting a smaller dataset.')
        return empty
    except Exception as e:
        st.error(f'Telemetry data could not be loaded: {e}')
        return empty


def _check_checkpoints():
    ckpt_gk = config.GATEKEEPER_CHECKPOINTS[config.GATEKEEPER_MODEL]
    ckpt_wear = str(WEAR_V2_CKPT)
    missing = []
    if not Path(ckpt_gk).is_file():
        missing.append('Gatekeeper model: ' + ckpt_gk)
    if not Path(ckpt_wear).is_file():
        missing.append('Wear model: ' + ckpt_wear)
    return missing


# ======================================================================
# TAB 1 - Computer Vision Inspection
# ======================================================================

def render_image_tab():
    st.markdown("""
    <style>
        .img-metric { background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 10px; padding: 0.75rem 1rem; text-align: center; }
        .img-metric .label { font-size: 0.75rem; color: #6c757d; text-transform: uppercase; letter-spacing: 0.5px; }
        .img-metric .value { font-size: 1.1rem; font-weight: 700; color: #212529; margin-top: 2px; }
        .img-metric .value.good { color: #28a745; }
        .img-metric .value.bad { color: #dc3545; }
        .img-metric .value.info { color: #17a2b8; }
        .img-metric .value.pass { color: #28a745; }
        .img-metric .value.fail { color: #dc3545; }
    </style>
    """, unsafe_allow_html=True)

    missing = _check_checkpoints()
    if missing:
        st.error('Missing model file(s):\n' + '\n'.join(missing))
        return

    st.subheader('Computer Vision Tyre Inspection')
    st.markdown('Upload a tyre image or select an example below.')

    eg_cols = st.columns(4)
    example_labels = [('Good Tyre', 'Good Tyre'), ('Damaged Tyre', 'Bad Tyre'), ('Non-Tyre Object', 'Non Tyre')]
    with eg_cols[0]:
        st.caption('Examples:')
    for i, (btn_label, cat) in enumerate(example_labels):
        with eg_cols[i + 1]:
            if st.button(btn_label, width='stretch', key='img_example_' + cat):
                img = _load_example_image(cat)
                if img:
                    st.session_state['unified_image'] = img
                    st.session_state['unified_image_source'] = cat
                    st.session_state['unified_scan_ready'] = False
                    st.rerun()

    uploaded_file = st.file_uploader(
        'Upload a tyre image', type=['jpg', 'jpeg', 'png'],
        key='unified_uploader',
        help='Drag & drop or click to upload a mining truck tyre image',
    )
    if uploaded_file is not None:
        try:
            image = load_image(uploaded_file)
            st.session_state['unified_image'] = image
            st.session_state['unified_image_source'] = 'upload'
            st.session_state['unified_scan_ready'] = False
        except Exception as e:
            st.error('Invalid image file: ' + str(e))
            return

    if 'unified_image' not in st.session_state:
        st.info('Upload an image or select an example above to begin.')
        with st.expander('How It Works', expanded=True):
            st.markdown("""
**1. Tyre Verification** -- A computer vision model checks whether the image contains a mining truck tyre.

**2. Wear Detection** -- A second model inspects the tyre for cuts, surface damage, and wear indicators.

**3. Classification** -- The system classifies the tyre as Good, Damaged, or Non-Tyre based on both inspections.
            """)
        return

    image = st.session_state['unified_image']
    scan_cols = st.columns([1, 3])
    with scan_cols[0]:
        scan_clicked = st.button('Start Inspection', type='primary', width='stretch')
    with scan_cols[1]:
        st.caption('Run tyre verification and damage detection')

    if scan_clicked:
        st.session_state['unified_scan_ready'] = True

    if not st.session_state.get('unified_scan_ready'):
        st.image(_resize_display_image(image), caption='Input Image', width='stretch')
        st.info('Press **Start Inspection** to run the analysis.')
        return

    with st.spinner('Running inspection...'):
        try:
            result = _process_image(image)
        except FileNotFoundError as e:
            st.error('Missing model file: ' + str(e))
            st.warning('Please ensure all model files are available.')
            return
        except Exception as e:
            st.error('Inspection failed: ' + str(e))
            return

    st.session_state['unified_image_result'] = result

    left, right = st.columns(2)
    with left:
        st.image(_resize_display_image(image), caption='Uploaded Image', width='stretch')
    with right:
        det_img = result.get('wear_vis') or result.get('gatekeeper_vis')
        if det_img is not None:
            st.image(_resize_display_image(det_img), caption='Detection Result', width='stretch')

    final_class = result['final_class']
    confidence = result['confidence']
    gatekeeper_passed = result['gatekeeper_result'] == 'Tyre'

    if final_class == 'Good-Tire':
        status_text, status_css = 'GOOD CONDITION', 'good'
    elif final_class == 'Bad-Tire':
        status_text, status_css = 'DAMAGED', 'bad'
    else:
        status_text, status_css = 'NON-TYRE', 'info'

    gatekeeper_text = 'PASS' if gatekeeper_passed else 'FAIL'
    gatekeeper_css = 'pass' if gatekeeper_passed else 'fail'
    wear_detail = result['wear_result']['reason'] if result['wear_result'] else 'Not inspected (non-tyre)'

    cols = st.columns(4)
    with cols[0]:
        st.markdown('<div class="img-metric"><div class="label">Tyre Status</div><div class="value ' + status_css + '">' + status_text + '</div></div>', unsafe_allow_html=True)
    with cols[1]:
        st.markdown('<div class="img-metric"><div class="label">Confidence</div><div class="value">' + f'{confidence:.1%}' + '</div></div>', unsafe_allow_html=True)
    with cols[2]:
        st.markdown('<div class="img-metric"><div class="label">Tyre Verification</div><div class="value ' + gatekeeper_css + '">' + gatekeeper_text + '</div></div>', unsafe_allow_html=True)
    with cols[3]:
        st.markdown('<div class="img-metric"><div class="label">Damage Detail</div><div class="value">' + wear_detail + '</div></div>', unsafe_allow_html=True)

    with st.expander('Detailed Results', expanded=False):
        det_tabs = st.tabs(['Verification Details', 'Wear Analysis', 'Summary'])
        with det_tabs[0]:
            st.json({
                'detections': [
                    {'class': d['class'], 'confidence': round(d['confidence'], 4), 'bbox': [round(b, 2) for b in d['bbox']]}
                    for d in result['gatekeeper_detections']
                ],
                'decision': {
                    'is_tyre': result['gatekeeper_result'] == 'Tyre',
                    'confidence': round(result['gatekeeper_confidence'], 4),
                    'reason': result['gatekeeper_reason'],
                },
            })
        with det_tabs[1]:
            if result['wear_result']:
                st.json({
                    'classification': result['wear_result']['final_class'],
                    'reason': result['wear_result']['reason'],
                    'detections': [
                        {'class': d['class'], 'confidence': round(d['confidence'], 4), 'bbox': [round(b, 2) for b in d['bbox']]}
                        for d in result['wear_result']['detections']
                    ],
                })
            else:
                st.info('Wear analysis was not executed (non-tyre detected).')
        with det_tabs[2]:
            st.json({
                'gatekeeper_result': result['gatekeeper_result'],
                'wear_result': result['wear_result']['final_class'] if result['wear_result'] else None,
                'final_class': result['final_class'],
                'confidence': round(result['confidence'], 4),
            })

    annotated = result.get('wear_vis') or result.get('gatekeeper_vis')
    if annotated is not None:
        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            st.download_button('Download Annotated Image', data=_image_to_bytes(annotated), file_name='inspection_result.png', mime='image/png', width='stretch')
        export_data = {
            'gatekeeper_result': result['gatekeeper_result'],
            'wear_result': result['wear_result']['final_class'] if result['wear_result'] else None,
            'final_class': result['final_class'],
            'confidence': round(result['confidence'], 4),
            'gatekeeper_confidence': round(result['gatekeeper_confidence'], 4),
            'gatekeeper_reason': result['gatekeeper_reason'],
        }
        with dl_col2:
            st.download_button('Download Inspection Summary', data=json.dumps(export_data, indent=2), file_name='inspection_result.json', mime='application/json', width='stretch')

    with st.expander('How It Works', expanded=False):
        st.markdown("""
**1. Tyre Verification** -- A computer vision model detects mining truck tyres in the image.

**2. Wear Detection** -- A second model inspects the tyre surface for cuts and damage.

**3. Final Classification:** Cut detected -> **Damaged** | Tyre only -> **Good** | Not a tyre -> **Non-Tyre**
        """)


# ======================================================================
# TAB 2 - Telemetry Analytics
# ======================================================================

def render_sensor_tab():
    st.subheader('Telemetry Analytics')
    st.markdown('---')

    with st.spinner('Loading telemetry data...'):
        data = _load_osi_data()
    stats = data['stats']
    events = data['events']
    severe_count = int((events['level'] == 'Severe').sum()) if len(events) > 0 else 0
    critical_count = int((events['level'] == 'Critical').sum()) if len(events) > 0 else 0
    stats['severe_count'] = severe_count
    stats['critical_count'] = critical_count
    sev_dist = data['sev_dist']
    comp_means = data['comp_means']

    with st.expander('Filters', expanded=False):
        fcol1, fcol2, fcol3 = st.columns(3)
        with fcol1:
            severity_filter = st.multiselect('Severity Level', ['Normal', 'Moderate', 'Severe', 'Critical'], default=['Moderate', 'Severe', 'Critical'], key='osi_severity_filter')
        with fcol2:
            component_filter = st.multiselect('Risk Component', [c.replace('_Risk_Score', '') for c in SCORE_COLS], default=[c.replace('_Risk_Score', '') for c in SCORE_COLS], key='osi_component_filter')
        with fcol3:
            info = data['info']
            st.metric('Data Duration', str(info['duration']))
            st.metric('Records', f'{info["records"]:,}')

    mcols = st.columns(6)
    for i, (label, val) in enumerate([('Average Severity', stats['mean_osi']), ('Peak Severity', stats['max_osi']), ('95th Percentile', stats['p95_osi'])]):
        with mcols[i]:
            color = '#2ecc71' if val <= 25 else '#f1c40f' if val <= 50 else '#e67e22' if val <= 75 else '#e74c3c'
            st.markdown(_build_metric_card(label, f'{val:.1f}', color), unsafe_allow_html=True)
    with mcols[3]:
        health, _ = get_health_status(stats['mean_osi'], stats['max_osi'])
        hc = {'HEALTHY': '#2ecc71', 'MODERATE': '#f1c40f', 'ELEVATED': '#e67e22', 'CRITICAL': '#e74c3c'}
        st.markdown(_build_metric_card('Vehicle Operating Condition', health, hc.get(health, '#999')), unsafe_allow_html=True)
    with mcols[4]:
        st.markdown(_build_metric_card('Detected High-Risk Events', str(stats['severe_count']), '#e67e22'), unsafe_allow_html=True)
    with mcols[5]:
        st.markdown(_build_metric_card('Critical Events', str(stats['critical_count']), '#e74c3c'), unsafe_allow_html=True)

    st.markdown('---')
    st.subheader('Current Operational Severity')
    osi_sample = data['osi_sample']
    current_osi = float(osi_sample['OSI'].iloc[-1]) if len(osi_sample) > 0 else stats.get('max_osi', 0.0)
    col1, col2 = st.columns([1, 2])
    with col1:
        st.plotly_chart(plot_osi_gauge(current_osi), width='stretch')
    with col2:
        st.table(pd.DataFrame({'Level': ['Normal', 'Moderate', 'Severe', 'Critical'], 'Range': ['0--25', '25--50', '50--75', '75--100'], 'Indication': ['Within limits', 'Elevated stress', 'High stress', 'Unsafe']}))

    st.markdown('---')
    st.subheader('Severity Trend')
    st.plotly_chart(plot_osi_timeseries(data['osi_sample']), width='stretch')

    st.markdown('---')
    st.subheader('Severity Distribution')
    sc1, sc2 = st.columns(2)
    with sc1:
        st.plotly_chart(plot_severity_pie(sev_dist), width='stretch')
    with sc2:
        st.plotly_chart(plot_severity_bar(sev_dist), width='stretch')
    st.dataframe(pd.DataFrame([{'Level': k, '% Time': f'{v:.2f}%'} for k, v in sev_dist.items()]), width='stretch', hide_index=True)

    st.markdown('---')
    full_to_short = {c: c.replace('_Risk_Score', '') for c in SCORE_COLS}
    short_to_full = {v: k for k, v in full_to_short.items()}
    filtered_means = {}
    for short in component_filter:
        full = short_to_full.get(short)
        if full and full in comp_means:
            filtered_means[full] = comp_means[full]

    if filtered_means:
        dominant = max(filtered_means, key=filtered_means.get)
        st.subheader('Operational Risk Analysis')
        st.markdown('**Dominant risk factor:** ' + dominant.replace('_Risk_Score', '') + ' (mean = ' + f'{filtered_means[dominant]:.2f}' + ')')
        ct1, ct2, ct3 = st.tabs(['Risk Profile', 'Weighted Contributions', 'Trend'])
        with ct1:
            st.plotly_chart(plot_radar_chart(filtered_means), width='stretch')
        with ct2:
            w = {k: OSI_WEIGHTS.get(k, 0) for k in filtered_means}
            st.plotly_chart(plot_component_contributions(filtered_means, w), width='stretch')
        with ct3:
            comp_sample = data['comp_sample']
            filtered_full = [c for c in SCORE_COLS if c in filtered_means]
            if 'timestamp' in comp_sample.columns:
                st.plotly_chart(plot_stacked_area(comp_sample, filtered_full), width='stretch')
    else:
        st.info('Select at least one risk component.')

    st.markdown('---')
    st.subheader('Detected High-Risk Events')
    events = data['events']
    if len(events) > 0:
        display = events.copy()
        if 'duration_seconds' in display.columns:
            display['duration_min'] = display['duration_seconds'] / 60
        display['peak_osi'] = display['peak_osi'].round(1)
        display['mean_osi'] = display['mean_osi'].round(1)

        def _color_level(val):
            colors = {'Critical': '#e74c3c', 'Severe': '#e67e22', 'Moderate': '#f1c40f'}
            return 'background-color: ' + colors.get(val, 'white') + '30'

        st.dataframe(display.style.map(_color_level, subset=['level']), width='stretch', hide_index=True)
        ec1, ec2 = st.columns(2)
        with ec1:
            csv = display.to_csv(index=False).encode('utf-8')
            st.download_button('Download Detected Events', data=csv, file_name='detected_events.csv', mime='text/csv')
        with ec2:
            st.write('**Total events:** ' + str(len(events)))
            for level in ['Critical', 'Severe', 'Moderate']:
                cnt = len(events[events['level'] == level])
                st.write('**' + level + ':** ' + str(cnt))
    else:
        st.info('No high-risk events detected in this dataset.')

    st.markdown('---')
    st.subheader('Operational Trends')
    dt1, dt2 = st.tabs(['Daily', 'Weekly'])
    with dt1:
        daily = data['daily']
        if len(daily) > 0:
            st.plotly_chart(plot_daily_trends(daily), width='stretch')
            with st.expander('View Daily Data'):
                st.dataframe(daily, width='stretch', hide_index=True)
    with dt2:
        weekly = data['weekly']
        if len(weekly) > 0:
            st.plotly_chart(plot_weekly_trends(weekly), width='stretch')
            with st.expander('View Weekly Data'):
                st.dataframe(weekly, width='stretch', hide_index=True)

    st.markdown('---')
    st.subheader('Abnormal Operating Behaviour')
    from src.severity_sensor.dashboard.dashboard_loader import load_anomaly_sample, load_top_anomalies
    anomaly_data = load_anomaly_sample()
    if len(anomaly_data) > 0 and 'Anomaly_Score' in anomaly_data.columns:
        comp_sample = data['comp_sample']
        at1, at2, at3 = st.tabs(['Distribution', 'Top Anomalies', 'Timeline'])
        with at1:
            st.plotly_chart(plot_anomaly_distribution(anomaly_data['Anomaly_Score']), width='stretch')
        with at2:
            top_df = load_top_anomalies(20)
            st.dataframe(top_df, width='stretch', hide_index=True)
        with at3:
            if 'timestamp' in comp_sample.columns:
                st.plotly_chart(plot_anomaly_timeline(comp_sample), width='stretch')
    else:
        st.info('Abnormality scores not available for this dataset.')

    st.markdown('---')
    st.subheader('Recommended Maintenance Actions')
    ordered = sorted(comp_means.items(), key=lambda x: -x[1])
    dominant = ordered[0][0] if ordered else 'N/A'
    status, desc = get_health_status(stats['mean_osi'], stats['max_osi'])
    rc1, rc2 = st.columns(2)
    with rc1:
        st.markdown('**Current Status:** `' + status + '`')
        st.markdown('*' + desc + '*')
        recs = get_component_recommendations(dominant, ordered)
        for r in recs:
            st.markdown(r)
    with rc2:
        general = get_general_recommendations(stats['mean_osi'], stats['max_osi'], stats['severe_count'], stats['critical_count'])
        for r in general:
            st.markdown(r)
        if not general and not any('**' in r for r in recs):
            st.success('No urgent recommendations. Continue regular monitoring.')

    st.markdown('---')
    st.subheader('Download Summary')
    st.markdown('Export summary data for offline review. Large raw datasets are not streamed directly.')
    export_files = {
        'Daily Summary': 'outputs/osi_phase4/dashboard_data/daily_summary.csv',
        'Detected Events': 'outputs/osi_phase4/dashboard_data/severity_events.csv',
    }
    ecols = st.columns(2)
    for i, (label, rel_path) in enumerate(export_files.items()):
        with ecols[i]:
            path = ROOT / rel_path
            if path.exists():
                df_export = pd.read_csv(path)
                if label == 'Daily Summary' and len(df_export) > 5000:
                    df_export = df_export.head(5000)
                elif label == 'Detected Events' and len(df_export) > 2000:
                    df_export = df_export.head(2000)
                csv_data = df_export.to_csv(index=False).encode('utf-8')
                st.download_button('Download ' + label, data=csv_data, file_name=rel_path.split('/')[-1], mime='text/csv')
            else:
                st.warning(label + ' not available')


# ======================================================================
# TAB 3 - Vehicle Health Report
# ======================================================================

def render_combined_tab():
    st.subheader('Vehicle Health Report')
    st.markdown('---')

    osi_data = _load_osi_data()
    stats = osi_data['stats']
    events = osi_data['events']
    severe_count = int((events['level'] == 'Severe').sum()) if len(events) > 0 else 0
    critical_count = int((events['level'] == 'Critical').sum()) if len(events) > 0 else 0
    stats['severe_count'] = severe_count
    stats['critical_count'] = critical_count
    sev_dist = osi_data['sev_dist']
    comp_means = osi_data['comp_means']
    ordered = sorted(comp_means.items(), key=lambda x: -x[1])
    dominant = ordered[0][0] if ordered else 'N/A'

    image_result = st.session_state.get('unified_image_result')
    image_class = image_result['final_class'] if image_result else None
    image_available = image_result is not None

    osi_level = classify_sensor_health(stats['mean_osi'])

    if not image_available:
        st.warning('No visual inspection result available. Run a scan in the **Computer Vision Inspection** tab first.')
        image_class = 'Non-Tire'

    health_color, health_label = classify_overall_health(image_class, osi_level)
    img_score, sen_score, total_score = compute_health_score(image_class, stats['mean_osi'])

    bg_color = HEALTH_COLORS.get(health_color, '#999')
    st.markdown(
        '<div style="background:' + bg_color + '15;padding:2rem;border-radius:16px;border:3px solid ' + bg_color + ';text-align:center;margin-bottom:1.5rem;">'
        '<h1 style="margin:0;color:' + bg_color + ';font-size:3rem;">' + health_color + '</h1>'
        '<h3 style="margin:0.3rem 0 0;color:' + bg_color + ';">' + health_label + '</h3>'
        '<p style="margin:0.5rem 0 0;font-size:1.2rem;">Health Score: <strong>' + str(total_score) + '</strong> / 100</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('### Visual Inspection')
        if image_available:
            cls = image_class
            icon = 'GOOD' if cls == 'Good-Tire' else 'DAMAGED' if cls == 'Bad-Tire' else 'NON-TYRE'
            bc = '#28a745' if cls == 'Good-Tire' else '#dc3545' if cls == 'Bad-Tire' else '#17a2b8'
            st.markdown(
                '<div style="background:#f8f9fa;padding:1rem;border-radius:8px;border-left:4px solid ' + bc + ';">'
                '<h4 style="margin:0">' + icon + ' ' + cls + '</h4>'
                '<p style="margin:0.3rem 0 0;color:#666;">Confidence: ' + f'{image_result["confidence"]:.1%}' + '</p>'
                '<p style="margin:0;color:#666;">Tyre Verification: ' + image_result["gatekeeper_result"] + '</p>'
                '</div>',
                unsafe_allow_html=True,
            )
            if img_score is not None:
                st.metric('Visual Score', f'{img_score:.0f} / 100')
        else:
            st.info('No scan data available.')
    with col2:
        st.markdown('### Telemetry Analysis')
        osi_icons = {'Normal': 'NORMAL', 'Moderate': 'MODERATE', 'Severe': 'SEVERE', 'Critical': 'CRITICAL'}
        st.markdown(
            '<div style="background:#f8f9fa;padding:1rem;border-radius:8px;border-left:4px solid ' + SEVERITY_COLORS.get(osi_level, '#999') + ';">'
            '<h4 style="margin:0">' + osi_icons.get(osi_level, '') + ' ' + osi_level + '</h4>'
            '<p style="margin:0.3rem 0 0;color:#666;">Mean Severity: ' + f'{stats["mean_osi"]:.2f}' + '</p>'
            '<p style="margin:0;color:#666;">Peak Severity: ' + f'{stats["max_osi"]:.2f}' + '</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.metric('Telemetry Score', f'{sen_score:.1f} / 100')

    st.markdown('---')
    st.subheader('Maintenance Recommendations')
    recs = generate_combined_recommendations(image_class, osi_level, dominant, comp_means)
    for r in recs:
        st.markdown(r)

    st.markdown('---')
    st.subheader('Severity Distribution')
    sc1, sc2 = st.columns(2)
    with sc1:
        st.plotly_chart(plot_severity_pie(sev_dist), width='stretch', key='combined_pie')
    with sc2:
        st.metric('Mean Severity', f'{stats["mean_osi"]:.2f}')
        st.metric('Peak Severity', f'{stats["max_osi"]:.2f}')

    st.subheader('Top Risk Components')
    top_components = sorted(comp_means.items(), key=lambda x: -x[1])[:5]
    comp_df = pd.DataFrame([{'Component': k.replace('_Risk_Score', ''), 'Mean Score': f'{v:.2f}'} for k, v in top_components])
    st.dataframe(comp_df, width='stretch', hide_index=True)

    st.markdown('---')
    st.subheader('Export Health Report')
    if st.button('Generate Vehicle Health Report', type='primary', width='stretch'):
        with st.spinner('Generating report...'):
            report_path = generate_combined_report(
                image_result or {'final_class': 'Non-Tire', 'confidence': 0.0, 'gatekeeper_result': 'Non-Tyre', 'gatekeeper_confidence': 0.0, 'gatekeeper_reason': '', 'wear_detail': 'Not available'},
                stats, sev_dist, comp_means, dominant,
            )
        st.success('Report saved to `' + str(report_path) + '`')
        if report_path.exists():
            with open(report_path, 'r', encoding='utf-8') as f:
                st.download_button('Download Health Report', data=f.read(), file_name='vehicle_health_report.md', mime='text/markdown', width='stretch')


# ======================================================================
# Sidebar
# ======================================================================

def render_sidebar():
    st.sidebar.image('https://cdn-icons-png.flaticon.com/512/1995/1995515.png', width=60)
    st.sidebar.title('Tyre Health Monitor')
    st.sidebar.markdown('---')
    st.sidebar.markdown('**Mining Truck Tyre Health Monitoring System**')
    st.sidebar.markdown('')
    st.sidebar.caption('Computer Vision Inspection')
    st.sidebar.caption('Telemetry Analytics')
    st.sidebar.markdown('---')

    st.sidebar.subheader('Data Source')
    data_mode = st.sidebar.radio('Mode', ['View Precomputed Analysis', 'Analyse New Sensor Data'], key='data_mode', label_visibility='collapsed')

    if data_mode == 'View Precomputed Analysis':
        _render_precomputed_selector()
    else:
        _render_upload_mode()

    st.sidebar.markdown('---')

    if 'unified_image_result' in st.session_state:
        fc = st.session_state['unified_image_result']['final_class']
        st.sidebar.success('Visual: ' + fc)

    try:
        osi_data = _load_osi_data()
        osi_stats = osi_data['stats']
        current_osi = osi_stats.get('max_osi', 0.0)
        osi_sample = osi_data.get('osi_sample')
        if osi_sample is not None and len(osi_sample) > 0:
            current_osi = float(osi_sample['OSI'].iloc[-1])
        status, _ = get_health_status(current_osi, osi_stats.get('max_osi', 0.0))
        st.sidebar.info('Current Health: ' + status)
    except Exception:
        st.sidebar.warning('Telemetry data not loaded')

    st.sidebar.markdown('---')
    st.sidebar.caption('Tyre Health Monitoring System')


def _render_precomputed_selector():
    available = get_available_datasets()
    if not available:
        st.sidebar.info('Building demo datasets...')
        try:
            success, msg = build_demo_datasets()
            if success:
                available = get_available_datasets()
                st.sidebar.success('Demo datasets ready')
            else:
                st.sidebar.warning(msg)
        except Exception as e:
            st.sidebar.error('Could not build demo datasets: ' + str(e))

    descriptions = get_dataset_descriptions()
    if available:
        selected = st.sidebar.selectbox('Operating Scenario', available, key='demo_scenario')
        if selected:
            st.sidebar.caption(descriptions.get(selected, ''))
            demo_dir = get_demo_dir_for_scenario(selected)
            set_data_root(demo_dir, demo_dir)
            st.session_state['data_root_set'] = True
    else:
        st.sidebar.warning('No precomputed datasets available.')
        st.sidebar.info('Using default dataset.')
        reset_data_root()


def _render_upload_mode():
    st.sidebar.caption('Upload raw sensor data for analysis.')
    uploaded = st.sidebar.file_uploader('Upload CSV or Parquet', type=['csv', 'parquet'], key='sensor_upload', help='Upload a telemetry dataset for analysis')
    if uploaded is not None:
        try:
            tmp_dir = Path(tempfile.mkdtemp(prefix='tyre_analysis_'))
            upload_path = tmp_dir / uploaded.name
            with open(upload_path, 'wb') as f:
                f.write(uploaded.getvalue())
        except Exception as e:
            st.sidebar.error(f'Could not save uploaded file: {e}')
            return
        if st.sidebar.button('Run Analysis', type='primary', width='stretch'):
            output_dir = tmp_dir / 'results'
            progress_bar = st.sidebar.progress(0, text='Starting analysis...')

            def _progress_callback(stage, frac):
                progress_bar.progress(frac, text=stage)

            try:
                result = run_full_pipeline(combined_path=upload_path, output_dir=output_dir, progress_callback=_progress_callback)
            except MemoryError:
                st.sidebar.error('Analysis failed: not enough memory. Try a smaller dataset.')
                return
            except Exception as e:
                st.sidebar.error(f'Analysis failed: {e}')
                return
            if result['success']:
                st.sidebar.success(result['message'])
                telemetry_dir = output_dir / 'telemetry'
                dashboard_dir = output_dir / 'dashboard_data'
                set_data_root(telemetry_dir, dashboard_dir)
                st.session_state['data_root_set'] = True
                st.rerun()
            else:
                st.sidebar.error(result['message'])
    else:
        st.sidebar.info('Upload a CSV or Parquet file to begin.')


# ======================================================================
# Main
# ======================================================================

def main():
    render_sidebar()
    tab1, tab2, tab3 = st.tabs(['Computer Vision Inspection', 'Telemetry Analytics', 'Vehicle Health Report'])
    with tab1:
        render_image_tab()
    with tab2:
        render_sensor_tab()
    with tab3:
        render_combined_tab()


if __name__ == '__main__':
    main()
