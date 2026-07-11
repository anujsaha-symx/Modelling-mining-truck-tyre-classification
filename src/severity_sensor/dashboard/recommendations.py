"""
Maintenance Recommendations Engine
====================================
Generates automated maintenance recommendations based on current
operational severity and dominant risk components.
"""

from typing import List, Dict, Tuple


SEVERITY_LEVELS = {
    'Normal':   (0, 25),
    'Moderate': (25, 50),
    'Severe':   (50, 75),
    'Critical': (75, 100),
}


def get_health_status(mean_osi: float, max_osi: float) -> Tuple[str, str]:
    """Return (status_label, description)."""
    if max_osi > 75:
        return 'CRITICAL', 'Immediate inspection required. Operation may be unsafe.'
    if max_osi > 50:
        return 'ELEVATED', 'Increased monitoring frequency. Investigate dominant risk factors.'
    if mean_osi > 25:
        return 'MODERATE', 'Periodic inspection recommended. Monitor load and usage components.'
    return 'HEALTHY', 'Continue regular monitoring. No immediate action required.'


def get_component_recommendations(dominant_component: str,
                                  component_order: List[Tuple[str, float]]) -> List[str]:
    """Generate specific recommendations based on risk component analysis."""
    recs = []

    component_advice = {
        'Pressure_Risk_Score': [
            'Check TPMS system and inflation schedule.',
            'Inspect tyres for punctures or slow leaks.',
            'Verify pressure settings match manufacturer specifications.',
        ],
        'Thermal_Risk_Score': [
            'Inspect tyre pressure and cooling systems.',
            'Check for brake drag generating excess heat.',
            'Reduce sustained high-speed operation in hot conditions.',
        ],
        'Load_Risk_Score': [
            'Reduce overload operations and inspect suspension.',
            'Verify payload distribution is within rated limits.',
            'Check load cells and payload monitoring system.',
        ],
        'Vibration_Risk_Score': [
            'Inspect suspension components for wear.',
            'Check wheel balance and alignment.',
            'Inspect road surface conditions on frequently travelled routes.',
        ],
        'Braking_Risk_Score': [
            'Inspect brake system for dragging or excessive use.',
            'Check brake adjustment and pad wear.',
            'Review operator behaviour for aggressive braking patterns.',
        ],
        'Terrain_Risk_Score': [
            'Inspect tyres for uneven wear from rough terrain.',
            'Consider route optimisation to avoid severe terrain.',
            'Check suspension and chassis for damage.',
        ],
        'Usage_Risk_Score': [
            'Review duty cycle and operational profile.',
            'Consider tyre upgrades for high-usage applications.',
            'Monitor distance-based wear indicators.',
        ],
        'Anomaly_Score': [
            'Investigate anomalous operating conditions.',
            'Review sensor data for intermittent faults.',
            'Schedule comprehensive vehicle inspection.',
        ],
    }

    if dominant_component in component_advice:
        recs.append(f'**Primary Risk - {dominant_component.replace("_Risk_Score", "")}**')
        recs.extend(f'- {a}' for a in component_advice[dominant_component])

    elevated = [(name, val) for name, val in component_order
                if name != dominant_component and val > 30]
    if elevated:
        recs.append('')
        recs.append('**Secondary Observations**')
        for name, val in elevated:
            short = name.replace('_Risk_Score', '')
            recs.append(f'- {short} (mean={val:.1f}) - monitor for escalation.')

    return recs


def get_general_recommendations(mean_osi: float, max_osi: float,
                                 severe_count: int, critical_count: int) -> List[str]:
    """General operational recommendations."""
    recs = []
    if critical_count > 0:
        recs.append('**URGENT**: Critical severity events detected. Immediate operational review required.')
    if severe_count > 5:
        recs.append(f'Frequent severe events ({severe_count} detected). Review operating conditions.')
    if mean_osi > 30:
        recs.append('Sustained moderate severity - consider preventive maintenance scheduling.')
    if max_osi > 50:
        recs.append('Peak severity exceeds threshold - review high-severity periods for root causes.')
    return recs
