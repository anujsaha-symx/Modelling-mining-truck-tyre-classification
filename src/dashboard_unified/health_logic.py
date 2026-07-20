"""
Unified Health Logic
====================
Combines image-based inspection results with sensor-based operational
severity to produce overall tyre health status, colour-coded alerts,
and a composite health score.
"""

from typing import Tuple, Optional

HEALTH_LEVELS = ['GREEN', 'YELLOW', 'ORANGE', 'RED']
HEALTH_COLORS = {
    'GREEN': '#2ecc71',
    'YELLOW': '#f1c40f',
    'ORANGE': '#e67e22',
    'RED': '#e74c3c',
}


def classify_image_health(final_class: str) -> str:
    """Map the wear-detection final class to a canonical label."""
    mapping = {
        'Good-Tire': 'Good-Tire',
        'Bad-Tire': 'Bad-Tire',
        'Non-Tire': 'Non-Tire',
    }
    return mapping.get(final_class, 'Non-Tire')


def classify_sensor_health(osi_value: float) -> str:
    """Map an operational severity value to a severity level."""
    if osi_value > 75:
        return 'Critical'
    if osi_value > 50:
        return 'Severe'
    if osi_value > 25:
        return 'Moderate'
    return 'Normal'


def classify_overall_health(
    image_class: str,
    osi_level: str,
) -> Tuple[str, str]:
    """Determine overall tyre health.

    Returns (colour_label, short_description).

    Rules (evaluated in order):
      1. Bad-Tire OR Critical severity     -> RED
      2. Bad-Tire OR Severe severity       -> ORANGE
      3. Good-Tire AND Moderate severity   -> YELLOW
      4. Good-Tire AND Normal severity     -> GREEN
      5. default                           -> YELLOW
    """
    if image_class == 'Bad-Tire' or osi_level == 'Critical':
        return 'RED', 'Critical'
    if image_class == 'Bad-Tire' or osi_level == 'Severe':
        return 'ORANGE', 'Elevated'
    if image_class == 'Good-Tire' and osi_level == 'Moderate':
        return 'YELLOW', 'Moderate'
    if image_class == 'Good-Tire' and osi_level == 'Normal':
        return 'GREEN', 'Healthy'
    return 'YELLOW', 'Moderate'


def compute_health_score(
    image_class: str,
    mean_osi: float,
) -> Tuple[Optional[float], Optional[float], float]:
    """Compute overall health score in [0, 100].

    Composition:
      50% visual inspection pipeline
      50% sensor pipeline (100 - mean severity)

    Returns (image_component, sensor_component, overall).
    Image component is None for Non-Tire (N/A).
    """
    if image_class == 'Good-Tire':
        image_score = 100.0
    elif image_class == 'Bad-Tire':
        image_score = 30.0
    else:
        image_score = None

    sensor_score = max(0.0, 100.0 - mean_osi)

    if image_score is None:
        overall = sensor_score
    else:
        overall = 0.5 * image_score + 0.5 * sensor_score

    return image_score, sensor_score, round(overall, 1)


def generate_combined_recommendations(
    image_class: str,
    osi_level: str,
    dominant_component: str,
    component_means: dict,
) -> list:
    """Generate recommendations based on combined visual + sensor state."""
    recs = []

    if image_class == 'Bad-Tire':
        recs.append('**WARNING: Visual damage detected - immediate physical inspection required.**')

    if osi_level == 'Critical':
        recs.append('**CRITICAL: Operational severity index indicates unsafe conditions. Immediate review required.**')
    elif osi_level == 'Severe':
        recs.append('**CAUTION: Severe operational severity - increased monitoring frequency recommended.**')

    if image_class == 'Bad-Tire' and any(
        'Load' in k and v > 60 for k, v in component_means.items()
    ):
        recs.append('- Immediate inspection recommended (visual damage + high load stress).')

    if image_class == 'Bad-Tire' and any(
        'Usage' in k and v > 60 for k, v in component_means.items()
    ):
        recs.append('- High-usage tyre with visible damage - consider replacement evaluation.')

    if image_class == 'Good-Tire' and osi_level in ('Severe', 'Critical'):
        recs.append(
            '- No visible damage, but operational conditions are harsh. '
            'Monitor sensor trends closely.'
        )

    if image_class == 'Good-Tire' and osi_level == 'Moderate':
        recs.append(
            '- Tyre appears undamaged, but moderate severity suggests '
            'operational stress. Periodic inspection advised.'
        )

    if osi_level == 'Normal' and image_class == 'Good-Tire':
        recs.append('- Tyre is healthy across both inspections. Continue regular monitoring.')

    if image_class == 'Non-Tire':
        recs.append(
            '- Image inspection returned Non-Tire. Sensor health is based on telemetry data alone.'
        )

    short_dom = dominant_component.replace('_Risk_Score', '') if dominant_component else ''
    if short_dom and osi_level in ('Moderate', 'Severe'):
        recs.append(f'- Dominant sensor risk: **{short_dom}** - review related sub-systems.')

    if not recs:
        recs.append('- No urgent recommendations. Continue regular monitoring.')

    return recs
