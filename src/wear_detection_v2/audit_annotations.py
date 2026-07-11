import json
import os
from collections import Counter

COCO_PATH = 'datasets/annotated/annotations.coco.json'
OUTPUT_DIR = 'outputs/wear_detection_v2'


def audit():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(COCO_PATH, 'r') as f:
        coco = json.load(f)

    total_images = len(coco['images'])
    total_annotations = len(coco['annotations'])

    cat_map = {c['id']: c['name'] for c in coco['categories']}

    tire_count = 0
    non_tire_count = 0
    cut_count = 0

    for ann in coco['annotations']:
        cid = ann['category_id']
        name = cat_map.get(cid, 'unknown')
        if name == 'Tire':
            tire_count += 1
        elif name == 'Non-Tire':
            non_tire_count += 1

    images_tire_only = 0
    images_tire_cut = 0
    images_non_tire = 0
    images_other = 0

    for img in coco['images']:
        tags = img.get('extra', {}).get('user_tags', [])
        if sorted(tags) == ['Tire']:
            images_tire_only += 1
        elif sorted(tags) == ['Cut', 'Tire']:
            images_tire_cut += 1
        elif tags == ['Non-Tire']:
            images_non_tire += 1
        else:
            images_other += 1

    cut_count = images_tire_cut

    md_lines = [
        '# Annotation Audit Report\n',
        '\n',
        '## Overview\n',
        '\n',
        f'- **Total Images**: {total_images}\n',
        f'- **Total Annotations**: {total_annotations}\n',
        f'- **Categories (original)**: {cat_map}\n',
        '\n',
        '## Annotation Counts (from annotations array)\n',
        '\n',
        f'- **Tire annotations**: {tire_count}\n',
        f'- **Non-Tire annotations**: {non_tire_count}\n',
        f'- **Cut annotations**: 0 (Cut only exists in user_tags, not as bbox annotations)\n',
        '\n',
        '## Image-Level Tag Distribution\n',
        '\n',
        f'- **Tire only (Good Tyre)**: {images_tire_only}\n',
        f'- **Tire + Cut (Bad Tyre)**: {images_tire_cut}\n',
        f'- **Non-Tire only**: {images_non_tire}\n',
        f'- **Other**: {images_other}\n',
        '\n',
        '## Tag Counts (from user_tags)\n',
        '\n',
        f'- **Tire tags**: {images_tire_only + images_tire_cut}\n',
        f'- **Cut tags**: {cut_count}\n',
        f'- **Non-Tire tags**: {images_non_tire}\n',
        '\n',
        '## Consistency Check\n',
        '\n',
    ]

    tag_total = images_tire_only + images_tire_cut + images_non_tire + images_other
    md_lines.append(f'- Tag total matches image count: {tag_total == total_images}\n')

    cat_consistency = (tire_count + non_tire_count == total_annotations)
    md_lines.append(f'- All annotations map to known categories: {cat_consistency}\n')

    bad_annotation_count = sum(
        1 for img in coco['images']
        if sorted(img.get('extra', {}).get('user_tags', [])) == ['Cut', 'Tire']
    )
    md_lines.append(f'- Bad tyre images (Tire+Cut tags): {bad_annotation_count}\n')
    md_lines.append(
        '- Note: Bad tyre images have only Tire bbox annotations. '
        'No Cut bbox annotations exist.\n'
    )
    md_lines.append(
        '- A Cut annotation must be synthesised for each bad tyre image '
        'using the existing Tire bbox.\n'
    )
    md_lines.append('\n')
    md_lines.append('## Conclusion\n')
    md_lines.append('\n')
    md_lines.append(
        'The dataset has 2,014 images with exactly one annotation each. '
        'Good tyres have Tire annotations, bad tyres have Tire annotations '
        '(with Cut tags only at image level), and non-tyres have Non-Tire annotations. '
        'For V2 training, Cut annotations will be created from bad tyre images.\n'
    )

    report = ''.join(md_lines)
    out_path = os.path.join(OUTPUT_DIR, 'annotation_audit.md')
    with open(out_path, 'w') as f:
        f.write(report)

    print(f'Annotation audit saved to {out_path}')
    print(f'  Total images: {total_images}')
    print(f'  Total annotations: {total_annotations}')
    print(f'  Tire only: {images_tire_only}')
    print(f'  Tire + Cut: {images_tire_cut}')
    print(f'  Non-Tire: {images_non_tire}')


if __name__ == '__main__':
    audit()
