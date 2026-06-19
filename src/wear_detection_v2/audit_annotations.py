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

    tag_total = images_tire_only + images_tire_cut + images_non_tire + images_other
    md_lines.append(f'- Tag total matches image count: {tag_total == total_images}\n')

    cat_consistency = (tire_count + non_tire_count == total_annotations)
    md_lines.append(f'- All annotations map to known categories: {cat_consistency}\n')

    bad_annotation_count = sum(
        1 for img in coco['images']
        if sorted(img.get('extra', {}).get('user_tags', [])) == ['Cut', 'Tire']
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