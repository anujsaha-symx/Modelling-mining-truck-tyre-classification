import json
import copy
import os

ANNOTATIONS_PATH = 'datasets/annotated/annotations.coco.json'
OUTPUT_PATH = 'datasets/annotated/annotations_wear_v2.coco.json'

V2_CATEGORIES = [
    {'id': 0, 'name': 'Tire', 'supercategory': 'none'},
    {'id': 1, 'name': 'Cut', 'supercategory': 'none'},
    {'id': 2, 'name': 'Non-Tire', 'supercategory': 'none'},
]


def _infer_source_class(filename):
    if filename.startswith('good_'):
        return 'good'
    if filename.startswith('bad_'):
        return 'bad'
    if filename.startswith('negative_'):
        return 'negative'
    return None


def convert():
    with open(ANNOTATIONS_PATH, 'r') as f:
        coco = json.load(f)

    print(f'Original categories: {[c["name"] for c in coco["categories"]]}')
    print(f'Total images: {len(coco["images"])}')
    print(f'Total annotations: {len(coco["annotations"])}')

    new_coco = copy.deepcopy(coco)
    new_coco['categories'] = V2_CATEGORIES

    image_map = {img['id']: img for img in coco['images']}

    new_annotations = []
    next_ann_id = 1
    stats = {'good': 0, 'bad': 0, 'negative': 0, 'unknown': 0}
    cut_created = 0

    for ann in coco['annotations']:
        img_id = ann['image_id']
        img_info = image_map.get(img_id)
        if img_info is None:
            continue

        filename = img_info['file_name']
        source = _infer_source_class(filename)
        if source is None:
            stats['unknown'] += 1
            continue

        x, y, w, h = ann['bbox']

        if source == 'good':
            new_ann = copy.deepcopy(ann)
            new_ann['category_id'] = 0
            new_ann['id'] = next_ann_id
            next_ann_id += 1
            new_annotations.append(new_ann)
            stats['good'] += 1

        elif source == 'bad':
            tire_ann = copy.deepcopy(ann)
            tire_ann['category_id'] = 0
            tire_ann['id'] = next_ann_id
            next_ann_id += 1
            new_annotations.append(tire_ann)

            cut_ann = copy.deepcopy(ann)
            cut_ann['category_id'] = 1
            cut_ann['id'] = next_ann_id
            next_ann_id += 1
            new_annotations.append(cut_ann)
            cut_created += 1
            stats['bad'] += 1

        elif source == 'negative':
            new_ann = copy.deepcopy(ann)
            new_ann['category_id'] = 2
            new_ann['id'] = next_ann_id
            next_ann_id += 1
            new_annotations.append(new_ann)
            stats['negative'] += 1

    new_coco['annotations'] = new_annotations

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(new_coco, f, indent=2)

    print(f'\nConversion complete:')
    print(f'  Tire annotations (from good/):     {stats["good"]}')
    print(f'  Tire + Cut annotations (from bad/): {stats["bad"]} each = {stats["bad"] * 2}')
    print(f'    -> Cut annotations created:       {cut_created}')
    print(f'  Non-Tire annotations (from negative/): {stats["negative"]}')
    print(f'  Total annotations in output:        {len(new_annotations)}')
    if stats['unknown']:
        print(f'  Unknown:                             {stats["unknown"]}')
    print(f'\nSaved to {OUTPUT_PATH}')


if __name__ == '__main__':
    convert()
