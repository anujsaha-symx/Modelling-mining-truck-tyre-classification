import json
import os
import copy

ANNOTATIONS_PATH = 'datasets/annotated/annotations.coco.json'
OUTPUT_PATH = 'datasets/annotated/annotations_wear.coco.json'

WEAR_CATEGORIES = [
    {'id': 0, 'name': 'Good-Tire', 'supercategory': 'none'},
    {'id': 1, 'name': 'Bad-Tire', 'supercategory': 'none'},
    {'id': 2, 'name': 'Non-Tire', 'supercategory': 'none'},
]

ORIGINAL_TO_WEAR = {
    'good': {0: 0},
    'bad': {0: 1},
    'negative': {1: 2},
}


def _infer_source(filename):
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

    original_cats = {c['id']: c['name'] for c in coco['categories']}
    print(f'Original categories: {original_cats}')
    print(f'Total images: {len(coco["images"])}')
    print(f'Total annotations: {len(coco["annotations"])}')

    new_coco = copy.deepcopy(coco)
    new_coco['categories'] = WEAR_CATEGORIES

    new_annotations = []
    stats = {'good': 0, 'bad': 0, 'negative': 0, 'unknown': 0}

    for ann in coco['annotations']:
        img_id = ann['image_id']
        img_info = next(img for img in coco['images'] if img['id'] == img_id)
        filename = img_info['file_name']
        source = _infer_source(filename)
        orig_cat_id = ann['category_id']

        if source is None:
            print(f'  WARNING: unknown source for {filename}')
            stats['unknown'] += 1
            continue

        mapping = ORIGINAL_TO_WEAR[source]
        if orig_cat_id not in mapping:
            print(f'  WARNING: unmapped category {orig_cat_id} for {filename}')
            stats['unknown'] += 1
            continue

        new_ann = copy.deepcopy(ann)
        new_ann['category_id'] = mapping[orig_cat_id]
        new_annotations.append(new_ann)
        stats[source] += 1

    new_coco['annotations'] = new_annotations

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(new_coco, f, indent=2)

    print(f'\nConversion complete:')
    print(f'  Good-Tire (from good/): {stats["good"]}')
    print(f'  Bad-Tire  (from bad/):  {stats["bad"]}')
    print(f'  Non-Tire  (from negative/): {stats["negative"]}')
    if stats['unknown']:
        print(f'  Unknown:  {stats["unknown"]}')
    print(f'\nSaved to {OUTPUT_PATH}')


if __name__ == '__main__':
    convert()
