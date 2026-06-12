import json
import os
from glob import glob

INPUT_PATH = "datasets/resized/_annotations.coco.json"
OUTPUT_PATH = "datasets/resized/annotations_complete.coco.json"
SUMMARY_PATH = "datasets/resized/annotation_summary.json"
GOOD_DIR = "datasets/resized/good"
NEGATIVE_DIR = "datasets/resized/negative"
WIDTH = 224
HEIGHT = 224
FULL_BBOX = [0, 0, WIDTH, HEIGHT]
FULL_AREA = WIDTH * HEIGHT

with open(INPUT_PATH) as f:
    data = json.load(f)

# --- STEP 5: Categories ---
data["categories"] = [
    {"id": 0, "name": "Tire", "supercategory": "none"},
    {"id": 1, "name": "Non-Tire", "supercategory": "none"},
]

# Track originals for reporting
orig_image_count = len(data["images"])
orig_ann_count = len(data["annotations"])

# Get existing image IDs and annotation IDs
existing_img_ids = {img["id"] for img in data["images"]}
existing_ann_ids = {ann["id"] for ann in data["annotations"]}
next_img_id = max(existing_img_ids) + 1
next_ann_id = max(existing_ann_ids) + 1

# ---- STEP 1 & 2: Modify existing bad images ----
for img in data["images"]:
    # STEP 1: Add "Tire" tag
    tags = img.get("extra", {}).get("user_tags", [])
    if "Tire" not in tags:
        tags.append("Tire")
    if "extra" not in img:
        img["extra"] = {}
    img["extra"]["user_tags"] = tags
    # STEP 2: Clean filename
    clean_name = img.get("extra", {}).get("name", "")
    if clean_name:
        img["file_name"] = clean_name
    # Also preserve original license and date_captured
    img.setdefault("date_captured", data.get("info", {}).get("date_created", ""))
    img.setdefault("license", data.get("licenses", [{"id": 1}])[0].get("id", 1))

# Map existing category_ids (old categories were 0 and 1, both "Tire-Annotation")
# All existing annotations should map to Tire (id 0)
for ann in data["annotations"]:
    ann["category_id"] = 0

# ---- STEP 3: Add good images ----
good_files = sorted(glob(os.path.join(GOOD_DIR, "good_*.jpg")))
for fpath in good_files:
    fname = os.path.basename(fpath)
    img_entry = {
        "id": next_img_id,
        "license": 1,
        "file_name": fname,
        "height": HEIGHT,
        "width": WIDTH,
        "date_captured": data.get("info", {}).get("date_created", ""),
        "extra": {
            "user_tags": ["Tire"],
            "name": fname,
        },
    }
    ann_entry = {
        "id": next_ann_id,
        "image_id": next_img_id,
        "category_id": 0,
        "bbox": FULL_BBOX,
        "iscrowd": 0,
        "area": FULL_AREA,
        "segmentation": [],
    }
    data["images"].append(img_entry)
    data["annotations"].append(ann_entry)
    next_img_id += 1
    next_ann_id += 1

# ---- STEP 4: Add negative (non-tire) images ----
neg_files = sorted(glob(os.path.join(NEGATIVE_DIR, "negative_*.jpg")))
for fpath in neg_files:
    fname = os.path.basename(fpath)
    img_entry = {
        "id": next_img_id,
        "license": 1,
        "file_name": fname,
        "height": HEIGHT,
        "width": WIDTH,
        "date_captured": data.get("info", {}).get("date_created", ""),
        "extra": {
            "user_tags": ["Non-Tire"],
            "name": fname,
        },
    }
    ann_entry = {
        "id": next_ann_id,
        "image_id": next_img_id,
        "category_id": 1,
        "bbox": FULL_BBOX,
        "iscrowd": 0,
        "area": FULL_AREA,
        "segmentation": [],
    }
    data["images"].append(img_entry)
    data["annotations"].append(ann_entry)
    next_img_id += 1
    next_ann_id += 1

# ---- STEP 6: Validation ----
img_ids = [img["id"] for img in data["images"]]
ann_ids = [ann["id"] for ann in data["annotations"]]
img_id_set = set(img_ids)
ann_id_set = set(ann_ids)

assert len(img_ids) == len(img_id_set), "Duplicate image IDs found!"
assert len(ann_ids) == len(ann_id_set), "Duplicate annotation IDs found!"

fnames = [img["file_name"] for img in data["images"]]
assert len(fnames) == len(set(fnames)), "Duplicate filenames found!"

for ann in data["annotations"]:
    assert ann["image_id"] in img_id_set, f"Annotation {ann['id']} references missing image {ann['image_id']}"

# ---- Write output ----
with open(OUTPUT_PATH, "w") as f:
    json.dump(data, f, indent=2)

# ---- STEP 7: Summary ----
good_count = len(good_files)
bad_count = orig_image_count
neg_count = len(neg_files)
tire_count = good_count + bad_count
non_tire_count = neg_count
total_images = len(data["images"])
total_annotations = len(data["annotations"])

summary = {
    "good_images": good_count,
    "bad_images": bad_count,
    "negative_images": neg_count,
    "total_images": total_images,
    "tire_images": tire_count,
    "non_tire_images": non_tire_count,
    "total_annotations": total_annotations,
    "categories": data["categories"],
}

with open(SUMMARY_PATH, "w") as f:
    json.dump(summary, f, indent=2)

# ---- STEP 8: Report ----
print("=" * 60)
print("COCO ANNOTATION TRANSFORMATION REPORT")
print("=" * 60)
print(f"Original image count:      {orig_image_count}")
print(f"Final image count:         {total_images}")
print(f"Original annotation count: {orig_ann_count}")
print(f"Final annotation count:    {total_annotations}")
print()
print("Category distribution:")
print(f"  Tire    (id=0): {tire_count} images")
print(f"  Non-Tire(id=1): {non_tire_count} images")
print()
print("Per-class breakdown:")
print(f"  GOOD tyre images:      {good_count}")
print(f"  BAD tyre images:       {bad_count}")
print(f"  NON-TYRE images:       {neg_count}")
print()
print("Sample entries:")
print()
tire_imgs = [img for img in data["images"] if "Tire" in img.get("extra", {}).get("user_tags", [])]
non_tire_imgs = [img for img in data["images"] if "Non-Tire" in img.get("extra", {}).get("user_tags", [])]
print("Tire samples:")
for img in tire_imgs[:3]:
    img_anns = [a for a in data["annotations"] if a["image_id"] == img["id"]]
    print(f"  {img['file_name']:25s} tags={img['extra']['user_tags']} bboxes={len(img_anns)}")
print("Non-Tire samples:")
for img in non_tire_imgs[:3]:
    img_anns = [a for a in data["annotations"] if a["image_id"] == img["id"]]
    print(f"  {img['file_name']:25s} tags={img['extra']['user_tags']} bboxes={len(img_anns)}")
print()
print(f"Summary written to:    {SUMMARY_PATH}")
print(f"Full annotations to:   {OUTPUT_PATH}")
print("=" * 60)
