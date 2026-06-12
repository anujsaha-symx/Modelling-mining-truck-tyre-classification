import json

with open("datasets/resized/annotations_complete.coco.json") as f:
    d = json.load(f)

assert "info" in d, "Missing info"
assert "licenses" in d, "Missing licenses"
assert "categories" in d, "Missing categories"
assert "images" in d, "Missing images"
assert "annotations" in d, "Missing annotations"

cats = d["categories"]
assert len(cats) == 2, f"Expected 2 categories, got {len(cats)}"
assert cats[0]["name"] == "Tire"
assert cats[1]["name"] == "Non-Tire"

img_ids = [i["id"] for i in d["images"]]
ann_ids = [a["id"] for a in d["annotations"]]
assert len(img_ids) == len(set(img_ids)), "Duplicate image IDs"
assert len(ann_ids) == len(set(ann_ids)), "Duplicate annotation IDs"

img_id_set = set(img_ids)
for a in d["annotations"]:
    assert a["image_id"] in img_id_set, f"Missing image {a['image_id']} for ann {a['id']}"

fnames = [i["file_name"] for i in d["images"]]
assert len(fnames) == len(set(fnames)), "Duplicate filenames"

bad_imgs = [i for i in d["images"] if "Cut" in i.get("extra", {}).get("user_tags", [])]
assert len(bad_imgs) == 200, f"Expected 200 bad images, got {len(bad_imgs)}"

for i in d["images"]:
    assert ".rf." not in i["file_name"], f"Dirty filename: {i['file_name']}"
    tags = i.get("extra", {}).get("user_tags", [])
    if "Cut" in tags:
        assert "Tire" in tags, f"Bad image missing Tire tag: {i['file_name']}"

good_imgs = [i for i in d["images"] if i["file_name"].startswith("good_")]
assert len(good_imgs) == 1658, f"Expected 1658 good images, got {len(good_imgs)}"
for gi in good_imgs:
    anns = [a for a in d["annotations"] if a["image_id"] == gi["id"]]
    assert len(anns) == 1
    assert anns[0]["bbox"] == [0, 0, 224, 224]
    assert anns[0]["category_id"] == 0

neg_imgs = [i for i in d["images"] if i["file_name"].startswith("negative_")]
assert len(neg_imgs) == 156, f"Expected 156 negative images, got {len(neg_imgs)}"
for ni in neg_imgs:
    anns = [a for a in d["annotations"] if a["image_id"] == ni["id"]]
    assert len(anns) == 1
    assert anns[0]["bbox"] == [0, 0, 224, 224]
    assert anns[0]["category_id"] == 1

assert len(d["images"]) == 2014, f"Expected 2014 images, got {len(d['images'])}"
assert len(d["annotations"]) == 2014, f"Expected 2014 annotations, got {len(d['annotations'])}"

print("ALL VALIDATIONS PASSED")
