import os
import json
import shutil
import random

DATASET_PATH = "dataset"
ANNOTATION_FILE = "annotations.json"
TRAIN_PATH = "train"
TEST_PATH = "test"
TRAIN_ANNOTATION_FILE = "train_annotations.json"
TEST_ANNOTATION_FILE = "test_annotations.json"

# for reproducibility
# random.seed(4)

with open(ANNOTATION_FILE, "r") as f:
    annotations = json.load(f)

all_images = annotations["images"]
all_annotations = annotations["annotations"]

image_id_to_annotations = {}
for ann in all_annotations:
    image_id_to_annotations.setdefault(ann["image_id"], []).append(ann)

#  90% train, 10% test
random.shuffle(all_images)
split_index = int(0.9 * len(all_images))
train_images = all_images[:split_index]
test_images = all_images[split_index:]

train_image_ids = {img["id"] for img in train_images}
test_image_ids = {img["id"] for img in test_images}

train_annotations = [
    ann for ann in all_annotations if ann["image_id"] in train_image_ids
]
test_annotations = [ann for ann in all_annotations if ann["image_id"] in test_image_ids]

os.makedirs(TRAIN_PATH, exist_ok=True)
os.makedirs(TEST_PATH, exist_ok=True)

for img in train_images:
    file_name = f"{img['file_name']}"
    src_path = os.path.join(DATASET_PATH, file_name)
    dest_path = os.path.join(TRAIN_PATH, file_name)

    if os.path.exists(src_path):
        shutil.copy(src_path, dest_path)

for img in test_images:
    file_name = f"{img['file_name']}"
    src_path = os.path.join(DATASET_PATH, file_name)
    dest_path = os.path.join(TEST_PATH, file_name)

    if os.path.exists(src_path):
        shutil.copy(src_path, dest_path)

train_annotation_json = {
    "images": train_images,
    "annotations": train_annotations,
}
test_annotation_json = {
    "images": test_images,
    "annotations": test_annotations,
}

with open(TRAIN_ANNOTATION_FILE, "w") as f:
    json.dump(train_annotation_json, f, indent=4)

with open(TEST_ANNOTATION_FILE, "w") as f:
    json.dump(test_annotation_json, f, indent=4)

print(f"Copied {len(train_images)} files to {TRAIN_PATH}")
print(f"Copied {len(test_images)} files to {TEST_PATH}")
print(f"Train annotations saved to {TRAIN_ANNOTATION_FILE}")
print(f"Test annotations saved to {TEST_ANNOTATION_FILE}")
