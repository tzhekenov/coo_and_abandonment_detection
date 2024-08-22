import fiftyone as fo
import fiftyone.zoo as foz
import os

dataset = foz.load_zoo_dataset(
    "coco-2017",
    splits=["train", "validation", "test"],
    label_types=["detections"],
    classes=["person", "suitcase", "handbag", "backpack"],
    # max_samples=50,
)