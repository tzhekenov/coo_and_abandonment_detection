# Thesis Project Repository
**Title: A Machine Vision Approach to Identify and Monitor Passengers and Luggage**



This repository contains the code and tools developed as part of master's thesis. The thesis focuses on multi-object detection, tracking, and re-identification using state-of-the-art methodologies, built upon the BoxMOT framework. Below is an overview of the code and scripts included in this repository.

<div align="center">
   <p>
    <img src="https://github.com/tzhekenov/coo_and_abandonment_detection/blob/dev/assets/output_comp.gif?raw=true" width="400px">
  </p>
  <br>
  <div>
Links for reference repository. GDrive large files link

<a href="https://github.com/mikel-brostrom/yolov8_tracking/actions/workflows/ci.yml"><img src="https://github.com/mikel-brostrom/yolov8_tracking/actions/workflows/ci.yml/badge.svg" alt="CI CPU testing"></a>
  <a href="https://drive.google.com/drive/folders/1UjbycA1CWhxjyZgymmXtaLXKN8wRph3f?usp=sharing"><img src="https://static.pepy.tech/badge/boxmot"></a>
  <br>


  </div>
</div>

## Abstract
This thesis focuses on enhancing railway safety and efficiency by improving the detection and management of security threats from unattended luggage. Given the high volume of daily passengers, accurately tracking and linking passengers with their belongings is crucial for preventing incidents and ensuring smooth operations. The research explores the challenges of developing and scaling a machine vision system for railway surveillance, highlighting various scenarios and system limitations.

The proposed solution leverages a comprehensive machine vision approach, utilizing advanced object detection (YOLO), re-identification models (OSNet), and tracking algorithms (OC-SORT) to accurately detect, track, and link passengers with their luggage. A spatio-temporal approach is developed for ownership assignment and abandonment detection, effectively flagging potential security threats, such as when passengers exit without their belongings. This research not only addresses current challenges but also proposes directions for future system improvements.


## Project Structure

| Folder / File           | Description                                                              |
|-------------------------|--------------------------------------------------------------------------|
| `annotations/`          | Contains zip files with datasets for different scenarios of abandonment. Annotations correspond to videos available in Google Drive link. However, some videos were not uploaded for privacy reasons|
| `assets/`               | Contains media assets such as GIFs used in the project.                  |
| `data/`                 | Contains ground truth data for evaluation on Benchmark case.                               |
| `eval/`                 | Contains scripts  related to model evaluation (MOTA, IDF1).                            |
| `utils/`                | Utility scripts and notebooks for data processing and model training.    |
| `weights_finetuned/`    | Directory containing finetuned model weights.                            |
| `README.md`             | This file, documenting the project and its structure.                    |

## Annotations Folder Content

| Image Preview | Filename                                              | Description                                                |
|---------------|-------------------------------------------------------|------------------------------------------------------------|
| <img src="https://github.com/tzhekenov/coo_and_abandonment_detection/blob/dev/annotations/1_Abandonment.png?raw=true" alt="1_Abandonment.png" width="100"/> | `1_Abandonment.zip`                                    | Dataset for general abandonment detection scenarios.       |
| <img src="https://github.com/tzhekenov/coo_and_abandonment_detection/blob/dev/annotations/2_Abandonment_zoomed_view.png?raw=true" alt="2_Abandonment_zoomed_view.png" width="100"/> | `2_Abandonment_zoomed_view.zip`                        | Dataset focusing on zoomed views of abandonment scenarios. |
| <img src="https://github.com/tzhekenov/coo_and_abandonment_detection/blob/dev/annotations/3_Benchmark_case_abandonment.png?raw=true" alt="3_Benchmark_case_abandonment.png" width="100"/> | `3_Benchmark_case_abandonment.zip`                     | Benchmark datasets for testing and comparison.             |
| <img src="https://github.com/tzhekenov/coo_and_abandonment_detection/blob/dev/annotations/4_Temporary_abandonment_and_abandonment_zoomed.png?raw=true" alt="4_Temporary_abandonment_and_abandonment_zoomed.png" width="100"/> | `4_Temporary_abandonment_and_abandonment_zoomed.zip`   | Temporary abandonment and zoomed views dataset.            |


## Table of Contents

- [Introduction](#introduction)
- [Setup and Installation](#setup-and-installation)
- [Scripts Overview](#scripts-overview)
  - [Main Pipeline for Detection and Tracking with Re-Identification](#a1-main-pipeline-for-detection-and-tracking-with-re-identification)
  - [Model Evaluation Pipeline](#a2-main-pipeline-for-evaluating-models)
  - [Tracking Results Conversion](#a3-conversion-of-tracking-results-to-mot-format)
  - [Image Scaling and Compression](#a4-image-scaling-and-compression-for-training-pipeline)
  - [COCO Dataset Subset Extraction](#a5-getting-subset-of-the-coco-dataset)
  - [Label Remapping Tool](#a6-label-remapping-tool)
  - [Label Verification Script](#a7-label-verification-script)
  - [Frame Visualization Tool](#a8-frame-visualization-tool)
  - [Zone Drawing on Video](#a9-zone-drawing-on-video)
  - [Abandonment and Ownership Detection](#a10-abandonment-and-ownership-detection)
  - [Frame Extractor Tool](#a11-frame-extractor-tool)


## Introduction

This repository contains a series of scripts and tools developed during my master's thesis, focusing on the detection, tracking, and re-identification of objects in video sequences. The project leverages the BoxMOT framework, a robust collection of state-of-the-art trackers integrated with various detection models, providing a flexible and powerful platform for multi-object tracking tasks.
Note that the spatio-temporal labeling tool should be incorporated as a function which will work with YOLO outputs. Currently, it can work with either YOLO outputs or annotations in manual mode (directories must be assigned.)

## Setup and Installation

To get started with the code in this repository, follow these steps:
    
1. **Get data**:
    Get data file from the download link (downloads icon before abstract).
    
    <span style="font-size: 24px; font-weight: bold;">Video data available from -> </span> 
      <a href="https://drive.google.com/drive/folders/1UjbycA1CWhxjyZgymmXtaLXKN8wRph3f?usp=sharing"><img src="https://upload.wikimedia.org/wikipedia/commons/1/12/Google_Drive_icon_%282020%29.svg"  width="30" height="30"></a> 
2. **Clone the Repository**:
    ```bash
    git clone https://github.com/mikel-brostrom/boxmot.git
    cd boxmot
    ```

3. **Install Dependencies**:
    ```bash
    pip install poetry
    poetry install --with yolo
    poetry shell
    ```

4. **Run Scripts**:
    Each script in this repository can be run individually, depending on your specific task. 


## Scripts Overview

### A.1 Main Pipeline for Detection and Tracking with Re-Identification

This script initiates the detection, tracking, and re-identification of objects (persons, suitcases, backpacks, handbags) within a video sequence using specified models.

**Command**:
```bash
python tracking/track.py \
    --source /workspaces/coo_and_abandonment_detection/9505_30fr_cut.mp4 \
    --yolo-model yolov8l.pt \
    --tracking-method ocsort \
    --reid-model osnet_x1_0_msmt17.pt \
    --classes 0 26 30 32 \ #classes correspondnig to person, backpack, handbag and suitcase 
    --save-txt \
    --save \
    --project /usr/src/boxmot/data/track \
    --name tracking_results \
    --verbose
 ```

### A.2 Main Pipeline for Evaluating models 
Purpose: Custom made scrip provides MOTA and IDF1 metrics for tracking problems
**Command**:
```bash
python tracking/eval.py 
 ```


Purpose: This is an alternative script which can help evalute different MOT benchmarks.
Methods for validating the model performance here are taken from the BoxMot repository (Broström, 2023):

**Command**:
```bash
python tracking/val.py \     
--benchmark MOT17     
--yolo-model yolov8l.pt     
--reid-model osnet_x1_0_msmt17.pt     
--tracking-method ocsort     
--verbose     
--source /usr/src/boxmot/data/9505_30fr_cut_MOT  
--classes 0 26 30 32
 ```

### A.3 Conversion of Tracking Results to MOT Format
Purpose: Converts tracking results with assigned IDs into the MOT format, making them compatible for further analysis and benchmarking.

**Command**:
```python
import os

# Path to the directory containing the detection results
tracking_dir = r"tracking_results/labels"
output_file = r'det/det.txt'

with open(output_file, 'w') as out_file:
    for track_file in os.listdir(tracking_dir):
        if track_file.endswith('.txt'):
            frame_id = int(track_file.split('_')[-1].split('.')[0])
            with open(os.path.join(tracking_dir, track_file), 'r') as in_file:
                for line in in_file:
                    parts = line.strip().split()
                    x_center, y_center, width, height, confidence = map(float, parts[1:6])
                    
                    # Convert center x, y to top-left x, y for MOT format
                    x_min = x_center - width / 2
                    y_min = y_center - height / 2

                    out_file.write(f"{frame_id},-1,{x_min},{y_min},{width},{height},{confidence}\n")
 ```
________________________________________
### A.4 Image Scaling and Compression for training pipeline
Purpose: Resizes images and compresses them to a specified quality, facilitating more efficient storage and processing during model training and testing.


**Command**:
```python
import os
import cv2 as cv
from ultralytics.utils.ops import scale_boxes
from ultralytics.data.utils import compress_one_image
from pathlib import Path

input_dir = r"obj_TEST_data-large"
output_dir = r"test_data_output"
scale_factor = 0.5
compression_quality = 100

os.makedirs(output_dir, exist_ok=True)

for i in range(5358):
    img_file = f"frame_{i:06d}.PNG"
    txt_file = f"frame_{i:06d}.txt"
    img_path = os.path.join(input_dir, img_file)
    txt_path = os.path.join(input_dir, txt_file)

    if not os.path.exists(img_path) or not os.path.exists(txt_path):
        continue

    image = cv.imread(img_path)
    h, w, _ = image.shape
    resized_image = cv.resize(image, (int(w * scale_factor), int(h * scale_factor)))
    output_img_path = os.path.join(output_dir, img_file)
    cv.imwrite(output_img_path, resized_image)
    compress_one_image(Path(output_img_path), quality=compression_quality)

    
 ```
________________________________________
### A.5 Getting subset of the COCO dataset

Purpose: Extracts a subset of images and corresponding annotations from the COCO dataset based on specified object classes (e.g., handbags, backpacks, suitcases).

**Command**:
```python

import os
import json
import urllib.request
from pycocotools.coco import COCO

class_names = ['handbag', 'backpack', 'suitcase']
class_ids = [27, 31, 33]
save_dir = 'coco_subset'
images_dir = os.path.join(save_dir, 'images_folder')
annotations_dir = os.path.join(save_dir, 'annotations')
os.makedirs(images_dir, exist_ok=True)
os.makedirs(annotations_dir, exist_ok=True)

annotation_file = r'instances_train2017.json'
coco = COCO(annotation_file)
img_ids = []

for class_id in class_ids:
    img_ids += coco.getImgIds(catIds=[class_id])

img_ids = list(set(img_ids))

def download_image(img_url, img_path):
    urllib.request.urlretrieve(img_url, img_path)

for img_id in img_ids:
    img_info = coco.loadImgs(img_id)[0]
    img_url = img_info['coco_url']
    img_path = os.path.join(images_dir, img_info['file_name'])
    if not os.path.exists(img_path):
        download_image(img_url, img_path)
    ann_ids = coco.getAnnIds(imgIds=[img_id], catIds=class_ids, iscrowd=None)
    anns = coco.loadAnns(ann_ids)
    annotation_path = os.path.join(annotations_dir, f"{img_info['file_name'].split('.')[0]}.json")
    with open(annotation_path, 'w') as f:
        json.dump(anns, f)
 ```

### A.6 Label Remapping Tool

Purpose: Re-maps class IDs in annotation files to new IDs, ensuring consistency across different datasets, when merging. 


**Command**:
```python
import os

# Paths to your first dataset
first_dataset_labels_path = r'coco_yolo/labels'
remap_first_dataset = {0: 3, 1: 1, 2: 2}

for filename in os.listdir(first_dataset_labels_path):
    if filename.endswith('.txt'):
        label_file_path = os.path.join(first_dataset_labels_path, filename)
        new_labels = []
        with open(label_file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                class_id = int(parts[0])
                if class_id in remap_first_dataset:
                    parts[0] = str(remap_first_dataset[class_id])
                    new_labels.append(' '.join(parts))
        with open(label_file_path, 'w') as f:
            f.write('\n'.join(new_labels))
 ```

________________________________________
### A.7 Label Verification Script

Purpose: Scans the dataset to verify the integrity of annotation files, ensuring all bounding boxes are within the image boundaries.


**Command**:
```python
import os
import cv2

dataset_paths = {
    'test': r'obj_Test_data',
    'train': r'obj_Train_data',
    'val': r'obj_Validation_data'
}

log_file_path = r'corrupt_labels_log.txt'

def check_label_file(label_file_path, image_width, image_height):
    with open(label_file_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            return False
        _, x_center, y_center, width, height = map(float, parts)
        if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1):
            return False
    return True

def scan_dataset(dataset_path, log_file):
    for filename in os.listdir(dataset_path):
        if filename.lower().endswith('.jpg') or filename.lower().endswith('.png'):
            image_path = os.path.join(dataset_path, filename)
            label_path = os.path.splitext(image_path)[0] + '.txt'
            if not os.path.exists(label_path):
                log_file.write(f"Missing label for image: {image_path}\n")
                continue
            image = cv2.imread(image_path)
            if image is None:
                log_file.write(f"Corrupt image: {image_path}\n")
                continue
            h, w = image.shape[:2]
            if not check_label_file(label_path, w, h):
                log_file.write(f"Invalid label: {label_path}\n")

with open(log_file_path, 'w') as log_file:
    for key, dataset_path in dataset_paths.items():
        log_file.write(f"Scanning {key} dataset...\n")
        scan_dataset(dataset_path, log_file)
 ```
________________________________________
### A.8 Frame Visualization Tool

Purpose: Visualizes bounding boxes from YOLO annotations on the corresponding image frames for verification purposes.



**Command**:
```python
import cv2
import matplotlib.pyplot as plt

labels_file = r"test_data_output/frame_005169.txt"
image_file = r"test_data_output/frame_005169.PNG"

image = cv2.imread(image_file)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_height, image_width, _ = image.shape

annotations = []
with open(labels_file, 'r') as f:
    for line in f.readlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        class_id, x_center, y_center, width, height = map(float, parts)
        x_center *= image_width
        y_center *= image_height
        width *= image_width
        height *= image_height
        x_min = x_center - width / 2
        y_min = y_center - height / 2
        annotations.append((x_min, y_min, width, height))

plt.figure(figsize=(10, 10))
plt.imshow(image)

for bbox in annotations:
    x_min, y_min, width, height = bbox
    rect = plt.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='red', facecolor='none')
    plt.gca().add_patch(rect)

plt.axis('off')
plt.show()
 ```
________________________________________
### A.9 Zone Drawing on Video

Purpose: Interactively draws and saves polygonal zones on video frames, which can be used for subsequent analysis, such as time spent within certain areas.


**Command**:
```python

import argparse
import json
import os
import cv2
import numpy as np

import supervision as sv

KEY_ENTER = 13
KEY_ESCAPE = 27
KEY_SAVE = ord("s")
KEY_QUIT = ord("q")
THICKNESS = 2
COLORS = sv.ColorPalette.DEFAULT
WINDOW_NAME = "Draw Zones"
POLYGONS = [[]]

current_mouse_position = None

def resolve_source(source_path):
    if not os.path.exists(source_path):
        return None
    image = cv2.imread(source_path)
    if image is not None:
        return image
    frame_generator = sv.get_video_frames_generator(source_path=source_path)
    frame = next(frame_generator)
    return frame

def mouse_event(event, x, y, flags, param):
    global current_mouse_position
    if event == cv2.EVENT_MOUSEMOVE:
        current_mouse_position = (x, y)
    elif event == cv2.EVENT_LBUTTONDOWN:
        POLYGONS[-1].append((x, y))

def redraw(image, original_image):
    global POLYGONS, current_mouse_position
    image[:] = original_image.copy()
    for idx, polygon in enumerate(POLYGONS):
        color = COLORS.by_idx(idx).as_bgr() if idx < len(POLYGONS) - 1 else sv.Color.WHITE.as_bgr()
        if len(polygon) > 1:
            for i in range(1, len(polygon)):
                cv2.line(image, polygon[i - 1], polygon[i], color, THICKNESS)
            if idx < len(POLYGONS) - 1:
                cv2.line(image, polygon[-1], polygon[0], color, THICKNESS)
        if idx == len(POLYGONS) - 1 and current_mouse_position is not None and polygon:
            cv2.line(image, polygon[-1], current_mouse_position, color, THICKNESS)
    cv2.imshow(WINDOW_NAME, image)

def save_polygons_to_json(polygons, target_path):
    data_to_save = polygons if polygons[-1] else polygons[:-1]
    with open(target_path, "w") as f:
        json.dump(data_to_save, f)

def main(source_path, zone_configuration_path):
    global current_mouse_position
    original_image = resolve_source(source_path=source_path)
    if original_image is None:
        print("Failed to load source image.")
        return

    image = original_image.copy()
    cv2.imshow(WINDOW_NAME, image)
    cv2.setMouseCallback(WINDOW_NAME, mouse_event, image)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == KEY_ENTER:
            if len(POLYGONS[-1]) > 2:
                POLYGONS.append([])
        elif key == KEY_ESCAPE:
            POLYGONS[-1] = []
            current_mouse_position = None
        elif key == KEY_SAVE:
            save_polygons_to_json(POLYGONS, zone_configuration_path)
            print(f"Polygons saved to {zone_configuration_path}")
            break
        redraw(image, original_image)
        if key == KEY_QUIT:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interactively draw polygons on images or video frames and save "
        "the annotations."
    )
    parser.add_argument("--source_path", type=str, required=True, help="Path to the source image or video file for drawing polygons.")
    parser.add_argument("--zone_configuration_path", type=str, required=True, help="Path where the polygon annotations will be saved as a JSON file.")
    arguments = parser.parse_args()
    main(source_path=arguments.source_path, zone_configuration_path=arguments.zone_configuration_path)
 ```
________________________________________
### A.10 Abandonment and Ownership Detection

Purpose: Detects the abandonment of objects (e.g., bags) by their owners within predefined zones in video sequences.


**Command**:
```python

import cv2
import numpy as np
import os
import json
from collections import defaultdict
from datetime import datetime, timedelta

zone_config_path = 'zone_config_9505_fullHD.json'
video_path = '9505_30fr_cut.mp4'
annotation_folder = 'obj_train_data'

with open(zone_config_path) as f:
    zone_config = json.load(f)

zone = np.array(zone_config, np.int32).reshape((-1, 1, 2))

object_counter = {0: 7, 1: 12, 2: 130}
entry_time = defaultdict(list)
exit_time = defaultdict(list)
time_spent = defaultdict(float)
frame_data = []
ownership = defaultdict(lambda: {'start_time': None, 'owned': False})

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, fps, (1920, 1080))

def is_inside_zone(center, zone):
    return cv2.pointPolygonTest(zone, center, False) >= 0

def draw_zone(frame, zone):
    cv2.polylines(frame, [zone], isClosed=True, color=(0, 255, 255), thickness=2)

def process_annotation_file(annotation_file, frame_num):
    bbox_data = []
    with open(annotation_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            obj_class = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:])
            x_center *= 1920
            y_center *= 1080
            width *= 1920
            height *= 1080
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)
            bbox_data.append((obj_class, x1, y1, x2, y2))
    return bbox_data

def calculate_distance(center1, center2):
    return np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)

def update_ownership(centers, fps):
    for obj_id in [12, 130]:
        if 7 in centers and obj_id in centers:
            distance = calculate_distance(centers[7], centers[obj_id])
            if distance < 200:
                if ownership[obj_id]['start_time'] is None:
                    ownership[obj_id]['start_time'] = datetime.now()
                elif (datetime.now() - ownership[obj_id]['start_time']).total_seconds() > 0.3:
                    ownership[obj_id]['owned'] = True
            else:
                ownership[obj_id]['start_time'] = None
        else:
            ownership[obj_id]['start_time'] = None

for frame_num in range(frame_count):
    ret, frame = cap.read()
    if not ret:
        break
    
    annotation_file = os.path.join(annotation_folder, f'frame_{frame_num:06d}.txt')
    if not os.path.exists(annotation_file):
        continue
    
    bbox_data = process_annotation_file(annotation_file, frame_num)
    draw_zone(frame, zone)
    
    frame_info = {
        'frame': frame_num,
        'inside_zone': {
            'persons': [],
            'luggage': []
        },
        'outside_zone': {
            'persons': [],
            'luggage': []
        },
        'distances': {},
        'time_in_zone': {},
        'ownership': {}
    }
    
    centers = {}
    
    for obj_class, x1, y1, x2, y2 in bbox_data:
        color = (0, 255, 0) if obj_class == 0 else (255, 0, 0) if obj_class == 1 else (0, 0, 255)
        label = 'Person' if obj_class == 0 else 'Handbag' if obj_class == 1 else 'Backpack'
        obj_id = object_counter[obj_class]
        
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        centers[obj_id] = center
        
        if is_inside_zone(center, zone):
            if obj_class not in entry_time or len(entry_time[obj_class]) == len(exit_time[obj_class]):
                entry_time[obj_class].append(datetime.now())
            frame_info['inside_zone']['persons' if obj_class == 0 else 'luggage'].append(obj_id)
        else:
            if obj_class in entry_time and len(entry_time[obj_class]) > len(exit_time[obj_class]):
                exit_time[obj_class].append(datetime.now())
                time_spent[obj_class] += (exit_time[obj_class][-1] - entry_time[obj_class][-1]).total_seconds()
            frame_info['outside_zone']['persons' if obj_class == 0 else 'luggage'].append(obj_id)
        
        time_in_zone = time_spent[obj_class] if obj_class in time_spent else 0
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f'{label} ID: {obj_id}', (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(frame, f'Time in zone: {time_in_zone:.2f}s', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        if ownership[obj_id]['owned']:
            cv2.putText(frame, f'{label} ID: {obj_id} owner: 7', (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    if 7 in centers and 12 in centers:
        distance = calculate_distance(centers[7], centers[12])
        distance_key = f'Distance between Person 7 and Handbag 12'
        frame_info['distances'][distance_key] = distance
        cv2.putText(frame, f'{distance_key}: {distance:.2f}px', (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    if 7 in centers and 130 in centers:
        distance = calculate_distance(centers[7], centers[130])
        distance_key = f'Distance between Person 7 and Backpack 130'
        frame_info['distances'][distance_key] = distance
        cv2.putText(frame, f'{distance_key}: {distance:.2f}px', (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    update_ownership(centers, fps)
    
    for obj_class, ids in frame_info['inside_zone'].items():
        for id in ids:
            frame_info['time_in_zone'][f'{obj_class[:-1]} ID#{id}'] = time_spent[obj_class]
    
    for obj_id, data in ownership.items():
        if data['owned']:
            frame_info['ownership'][f'ID {obj_id}'] = 7
    
    frame_data.append(frame_info)
    out.write(frame)

cap.release()
out.release()

# Save JSON data
with open('framewise_detections.json', 'w') as f:
    json.dump(frame_data, f, indent=4)

# Output time spent and counts
for obj_class, count in object_counter.items():
    label = 'Person' if obj_class == 0 else 'Handbag' if obj_class == 1 else 'Backpack'
    print(f'{label}: {count} times entered the zone, total time spent: {time_spent[obj_class]} seconds')
 ```
________________________________________
### A.11 Frame Extractor Tool

Purpose: Extracts specific frames from a video based on a given range, saving them as a separate video file.


**Command**:
```python
import cv2

def extract_frames(input_video_path, output_video_path, start_frame, end_frame):
    cap = cv2.VideoCapture(input_video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    current_frame = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if start_frame <= current_frame <= end_frame:
            out.write(frame)
        current_frame += 1
        if current_frame > end_frame:
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Usage
input_video_path = r"9505_30fr_cut.mp4"
output_video_path = 'output.mp4'
start_frame = 130
end_frame = 400

extract_frames(input_video_path, output_video_path, start_frame, end_frame)

 ```

 ### [BoxMOT Repository](#boxmot-repository)

- [Contributors](#contributors)
- [License](#license)