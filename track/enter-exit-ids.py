import cv2
import numpy as np
import json
from collections import defaultdict

# Paths to your files
gt_path = "C:/Users/temir/Documents/KULcourses/Thesis_CV/!ThesisFiles/annotated_orig/cvat_annotations/annotations/9505_benchmark_MOT11/gt/gt.txt"
video_path = "C:/Users/temir/Documents/KULcourses/Thesis_CV/code_yolo8_base/y8/yolo_tracking/assets/clips/9505_30fr_cut.mp4"
zone_config_path = "C:/Users/temir/Documents/KULcourses/Thesis_CV/yolo9_heatmap/supervision/examples/time_in_zone/zone_config_9505_fullHD.json"
labels_path = "C:/Users/temir/Documents/KULcourses/Thesis_CV/!ThesisFiles/annotated_orig/cvat_annotations/annotations/9505_benchmark_MOT11/gt/labels.txt"

# Load zone configuration from JSON
with open(zone_config_path) as f:
    zone_config = json.load(f)

# Convert zone coordinates to numpy array
zone = np.array(zone_config, np.int32).reshape((-1, 1, 2))

# Load class labels from labels.txt
with open(labels_path, 'r') as f:
    class_labels = [line.strip() for line in f.readlines()]

class TrackedObject:
    def __init__(self, obj_id, obj_class, center):
        self.id = obj_id
        self.obj_class = obj_class
        self.center = center
        self.inside_zone = False
        self.entered = False
        self.exited = False

    def is_inside_zone(self, zone):
        return cv2.pointPolygonTest(zone, self.center, False) >= 0

def parse_gt_file(gt_path):
    gt_data = defaultdict(list)
    with open(gt_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split(',')
            frame_num = int(parts[0])
            obj_id = int(parts[1])
            x = float(parts[2])
            y = float(parts[3])
            width = float(parts[4])
            height = float(parts[5])
            obj_class = int(parts[7])  # Assuming 'class_id' is at index 7
            center = (int(x + width / 2), int(y + height / 2))
            gt_data[frame_num].append((obj_id, obj_class, center))
    return gt_data

def format_output(class_labels, obj_id, obj_class):
    class_name = class_labels[obj_class - 1]  # Adjusting for 0-indexing in Python
    return f"{class_name} - id: {obj_id}"

def main(gt_path, video_path):
    # Load ground truth data
    gt_data = parse_gt_file(gt_path)
    
    # Initialize containers
    tracked_objects = {}
    ids_entered = set()
    ids_exited = set()
    abandoned = set()

    # Video properties
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for frame_num in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        if frame_num in gt_data:
            for obj_id, obj_class, center in gt_data[frame_num]:
                if obj_id not in tracked_objects:
                    tracked_objects[obj_id] = TrackedObject(obj_id, obj_class, center)
                tracked_obj = tracked_objects[obj_id]
                tracked_obj.center = center  # Update center

                inside_zone = tracked_obj.is_inside_zone(zone)

                if inside_zone and not tracked_obj.inside_zone:
                    tracked_obj.inside_zone = True
                    if not tracked_obj.entered:
                        tracked_obj.entered = True
                        ids_entered.add((tracked_obj.id, tracked_obj.obj_class))
                    elif tracked_obj.entered and tracked_obj.exited:
                        ids_exited.add((tracked_obj.id, tracked_obj.obj_class))

                if not inside_zone and tracked_obj.inside_zone:
                    tracked_obj.inside_zone = False
                    if tracked_obj.entered and not tracked_obj.exited:
                        tracked_obj.exited = True
                        if (tracked_obj.id, tracked_obj.obj_class) not in ids_exited:
                            abandoned.add((tracked_obj.id, tracked_obj.obj_class))

    cap.release()

    # Format output with class names
    output = {
        "Ids_entered": [format_output(class_labels, obj_id, obj_class) for obj_id, obj_class in ids_entered],
        "Ids_exited": [format_output(class_labels, obj_id, obj_class) for obj_id, obj_class in ids_exited],
        "abandoned": [format_output(class_labels, obj_id, obj_class) for obj_id, obj_class in abandoned - ids_exited]
    }

    with open('zone_activity.json', 'w') as f:
        json.dump(output, f, indent=4)

    print("Processing complete. JSON data has been saved.")

if __name__ == "__main__":
    main(gt_path, video_path)
