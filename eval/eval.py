#working on boxmot
# 
# 
import os
import pandas as pd
import motmetrics as mm

# Function to convert normalized bounding boxes to absolute coordinates
def convert_bbox_format(center_x, center_y, width, height, img_width, img_height):
    bb_left = (center_x * img_width) - (width * img_width) / 2
    bb_top = (center_y * img_height) - (height * img_height) / 2
    bb_width = width * img_width
    bb_height = height * img_height
    return bb_left, bb_top, bb_width, bb_height


# Paths BASE MODEL
# labels_dir = r"C:\Users\temir\Documents\KULcourses\Thesis_CV\code_yolo8_base\y8\yolo_tracking\runs\runs_final_conf025_iou05_osnet1_ocsort_yolo8Lbase_042100\track_output\labels"

# Paths
labels_dir = r"C:\Users\temir\Documents\KULcourses\Thesis_CV\code_yolo8_base\y8\yolo_tracking\runs\botsort_final_osnet_x1_0_25_05_FINETUNED_20242208_100231\track_output\labels"
gt_file = r"C:\Users\temir\Documents\KULcourses\Thesis_CV\code_yolo8_base\y8\yolo_tracking\COCO_AND_CUSTOM\track_9505\train\9505_all_items\gt\gt.txt" #2 person gt

# Image dimensions (assuming all frames are of the same size)
img_width = 1920  # Update with the correct width
img_height = 1080  # Update with the correct height

# Initialize a DataFrame to store tracking data
tracking_data = []

# Loop through each label file
for i in range(1, 629):
    frame_id = i
    file_path = os.path.join(labels_dir, f"9505_30fr_cut_{i}.txt")
    
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            for line in f:
                class_id, center_x, center_y, width, height, obj_id = line.strip().split()
                class_id = int(class_id)
                
                # Convert class ID to match the GT format
                if class_id == 0:
                    class_id = 1  # person
                elif class_id == 26:
                    class_id = 2  # backpack
                elif class_id == 30:
                    class_id = 3  # handbag
                elif class_id == 32:
                    class_id = 4  # suitcase
                
                # Convert normalized bbox to absolute bbox
                bb_left, bb_top, bb_width, bb_height = convert_bbox_format(float(center_x), float(center_y), float(width), float(height), img_width, img_height)
                
                # Append to tracking data
                tracking_data.append([frame_id, obj_id, bb_left, bb_top, bb_width, bb_height, 1, class_id, 1.0])

# Convert to DataFrame
columns = ["FrameId", "Id", "X", "Y", "Width", "Height", "Confidence", "ClassId", "Visibility"]
tracking_df = pd.DataFrame(tracking_data, columns=columns)

# Save the tracking data to a CSV in MOT format
tracking_df.to_csv("tracking_output.txt", index=False, header=False)

# Load the ground truth data
gt_df = pd.read_csv(gt_file, header=None)
gt_df.columns = ["FrameId", "Id", "X", "Y", "Width", "Height", "Confidence", "ClassId", "Visibility"]

# Create MOTAccumulator object for MOTA/IDF1 calculation
acc = mm.MOTAccumulator(auto_id=True)

# Iterate over each frame
for frame_id in gt_df["FrameId"].unique():
    gt_objects = gt_df[gt_df["FrameId"] == frame_id]
    tr_objects = tracking_df[tracking_df["FrameId"] == frame_id]
    
    gt_bboxes = gt_objects[["X", "Y", "Width", "Height"]].values
    tr_bboxes = tr_objects[["X", "Y", "Width", "Height"]].values
    
    gt_ids = gt_objects["Id"].values
    tr_ids = tr_objects["Id"].values
    
    distances = mm.distances.iou_matrix(gt_bboxes, tr_bboxes, max_iou=0.5)
    
    acc.update(gt_ids, tr_ids, distances)

# Calculate MOTA and IDF1
mh = mm.metrics.create()
summary = mh.compute(acc, metrics=['mota', 'idf1'], name='acc')

print(summary)
