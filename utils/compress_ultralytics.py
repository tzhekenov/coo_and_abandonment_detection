import os
import cv2 as cv
import numpy as np
from ultralytics.utils.ops import scale_boxes, xywh2xyxy, xyxy2xywh

# Define input and output directories
input_dir = r"C:\Users\temir\Documents\KULcourses\Thesis_CV\code_yolo8_base\y8\yolo_tracking\COCO_AND_CUSTOM\yolo\obj_Train_data-large"
output_dir = r"C:\Users\temir\Documents\KULcourses\Thesis_CV\code_yolo8_base\y8\yolo_tracking\COCO_AND_CUSTOM\yolo\tran_data_output"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Scale factor
scale_factor = 0.5

# Iterate over all PNG files in the input directory
for i in range(2343):
    img_file = f"frame_{i:06d}.PNG"
    txt_file = f"frame_{i:06d}.txt"
    
    img_path = os.path.join(input_dir, img_file)
    txt_path = os.path.join(input_dir, txt_file)

    if not os.path.exists(img_path) or not os.path.exists(txt_path):
        continue

    # Load image
    image = cv.imread(img_path)
    h, w, _ = image.shape

    # Resize image
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    resized_image = cv.resize(image, (new_w, new_h))

    # Save resized image
    output_img_path = os.path.join(output_dir, img_file)
    cv.imwrite(output_img_path, resized_image)

    # Load annotations
    with open(txt_path, 'r') as file:
        lines = file.readlines()

    # Check if the annotation file is empty
    if not lines:
        # If the annotation file is empty, just copy it as is
        output_txt_path = os.path.join(output_dir, txt_file)
        with open(output_txt_path, 'w') as file:
            file.writelines(lines)
        print(f"Processed {img_file} with empty annotations.")
        continue

    # Convert YOLO annotations to xyxy format
    original_boxes = []
    for line in lines:
        class_id, x_center, y_center, width, height = map(float, line.strip().split())
        # Convert from xywh (YOLO format) to xyxy
        x1 = (x_center - width / 2) * w
        y1 = (y_center - height / 2) * h
        x2 = (x_center + width / 2) * w
        y2 = (y_center + height / 2) * h
        original_boxes.append([x1, y1, x2, y2])

    original_boxes = np.array(original_boxes)

    # Scale the bounding boxes
    scaled_boxes = scale_boxes(
        img1_shape=(h, w),  # original image dimensions
        boxes=original_boxes,  # boxes from original image in xyxy format
        img0_shape=(new_h, new_w),  # resized image dimensions
    )

    # Convert scaled boxes back to YOLO format
    new_annotations = []
    for box, line in zip(scaled_boxes, lines):
        x1, y1, x2, y2 = box
        # Convert from xyxy to xywh
        x_center = (x1 + x2) / 2 / new_w
        y_center = (y1 + y2) / 2 / new_h
        width = (x2 - x1) / new_w
        height = (y2 - y1) / new_h
        class_id = line.split()[0]
        # Store the new annotation
        new_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    # Save updated annotations
    output_txt_path = os.path.join(output_dir, txt_file)
    with open(output_txt_path, 'w') as file:
        file.writelines(new_annotations)

    print(f"Processed {img_file} and {txt_file}")

print("All images and annotations have been resized and saved.")
