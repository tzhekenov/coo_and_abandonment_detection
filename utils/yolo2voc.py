# this files converts the YOLO format annotations to VOC format annotations

import os
import cv2
import glob

# Function to convert YOLO coordinates to VOC format
def convert_yolo_to_voc(x_c_n, y_c_n, width_n, height_n, img_width, img_height):
    # Convert normalized center coordinates and dimensions to actual pixel values
    x_c = float(x_c_n) * img_width
    y_c = float(y_c_n) * img_height
    width = float(width_n) * img_width
    height = float(height_n) * img_height
    # Calculate top-left and bottom-right corner of the bounding box
    half_width = width / 2
    half_height = height / 2
    left = int(x_c - half_width)
    top = int(y_c - half_height)
    right = int(x_c + half_width)
    bottom = int(y_c + half_height)
    return left, top, right, bottom

# Directories
model_output_dir = '/vsc-hard-mounts/leuven-data/359/vsc35938/COCO/test/coco-base-out/coco-base-out/labels'
ground_truth_dir = '/vsc-hard-mounts/leuven-data/359/vsc35938/COCO/test/labels-coco'
images_dir = '/vsc-hard-mounts/leuven-data/359/vsc35938/COCO/test/images'

# Output directories
detection_results_dir = 'input_/detection-results'
ground_truth_output_dir = 'input_/ground-truth'

# Create output directories if they don't exist
os.makedirs(detection_results_dir, exist_ok=True)
os.makedirs(ground_truth_output_dir, exist_ok=True)

# Load class list (COCO class IDs)
class_list = {
    0: "person",
    24: "backpack",
    26: "handbag",
    28: "suitcase"
}

# Process a single file
def process_file(txt_file, img_file, output_dir, is_model_output):
    with open(txt_file, 'r') as f:
        lines = f.readlines()

    # Load the image to get its dimensions
    img = cv2.imread(img_file)
    if img is None:
        print(f"Error: Unable to load image {img_file}")
        return
    img_height, img_width = img.shape[:2]

    # Create the corresponding VOC format file
    base_name = os.path.basename(txt_file).replace('.txt', '.txt')
    output_file = os.path.join(output_dir, base_name)
    
    with open(output_file, 'w') as out_f:
        for line in lines:
            parts = line.strip().split()
            obj_id = int(parts[0])  # Object class ID
            x_c_n, y_c_n, width_n, height_n = map(float, parts[1:5])
            
            # Convert YOLO format to VOC format
            left, top, right, bottom = convert_yolo_to_voc(x_c_n, y_c_n, width_n, height_n, img_width, img_height)
            
            # For detection results, include the confidence score
            if is_model_output:
                confidence = float(parts[5])
                out_f.write(f"{class_list[obj_id]} {confidence:.6f} {left} {top} {right} {bottom}\n")
            else:
                out_f.write(f"{class_list[obj_id]} {left} {top} {right} {bottom}\n")

# Process all ground truth files
gt_files = glob.glob(os.path.join(ground_truth_dir, '*.txt'))
for gt_file in gt_files:
    # Find the corresponding image
    image_name = os.path.basename(gt_file).replace('.txt', '.jpg')
    image_path = os.path.join(images_dir, image_name)
    if not os.path.exists(image_path):
        print(f"Error: Image not found for {gt_file}")
        continue

    # Convert ground truth annotations
    process_file(gt_file, image_path, ground_truth_output_dir, is_model_output=False)

# Process all model output files
model_output_files = glob.glob(os.path.join(model_output_dir, '*.txt'))
for model_file in model_output_files:
    # Find the corresponding image
    image_name = os.path.basename(model_file).replace('.txt', '.jpg')
    image_path = os.path.join(images_dir, image_name)
    if not os.path.exists(image_path):
        print(f"Error: Image not found for {model_file}")
        continue

    # Convert detection results (model output)
    process_file(model_file, image_path, detection_results_dir, is_model_output=True)

print("Conversion completed!")
