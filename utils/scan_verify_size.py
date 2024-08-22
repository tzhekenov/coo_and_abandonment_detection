import os
import cv2

# Define paths to the dataset folders
dataset_paths = {
    'test': r'C:\Users\temir\Documents\KULcourses\Thesis_CV\code_yolo8_base\y8\yolo_tracking\COCO_AND_CUSTOM\yolo\obj_Test_data',
    'train': r'C:\Users\temir\Documents\KULcourses\Thesis_CV\code_yolo8_base\y8\yolo_tracking\COCO_AND_CUSTOM\yolo\obj_Train_data',
    'val': r'C:\Users\temir\Documents\KULcourses\Thesis_CV\code_yolo8_base\y8\yolo_tracking\COCO_AND_CUSTOM\yolo\obj_Validation_data'
}

# Path to the log file where issues will be written
log_file_path = r'C:\Users\temir\Documents\KULcourses\Thesis_CV\code_yolo8_base\y8\yolo_tracking\corrupt_labels_log.txt'

def check_label_file(label_file_path, image_width, image_height):
    """Check if the annotations in a label file are valid."""
    with open(label_file_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            return False  # Wrong format
        
        _, x_center, y_center, width, height = map(float, parts)
        
        # Check for non-normalized or out-of-bounds values
        if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1):
            return False
    return True

def scan_dataset(dataset_path, log_file):
    """Scan the dataset folder for corrupt labels."""
    for filename in os.listdir(dataset_path):
        if filename.lower().endswith('.jpg') or filename.lower().endswith('.png'):
            image_path = os.path.join(dataset_path, filename)
            label_path = os.path.splitext(image_path)[0] + '.txt'
            
            # Check if label file exists
            if not os.path.exists(label_path):
                log_file.write(f"Missing label for image: {image_path}\n")
                continue
            
            # Get image dimensions
            image = cv2.imread(image_path)
            if image is None:
                log_file.write(f"Corrupt image: {image_path}\n")
                continue
            h, w = image.shape[:2]
            
            # Validate label file
            if not check_label_file(label_path, w, h):
                log_file.write(f"Invalid label: {label_path}\n")

with open(log_file_path, 'w') as log_file:
    for key, dataset_path in dataset_paths.items():
        log_file.write(f"Scanning {key} dataset...\n")
        scan_dataset(dataset_path, log_file)

print("Scanning complete. Check the log file for details.")
