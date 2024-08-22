import os

# Define paths
images_path = r'C:\Users\temir\Documents\KULcourses\Thesis_CV\code_yolo8_base\y8\yolo_tracking\COCO_AND_CUSTOM\coco_subset\images'
annotations_path = r'C:\Users\temir\Documents\KULcourses\Thesis_CV\code_yolo8_base\y8\yolo_tracking\COCO_AND_CUSTOM\coco_subset\annotations'
log_file_path = r'C:\Users\temir\Documents\KULcourses\Thesis_CV\code_yolo8_base\y8\yolo_tracking\COCO_AND_CUSTOM\coco_yolo\comparison_log.txt'

# Get the list of filenames without extensions from both directories
image_filenames = {os.path.splitext(filename)[0] for filename in os.listdir(images_path)}
annotation_filenames = {os.path.splitext(filename)[0] for filename in os.listdir(annotations_path)}

# Find mismatches
annotations_without_images = annotation_filenames - image_filenames
images_without_annotations = image_filenames - annotation_filenames

# Write results to a log file
with open(log_file_path, 'w') as log_file:
    log_file.write("Annotations without corresponding images:\n")
    for item in sorted(annotations_without_images):
        log_file.write(f"{item}.json\n")
    
    log_file.write("\nImages without corresponding annotations:\n")
    for item in sorted(images_without_annotations):
        log_file.write(f"{item}.jpg\n")

print(f"Comparison completed. Log file created at: {log_file_path}")
