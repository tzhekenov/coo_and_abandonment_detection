import os
import shutil

# Define paths
first_dataset_images_path = r'C:\Users\temir\Documents\KULcourses\Thesis_CV\code_yolo8_base\y8\yolo_tracking\COCO_AND_CUSTOM\coco_yolo\images'
first_dataset_labels_path = r'C:\Users\temir\Documents\KULcourses\Thesis_CV\code_yolo8_base\y8\yolo_tracking\COCO_AND_CUSTOM\coco_yolo\labels'

output_test_path = r'C:\Users\temir\Documents\KULcourses\Thesis_CV\code_yolo8_base\y8\yolo_tracking\COCO_AND_CUSTOM\yolo\test'
output_train_path = r'C:\Users\temir\Documents\KULcourses\Thesis_CV\code_yolo8_base\y8\yolo_tracking\COCO_AND_CUSTOM\yolo\train'
output_val_path = r'C:\Users\temir\Documents\KULcourses\Thesis_CV\code_yolo8_base\y8\yolo_tracking\COCO_AND_CUSTOM\yolo\val'

# Create directories if they don't exist
os.makedirs(output_test_path, exist_ok=True)
os.makedirs(output_train_path, exist_ok=True)
os.makedirs(output_val_path, exist_ok=True)

# Collect all files from the first dataset
all_files = [f for f in os.listdir(first_dataset_images_path) if f.endswith('.jpg') or f.endswith('.png')]

# Sort files for consistent order
all_files.sort()

# Define the splits
split_test = all_files[:1613]
split_train = all_files[1613:1613+9253]
split_val = all_files[1613+9253:1613+9253+1613]

# Function to move files and keep logs
def move_files(file_list, src_images_path, src_labels_path, dst_path, log_file):
    with open(log_file, 'a') as log:
        for file_name in file_list:
            image_src = os.path.join(src_images_path, file_name)
            label_src = os.path.join(src_labels_path, os.path.splitext(file_name)[0] + '.txt')
            
            image_dst = os.path.join(dst_path, 'images', file_name)
            label_dst = os.path.join(dst_path, 'labels', os.path.splitext(file_name)[0] + '.txt')
            
            # Ensure destination directories exist
            os.makedirs(os.path.join(dst_path, 'images'), exist_ok=True)
            os.makedirs(os.path.join(dst_path, 'labels'), exist_ok=True)
            
            # Move the image file
            if os.path.exists(image_src):
                shutil.move(image_src, image_dst)
                log.write(f"Moved image: {image_src} to {image_dst}\n")
            
            # Move the label file
            if os.path.exists(label_src):
                shutil.move(label_src, label_dst)
                log.write(f"Moved label: {label_src} to {label_dst}\n")

# Paths for the log files
log_test = r'C:\Users\temir\Documents\KULcourses\Thesis_CV\code_yolo8_base\y8\yolo_tracking\COCO_AND_CUSTOM\yolo\test_log.txt'
log_train = r'C:\Users\temir\Documents\KULcourses\Thesis_CV\code_yolo8_base\y8\yolo_tracking\COCO_AND_CUSTOM\yolo\train_log.txt'
log_val = r'C:\Users\temir\Documents\KULcourses\Thesis_CV\code_yolo8_base\y8\yolo_tracking\COCO_AND_CUSTOM\yolo\val_log.txt'

# Move files according to the splits and keep logs
move_files(split_test, first_dataset_images_path, first_dataset_labels_path, output_test_path, log_test)
move_files(split_train, first_dataset_images_path, first_dataset_labels_path, output_train_path, log_train)
move_files(split_val, first_dataset_images_path, first_dataset_labels_path, output_val_path, log_val)

print("Files moved successfully.")
