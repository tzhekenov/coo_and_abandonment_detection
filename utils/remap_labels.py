
# This script uses a dictionary remap_first_dataset to re-map the class IDs in the first dataset:
# 0 (Suitcase) → 3
# 1 (Backpack) → 1
# 2 (Handbag) → 2

#root@7ec0340ca13a:/usr/src/boxmot# python tracking/track.py     --source /usr/src/boxmot/data/9505_640_cut_30fr.mp4     --yolo-model yolov8n.pt     --tracking-method deepocsoroot@7ec0340ca13a:/usr/src/boxmot# python tracking/track.py     --source /usr/src/boxmot/data/9505_640_cut_30fr.mp4     --yolo-model yolov8n.pt     --tracking-method deepocsort     --reid-model osnet_x0_25_msmt17.pt     --classes 0 26 30 32 --save-txt

import os

# Paths to your first dataset
first_dataset_labels_path = r'C:\Users\temir\Documents\KULcourses\Thesis_CV\code_yolo8_base\y8\yolo_tracking\COCO_AND_CUSTOM\coco_yolo\labels'
first_dataset_images_path = r'C:\Users\temir\Documents\KULcourses\Thesis_CV\code_yolo8_base\y8\yolo_tracking\COCO_AND_CUSTOM\coco_yolo\images'

# Re-map class IDs for the first dataset
remap_first_dataset = {0: 3, 1: 1, 2: 2}  # Class re-mapping {old_class_id: new_class_id}

# Process and re-map the first dataset
for filename in os.listdir(first_dataset_labels_path):
    if filename.endswith('.txt'):
        label_file_path = os.path.join(first_dataset_labels_path, filename)
        new_labels = []
        with open(label_file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                class_id = int(parts[0])
                if class_id in remap_first_dataset:
                    parts[0] = str(remap_first_dataset[class_id])  # Re-map the class ID
                    new_labels.append(' '.join(parts))
        
        # Write the new labels to the same directory
        with open(label_file_path, 'w') as f:
            f.write('\n'.join(new_labels))

print("Label re-mapping complete.")
