import os
import cv2
import shutil

# Define paths to input folders
input_paths = {
    'test': r'C:\Users\temir\Documents\KULcourses\Thesis_CV\code_yolo8_base\y8\yolo_tracking\COCO_AND_CUSTOM\yolo\obj_Test_data',
    'train': r'C:\Users\temir\Documents\KULcourses\Thesis_CV\code_yolo8_base\y8\yolo_tracking\COCO_AND_CUSTOM\yolo\obj_Train_data',
    'val': r'C:\Users\temir\Documents\KULcourses\Thesis_CV\code_yolo8_base\y8\yolo_tracking\COCO_AND_CUSTOM\yolo\obj_Validation_data'
}

# Define paths to output folders
output_paths = {
    'test': r'C:\Users\temir\Documents\KULcourses\Thesis_CV\code_yolo8_base\y8\yolo_tracking\COCO_AND_CUSTOM\yolo\test',
    'train': r'C:\Users\temir\Documents\KULcourses\Thesis_CV\code_yolo8_base\y8\yolo_tracking\COCO_AND_CUSTOM\yolo\train',
    'val': r'C:\Users\temir\Documents\KULcourses\Thesis_CV\code_yolo8_base\y8\yolo_tracking\COCO_AND_CUSTOM\yolo\val'
}

# Create output directories if they don't exist
for key in output_paths:
    os.makedirs(output_paths[key], exist_ok=True)
    os.makedirs(os.path.join(output_paths[key], 'imgs_with_bbox'), exist_ok=True)

# Define target resolutions
target_16_9 = (1024, 576)
target_4_3 = (640, 480)

# Function to adjust annotations and draw bounding boxes
def adjust_annotations_and_draw_bbox(image, label_path, original_width, original_height, new_width, new_height, output_label_path, output_bbox_image_path):
    with open(label_path, 'r') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        class_id, x_center, y_center, width, height = map(float, line.strip().split())
        
        # Adjust coordinates based on the new image size
        x_center_new = (x_center * original_width) / new_width
        y_center_new = (y_center * original_height) / new_height
        width_new = (width * original_width) / new_width
        height_new = (height * original_height) / new_height
        
        new_lines.append(f"{int(class_id)} {x_center_new:.6f} {y_center_new:.6f} {width_new:.6f} {height_new:.6f}\n")
        
        # Draw the bounding box on the resized image
        x_min = int((x_center_new - width_new / 2) * new_width)
        y_min = int((y_center_new - height_new / 2) * new_height)
        x_max = int((x_center_new + width_new / 2) * new_width)
        y_max = int((y_center_new + height_new / 2) * new_height)
        
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
    
    # Save the adjusted annotation
    with open(output_label_path, 'w') as f:
        f.writelines(new_lines)
    
    # Save the resized image with bounding boxes
    cv2.imwrite(output_bbox_image_path, image)

# Process images and labels
for key, input_path in input_paths.items():
    output_folder = output_paths[key]
    output_bbox_image_folder = os.path.join(output_folder, 'imgs_with_bbox')
    
    for filename in os.listdir(input_path):
        if filename.lower().endswith('.jpg') or filename.lower().endswith('.png'):
            image_path = os.path.join(input_path, filename)
            label_path = os.path.join(input_path, os.path.splitext(filename)[0] + '.txt')
            
            # Change the output filenames to .jpg
            base_filename = os.path.splitext(filename)[0] + '.jpg'
            output_image_path = os.path.join(output_folder, base_filename)
            output_label_path = os.path.join(output_folder, os.path.splitext(base_filename)[0] + '.txt')
            output_bbox_image_path = os.path.join(output_bbox_image_folder, base_filename)
            
            image = cv2.imread(image_path)
            h, w = image.shape[:2]
            
            if w > 1024 or h > 1024:
                aspect_ratio = w / h
                if aspect_ratio == 16/9:
                    # Resize to 1024x576
                    resized_image = cv2.resize(image, target_16_9, interpolation=cv2.INTER_AREA)
                    cv2.imwrite(output_image_path, resized_image)
                    adjust_annotations_and_draw_bbox(resized_image, label_path, w, h, target_16_9[0], target_16_9[1], output_label_path, output_bbox_image_path)
                elif aspect_ratio == 4/3:
                    # Resize to 640x480
                    resized_image = cv2.resize(image, target_4_3, interpolation=cv2.INTER_AREA)
                    cv2.imwrite(output_image_path, resized_image)
                    adjust_annotations_and_draw_bbox(resized_image, label_path, w, h, target_4_3[0], target_4_3[1], output_label_path, output_bbox_image_path)
                else:
                    # Process without resizing if the aspect ratio is unsupported
                    shutil.copy(image_path, output_image_path)
                    shutil.copy(label_path, output_label_path)
                    
                    # Also draw bounding boxes on the original image
                    image_with_bbox = image.copy()
                    adjust_annotations_and_draw_bbox(image_with_bbox, label_path, w, h, w, h, output_label_path, output_bbox_image_path)
            else:
                # If the image is already small, just copy it and draw bbox
                shutil.copy(image_path, output_image_path)
                shutil.copy(label_path, output_label_path)
                
                # Also draw bounding boxes on the original image
                image_with_bbox = image.copy()
                adjust_annotations_and_draw_bbox(image_with_bbox, label_path, w, h, w, h, output_label_path, output_bbox_image_path)

print("Processing complete.")
