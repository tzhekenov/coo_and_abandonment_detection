import os
import shutil
import json
import cv2
import matplotlib.pyplot as plt

# Define the COCO categories you are interested in
categories_of_interest = {
    "person": 1,     # COCO category ID for person
    "suitcase": 33,  # COCO category ID for suitcase
    "backpack": 27,  # COCO category ID for backpack
    "handbag": 31    # COCO category ID for handbag
}

# Define paths
images_path = r'C:\Users\temir\Documents\KULcourses\Thesis_CV\code_yolo8_base\y8\yolo_tracking\COCO_AND_CUSTOM\coco_subset\images'
annotations_path = r'C:\Users\temir\Documents\KULcourses\Thesis_CV\code_yolo8_base\y8\yolo_tracking\COCO_AND_CUSTOM\coco_subset\annotations'
output_images_path = r'C:\Users\temir\Documents\KULcourses\Thesis_CV\code_yolo8_base\y8\yolo_tracking\COCO_AND_CUSTOM\coco_yolo\images'
output_labels_path = r'C:\Users\temir\Documents\KULcourses\Thesis_CV\code_yolo8_base\y8\yolo_tracking\COCO_AND_CUSTOM\coco_yolo\labels'
output_images_with_bboxes_path = r'C:\Users\temir\Documents\KULcourses\Thesis_CV\code_yolo8_base\y8\yolo_tracking\COCO_AND_CUSTOM\images_with_bboxes'


# Create directories if they don't exist
os.makedirs(output_images_path, exist_ok=True)
os.makedirs(output_labels_path, exist_ok=True)
os.makedirs(output_images_with_bboxes_path, exist_ok=True)

# Get the list of filenames without extensions from both directories
image_filenames = {os.path.splitext(filename)[0] for filename in os.listdir(images_path)}
annotation_filenames = {os.path.splitext(filename)[0] for filename in os.listdir(annotations_path)}

# Process only the files that have matching image and annotation names
matching_filenames = list(image_filenames.intersection(annotation_filenames))[:]

# Function to convert COCO bbox to YOLO format
def convert_bbox_to_yolo(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = box[0] + box[2] / 2.0
    y = box[1] + box[3] / 2.0
    w = box[2]
    h = box[3]
    x = x * dw
    y = y * dh
    w = w * dw
    h = h * dh
    return (x, y, w, h)

# Track the number of successful copies
successful_copies = 0

# Iterate through the first 100 matching files
for filename_no_ext in matching_filenames:
    # Load the corresponding JSON annotation file
    annotation_file_path = os.path.join(annotations_path, filename_no_ext + '.json')
    with open(annotation_file_path, 'r') as f:
        annotation_data = json.load(f)

    # Load the image
    src_image_path = os.path.join(images_path, filename_no_ext + '.jpg')
    image = cv2.imread(src_image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_height, image_width, _ = image.shape

    # Iterate over the annotations
    yolo_annotations = []
    for anno in annotation_data:
        category_id = anno['category_id']
        if category_id in categories_of_interest.values():
            bbox = anno['bbox']
            yolo_bbox = convert_bbox_to_yolo((image_width, image_height), bbox)
            yolo_annotations.append(f"{list(categories_of_interest.values()).index(category_id)} " + " ".join(map(str, yolo_bbox)))

            # Draw the bounding box on the image
            x_min = int(yolo_bbox[0] * image_width - (yolo_bbox[2] * image_width) / 2)
            y_min = int(yolo_bbox[1] * image_height - (yolo_bbox[3] * image_height) / 2)
            width = int(yolo_bbox[2] * image_width)
            height = int(yolo_bbox[3] * image_height)
            cv2.rectangle(image, (x_min, y_min), (x_min + width, y_min + height), (255, 0, 0), 2)

    # Write YOLO annotation to file if there are any
    if yolo_annotations:
        label_file_path = os.path.join(output_labels_path, filename_no_ext + '.txt')
        with open(label_file_path, 'w') as label_file:
            label_file.write("\n".join(yolo_annotations))

        # Copy the corresponding image to the output directory
        try:
            shutil.copy(src_image_path, output_images_path)
            successful_copies += 1
            print(f"Copied: {src_image_path} to {output_images_path}")
        except Exception as e:
            print(f"Error copying {src_image_path}: {e}")

        # Save the image with bounding boxes drawn on it
        output_bbox_image_path = os.path.join(output_images_with_bboxes_path, filename_no_ext + '_bbox.jpg')
        plt.imsave(output_bbox_image_path, image)
        print(f"Saved image with bounding boxes to: {output_bbox_image_path}")

# Print the number of successful copies
print(f"\nTotal successful image copies: {successful_copies}")
print("Conversion to YOLO format and saving images with bounding boxes completed.")





















# import os
# import shutil
# import json

# # Define the COCO categories you are interested in
# categories_of_interest = {
#     "suitcase": 33,  # COCO category ID for suitcase
#     "backpack": 27,  # COCO category ID for backpack
#     "handbag": 31    # COCO category ID for handbag
# }

# # Define paths
# images_path = r'C:\Users\temir\Documents\KULcourses\Thesis_CV\code_yolo8_base\y8\yolo_tracking\COCO_AND_CUSTOM\coco_subset\images'
# annotations_path = r'C:\Users\temir\Documents\KULcourses\Thesis_CV\code_yolo8_base\y8\yolo_tracking\COCO_AND_CUSTOM\coco_subset\annotations'
# output_images_path = r'C:\Users\temir\Documents\KULcourses\Thesis_CV\code_yolo8_base\y8\yolo_tracking\COCO_AND_CUSTOM\coco_yolo\images'
# output_labels_path = r'C:\Users\temir\Documents\KULcourses\Thesis_CV\code_yolo8_base\y8\yolo_tracking\COCO_AND_CUSTOM\coco_yolo\labels'

# # Create directories if they don't exist
# os.makedirs(output_images_path, exist_ok=True)
# os.makedirs(output_labels_path, exist_ok=True)

# # Get the list of filenames without extensions from both directories
# image_filenames = {os.path.splitext(filename)[0] for filename in os.listdir(images_path)}
# annotation_filenames = {os.path.splitext(filename)[0] for filename in os.listdir(annotations_path)}

# # Process only the files that have matching image and annotation names
# matching_filenames = image_filenames.intersection(annotation_filenames)

# # Function to convert COCO bbox to YOLO format
# def convert_bbox_to_yolo(size, box):
#     dw = 1. / size[0]
#     dh = 1. / size[1]
#     x = box[0] + box[2] / 2.0
#     y = box[1] + box[3] / 2.0
#     w = box[2]
#     h = box[3]
#     x = x * dw
#     y = y * dh
#     w = w * dw
#     h = h * dh
#     return (x, y, w, h)

# # Track the number of successful copies
# successful_copies = 0

# # Iterate through matching files
# for filename_no_ext in matching_filenames:
#     # Load the corresponding JSON annotation file
#     annotation_file_path = os.path.join(annotations_path, filename_no_ext + '.json')
#     with open(annotation_file_path, 'r') as f:
#         annotation_data = json.load(f)

#     # Get the image information
#     image_width = None
#     image_height = None

#     # Iterate over the annotations
#     yolo_annotations = []
#     for anno in annotation_data:
#         if image_width is None or image_height is None:
#             # Use the first bounding box dimensions as the size for YOLO conversion
#             # If image size is not provided, we will use these dimensions
#             image_width = int(anno['bbox'][2])  # Width from the first annotation bbox
#             image_height = int(anno['bbox'][3])  # Height from the first annotation bbox
        
#         category_id = anno['category_id']
#         if category_id in categories_of_interest.values():
#             bbox = anno['bbox']
#             yolo_bbox = convert_bbox_to_yolo((image_width, image_height), bbox)
#             yolo_annotations.append(f"{list(categories_of_interest.values()).index(category_id)} " + " ".join(map(str, yolo_bbox)))

#     # Write YOLO annotation to file if there are any
#     if yolo_annotations:
#         label_file_path = os.path.join(output_labels_path, filename_no_ext + '.txt')
#         with open(label_file_path, 'w') as label_file:
#             label_file.write("\n".join(yolo_annotations))
        
#         # Copy the corresponding image to the output directory
#         src_image_path = os.path.join(images_path, filename_no_ext + '.jpg')
#         try:
#             shutil.copy(src_image_path, output_images_path)
#             successful_copies += 1
#             print(f"Copied: {src_image_path} to {output_images_path}")
#         except Exception as e:
#             print(f"Error copying {src_image_path}: {e}")

# # Print the number of successful copies
# print(f"\nTotal successful image copies: {successful_copies}")

# print("Conversion to YOLO format completed.")
