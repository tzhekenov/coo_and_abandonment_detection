#tracklab. virt env. 
#Downloads images and based on annotations. 
#works after getCOCO.ipynb is executed (in case it fails to download images)

import os
import json
import requests
from pycocotools.coco import COCO
import urllib.request

# Define the class names and their corresponding COCO class IDs
class_names = ['handbag', 'backpack', 'suitcase']
class_ids = [27, 31, 33]  # These are the category IDs for handbag, backpack, and suitcase in COCO

# Directory to save the images and annotations
save_dir = 'coco_subset'
images_dir = os.path.join(save_dir, 'images_folder')
annotations_dir = os.path.join(save_dir, 'annotations')
os.makedirs(images_dir, exist_ok=True)
os.makedirs(annotations_dir, exist_ok=True)

# Use the already existing annotations
annotation_file = r'C:\Users\temir\Documents\KULcourses\Thesis_CV\code_yolo8_base\y8\yolo_tracking\tracking\coco_subset\annotations\annotations_trainval2017\annotations\instances_train2017.json'
coco = COCO(annotation_file)

# Get image IDs for the specified classes
img_ids = []
for class_id in class_ids:
    img_ids += coco.getImgIds(catIds=[class_id])

# Remove duplicates
img_ids = list(set(img_ids))

print(f"Found {len(img_ids)} images for the specified classes.")

# Function to download an image
def download_image(img_url, img_path):
    try:
        urllib.request.urlretrieve(img_url, img_path)
        print(f"Successfully downloaded {img_url} to {img_path}")
    except Exception as e:
        print(f"Error downloading {img_url}: {e}")

# Download images and save annotations for each image
for img_id in img_ids:
    img_info = coco.loadImgs(img_id)[0]
    img_url = img_info['coco_url']
    img_path = os.path.join(images_dir, img_info['file_name'])

    # Download image if it doesn't exist
    if not os.path.exists(img_path):
        print(f"Downloading image: {img_info['file_name']} from {img_url}")
        download_image(img_url, img_path)
    else:
        print(f"Image already exists: {img_path}")

    # Get annotations for the image
    ann_ids = coco.getAnnIds(imgIds=[img_id], catIds=class_ids, iscrowd=None)
    anns = coco.loadAnns(ann_ids)

    # Save annotations to a file
    annotation_path = os.path.join(annotations_dir, f"{img_info['file_name'].split('.')[0]}.json")
    with open(annotation_path, 'w') as f:
        json.dump(anns, f)
        print(f"Annotations saved to {annotation_path}")

print("Download and annotation extraction completed.")
