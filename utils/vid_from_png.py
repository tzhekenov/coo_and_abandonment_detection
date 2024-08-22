import cv2
import os

# Directory containing the PNG files
image_folder = r'C:\Users\temir\Documents\KULcourses\Thesis_CV\yolo9_heatmap\supervision\examples\time_in_zone\from_annotations\cam4_cctv26\obj_train_data'

# Output video file path
video_name = 'output_video.avi'

# Get a list of all PNG files in the directory
images = [img for img in os.listdir(image_folder) if img.endswith(".PNG")]
images.sort()  # Sort the images by filename

# Ensure there are images in the folder
if len(images) == 0:
    print("No images found in the specified directory.")
    exit()

# Read the first image to get the size
first_image_path = os.path.join(image_folder, images[0])
frame = cv2.imread(first_image_path)
height, width, layers = frame.shape

# Initialize the video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(video_name, fourcc, 30, (width, height))  # Assuming 30 FPS

# Loop through all images and write them to the video
for image in images:
    img_path = os.path.join(image_folder, image)
    frame = cv2.imread(img_path)
    out.write(frame)

# Release the video writer
out.release()

print(f"Video saved as {video_name}")
