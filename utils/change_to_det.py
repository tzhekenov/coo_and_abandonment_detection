import os

# Path to the directory containing the detection results
tracking_dir  = r"C:\Users\temir\Documents\KULcourses\Thesis_CV\code_yolo8_base\y8\yolo_tracking\COCO_AND_CUSTOM\track_9505\tracking_results\labels"
output_file = r"C:\Users\temir\Documents\KULcourses\Thesis_CV\code_yolo8_base\y8\yolo_tracking\COCO_AND_CUSTOM\track_9505\train\det\det.txt"

with open(output_file, 'w') as out_file:
    for track_file in os.listdir(tracking_dir):
        if track_file.endswith('.txt'):
            frame_id = int(track_file.split('_')[-1].split('.')[0])  # Extract frame number from the filename
            with open(os.path.join(tracking_dir, track_file), 'r') as in_file:
                for line in in_file:
                    parts = line.strip().split()
                    x_center, y_center, width, height, confidence = map(float, parts[1:6])
                    
                    # Convert center x, y to top-left x, y for MOT format
                    x_min = x_center - width / 2
                    y_min = y_center - height / 2

                    out_file.write(f"{frame_id},-1,{x_min},{y_min},{width},{height},{confidence}\n")