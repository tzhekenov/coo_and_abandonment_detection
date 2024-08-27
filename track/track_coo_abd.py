import cv2
import numpy as np
import os
import json
from collections import defaultdict
from datetime import datetime

# Directories
zone_config_path = '/workspaces/coo_and_abandonment_detection/track/zone_config.json'
video_path = '/workspaces/coo_and_abandonment_detection/data/3_Benchmark_case_abandonment.mp4'
gt_path = "/workspaces/coo_and_abandonment_detection/data/gt/gt.txt"

# Load zone configuration
with open(zone_config_path) as f:
    zone_config = json.load(f)

# Convert zone coordinates to numpy array
zone = np.array(zone_config, np.int32).reshape((-1, 1, 2))

class TrackedObject:
    def __init__(self, obj_id, obj_class, center):
        self.id = obj_id
        self.obj_class = obj_class
        self.center = center
        self.entry_frame = None
        self.exit_frame = None
        self.time_spent = 0.0
        self.ownership = {'start_frame': None, 'owned': False}
        self.abandonment_flag = 0
        self.abandonment_start_frame = None

    def is_inside_zone(self, zone):
        return cv2.pointPolygonTest(zone, self.center, False) >= 0

    def update_time_spent(self, current_frame, fps):
        if self.entry_frame is not None and self.exit_frame is None:
            self.time_spent += (current_frame - self.entry_frame) / fps
            self.entry_frame = current_frame  # Update entry_frame to current_frame for continuous time calculation

    def reset_ownership(self):
        self.ownership = {'start_frame': None, 'owned': False}

    def assign_ownership(self, owner_center, current_frame, fps):
        distance = calculate_distance(self.center, owner_center)
        if distance < 200:  # Assign ownership if within 200 pixels
            if self.ownership['start_frame'] is None:
                self.ownership['start_frame'] = current_frame
            elif (current_frame - self.ownership['start_frame']) / fps > 0.3:  # 0.3 seconds threshold
                self.ownership['owned'] = True
        else:
            self.reset_ownership()

    def check_abandonment(self, owner_center, current_frame, fps):
        distance = calculate_distance(self.center, owner_center)
        if distance > 400:  # Check for abandonment if distance > 400 pixels
            if self.abandonment_flag == 0:
                self.abandonment_start_frame = current_frame
                self.abandonment_flag = 1
            elif (current_frame - self.abandonment_start_frame) / fps > 3:  # 3 seconds threshold
                self.abandonment_flag = 2  # Abandonment confirmed
                return True
        else:
            self.abandonment_flag = 0  # Reset if condition is not met
        return False

def calculate_distance(center1, center2):
    return np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)

def draw_zone(frame, zone):
    cv2.polylines(frame, [zone], isClosed=True, color=(0, 255, 255), thickness=2)

def parse_gt_file(gt_path):
    gt_data = defaultdict(list)
    with open(gt_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split(',')
            frame_num = int(parts[0])
            obj_id = int(parts[1])
            x = float(parts[2])
            y = float(parts[3])
            width = float(parts[4])
            height = float(parts[5])
            obj_class = int(parts[7])  # Using the 'visibility' as class
            center = (int(x + width / 2), int(y + height / 2))
            gt_data[frame_num].append((obj_id, obj_class, center, x, y, width, height))
    return gt_data

def log_abandonment(tracked_obj, owner_id, frame_num, abandonment_log):
    log_entry = f"{datetime.now().strftime('%Y-%m-%d')} {datetime.now().strftime('%H:%M:%S')} {frame_num} {tracked_obj.id} {owner_id}\n"
    abandonment_log.write(log_entry)

# Load ground truth data
gt_data = parse_gt_file(gt_path)

# Initialize containers
frame_data = []
tracked_objects = {}
abandonment_log_path = 'abandonment_log.txt'

# Open abandonment log file and write column headers
with open(abandonment_log_path, 'w') as abandonment_log:
    abandonment_log.write("Date Time Frame_Number Abandoned_Object_ID Last_Known_Owner_ID\n")  # Add column names

    # Video properties
    cap = cv2.VideoCapture(video_path)
    fps = 30  # Set FPS to 30
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Output video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, fps, (1920, 1080))

    for frame_num in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Could not read frame {frame_num}.")
            break

        # Draw the zone and the frame number on the video
        draw_zone(frame, zone)
        cv2.putText(frame, f'Frame: {frame_num}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        frame_info = {
            'frame': frame_num,
            'inside_zone': {
                'persons': [],
                'luggage': []
            },
            'outside_zone': {
                'persons': [],
                'luggage': []
            },
            'distances': {},
            'time_in_zone': {},
            'ownership': {}
        }

        centers = {}
        persons = []
        luggage = []

        if frame_num in gt_data:
            for obj_id, obj_class, center, x, y, width, height in gt_data[frame_num]:
                if obj_id not in tracked_objects:
                    tracked_objects[obj_id] = TrackedObject(obj_id, obj_class, center)
                tracked_obj = tracked_objects[obj_id]
                tracked_obj.center = center  # Update center

                centers[tracked_obj.id] = center

                if tracked_obj.is_inside_zone(zone):
                    if tracked_obj.entry_frame is None:
                        tracked_obj.entry_frame = frame_num
                    else:
                        tracked_obj.update_time_spent(frame_num, fps)  # Cumulatively update time spent

                    if obj_class == 1:
                        persons.append(tracked_obj.id)
                        frame_info['inside_zone']['persons'].append(tracked_obj.id)
                    else:
                        luggage.append(tracked_obj.id)
                        frame_info['inside_zone']['luggage'].append(tracked_obj.id)
                else:
                    if tracked_obj.entry_frame and tracked_obj.exit_frame is None:
                        tracked_obj.exit_frame = frame_num
                        tracked_obj.update_time_spent(frame_num, fps)
                    if obj_class == 1:
                        persons.append(tracked_obj.id)
                        frame_info['outside_zone']['persons'].append(tracked_obj.id)
                    else:
                        luggage.append(tracked_obj.id)
                        frame_info['outside_zone']['luggage'].append(tracked_obj.id)

                # Update the time spent in the zone and store it in the frame_info
                frame_info['time_in_zone'][tracked_obj.id] = tracked_obj.time_spent

                time_in_zone = frame_info['time_in_zone'][tracked_obj.id]
                color = (0, 255, 0) if obj_class == 1 else (255, 0, 0)
                cv2.rectangle(frame, (int(x), int(y)), (int(x + width), int(y + height)), color, 2)
                cv2.putText(frame, f'{tracked_obj.obj_class} ID: {tracked_obj.id}', (int(x), int(y) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(frame, f'Time in zone: {time_in_zone:.2f}s', (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                if obj_class in [2, 3]:  # Check for luggage
                    owner_id = next((id for id in persons), None)
                    if owner_id:
                        tracked_obj.assign_ownership(centers[owner_id], frame_num, fps)  # Assign ownership if inside the zone
                        if tracked_obj.check_abandonment(centers[owner_id], frame_num, fps):  # Check for abandonment if the owner leaves
                            log_abandonment(tracked_obj, owner_id, frame_num, abandonment_log)

                        if tracked_obj.ownership['owned']:
                            cv2.putText(frame, f'{tracked_obj.obj_class} ID: {tracked_obj.id} owner: {owner_id}', (int(x), int(y) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Calculate distances between persons and their luggage
        for owner_id in persons:
            for luggage_id in luggage:
                if owner_id != luggage_id:
                    distance = calculate_distance(centers[owner_id], centers[luggage_id])
                    distance_key = f'Distance between Person {owner_id} and Luggage {luggage_id}'
                    frame_info['distances'][distance_key] = distance
                    cv2.putText(frame, f'{distance_key}: {distance:.2f}px', (50, 50 + 30 * luggage_id), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        frame_data.append(frame_info)
        out.write(frame)

    cap.release()
    out.release()

# Save JSON data
with open('framewise_detections.json', 'w') as f:
    json.dump(frame_data, f, indent=4)

print("Processing complete. Video and JSON data have been saved.")
