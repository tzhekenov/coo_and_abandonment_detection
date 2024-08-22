import cv2

def extract_frames(input_video_path, output_video_path, start_frame, end_frame):
    # Open the input video
    cap = cv2.VideoCapture(input_video_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 file
    
    # Create the VideoWriter object
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    current_frame = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if start_frame <= current_frame <= end_frame:
            out.write(frame)
        
        current_frame += 1
        
        if current_frame > end_frame:
            break
    
    # Release everything
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Usage
input_video_path = r"C:\Users\temir\Documents\KULcourses\Thesis_CV\code_yolo8_base\y8\yolo_tracking\assets\clips\9505_30fr_cut.mp4"
output_video_path = '9505_30fr_cut.mp4'
start_frame = 130
end_frame = 400

extract_frames(input_video_path, output_video_path, start_frame, end_frame)