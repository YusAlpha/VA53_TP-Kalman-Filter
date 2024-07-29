import cv2
from ultralytics import YOLO
import numpy as np

cap = cv2.VideoCapture("IMG_8256.MOV")
model = YOLO("yolov8m.pt")

# Get the video's original width, height, and frames per second
original_width = int(cap.get(3))
original_height = int(cap.get(4))
fps = int(cap.get(5))

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID') 
output_video = cv2.VideoWriter('output.avi', fourcc, fps, (original_width, original_height))

# List to store bbox positions
bbox_positions = []

# Set the target duration to 5 seconds
target_duration = 5  # in seconds

# Calculate the target frame count based on the frame rate
target_frame_count = fps * target_duration

current_frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    result = results[0]
    bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
    classes = np.array(result.boxes.cls.cpu(), dtype="int")

    frame_positions = []  # List to store bbox positions for the current frame

    for cls, bbox in zip(classes, bboxes):
        if cls == 0:
            (x, y, x2, y2) = bbox

            # # Calculate the centroid of the bounding box
            # centroid_x = int((x + x2) / 2)
            # centroid_y = int((y + y2) / 2)
           
            cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 225), 2)
            cv2.putText(frame, str(cls), (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 225), 2)

            # Save bbox position for the current frame
            frame_positions.append((x, y, x2, y2))

    # Append frame positions to the overall list
    bbox_positions.append(frame_positions)

    # Write the frame to the output video
    output_video.write(frame)

    current_frame_count += 1

    # Check if the target frame count has been reached
    # if current_frame_count >= target_frame_count:
    #     break

    key = cv2.waitKey(10)
    if key == 27:
        break

# Release resources
cap.release()
output_video.release()
cv2.destroyAllWindows()

# Save bbox positions to a file
with open('bbox_positions.txt', 'w') as f:
    for frame_idx, frame_positions in enumerate(bbox_positions):
        f.write(f"Frame {frame_idx}:\n")
        for position in frame_positions:
            f.write(f"{position}\n")
