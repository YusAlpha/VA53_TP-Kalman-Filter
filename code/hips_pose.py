import cv2
from ultralytics import YOLO

def write_coordinates_to_file(file_path, frame_index, coordinates):
    with open(file_path, 'a') as file:
        file.write(f"Frame {frame_index}:\n")
        for coord in coordinates:
            file.write(f"({coord[0]}, {coord[1]}, {coord[2]}, {coord[3]})\n")

if __name__ == "__main__":
    # Video path
    video_path = "output_video.mp4"

    # Output file path
    output_file_path = "output_coordinates.txt"

    # Run YOLO on the video and get predictions
    model = YOLO("yolov8x-pose-p6.pt")
    results = model(source=video_path, conf=0.2, classes=0)

    for index, r in enumerate(results):
        keypoints = r.keypoints.xy.numpy()

        # Extract and print coordinates
        frame_coordinates = []
        for detected_person in keypoints:
            if len(detected_person) > 12 and detected_person[11] is not None and detected_person[12] is not None:
              x_coord_1 = int(detected_person[11][0])
              y_coord_1 = int(detected_person[11][1])
              x_coord_2 = int(detected_person[12][0])
              y_coord_2 = int(detected_person[12][1])
              frame_coordinates.append((x_coord_1, y_coord_1, x_coord_2, y_coord_2))
        # Write coordinates to the output file
        write_coordinates_to_file(output_file_path, index, frame_coordinates)
