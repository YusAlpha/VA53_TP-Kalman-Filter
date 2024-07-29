import cv2
from ultralytics import YOLO

def display_keypoints(video_path, results, index_to_visualize):
    # Video capture using cv2.VideoCapture
    cap = cv2.VideoCapture(video_path)

    # Iterate over the video frames and YOLO results simultaneously
    for video_frame, result in zip(iter_frames(cap), results):
        # Get keypoints for the current frame
        keypoints = result.keypoints.xy.numpy()
        for detected_element in keypoints: 
            # Iterate through each keypoint
            for j, (x, y) in enumerate(detected_element):
                if j == 12 or j == 11:
                  # Draw a point on the frame1
                  cv2.circle(video_frame, (int(x), int(y)), 5, (0, 255, 0), -1)

                  # Display the frame with the keypoint index
                  cv2.putText(video_frame, str(j), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Display the frame
        cv2.imshow("Keypoints", video_frame)
        cv2.waitKey(0)

        key = cv2.waitKey(10)
        if key == 27:
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

def iter_frames(cap):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame

if __name__ == "__main__":
    # Video path
    video_path = "output_video.mp4"

    # Run YOLO on the video and get predictions
    model = YOLO("yolov8x-pose-p6.pt")
    results = model(source=video_path, conf=0.2, classes=0)

    index_to_visualize = 0

    display_keypoints(video_path, results, index_to_visualize)
