import cv2

def capture_first_seconds(input_video, output_video, seconds):
    # Video capture
    cap = cv2.VideoCapture(input_video)

    # Get the frames per second
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Calculate the number of frames to capture
    frames_to_capture = int(fps * seconds)

    # Capture frames
    frames_captured = 0
    frames = []

    while frames_captured < frames_to_capture:
        ret, frame = cap.read()
        if not ret:
            break

        frames.append(frame)
        frames_captured += 1

    # Write the frames to a new video file
    fourcc = cv2.VideoWriter_fourcc(*'H264')  # Use H.264 codec
    output_video_writer = cv2.VideoWriter(output_video, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))
    for frame in frames:
        output_video_writer.write(frame)

    # Release resources
    cap.release()
    output_video_writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Input and output video file paths
    input_video_path = "IMG_8256.MOV"
    output_video_path = "output_video.mp4"

    # Specify the number of seconds to capture
    seconds_to_capture = 22

    # Call the function to capture the specified number of seconds
    capture_first_seconds(input_video_path, output_video_path, seconds_to_capture)
