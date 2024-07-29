from filterpy.kalman import KalmanFilter
import numpy as np
import cv2

def initialize_kalman_filter(initial_position):
    kf = KalmanFilter(dim_x=4, dim_z=2)
    
    # Define values for dt and the variances (sigma values)
    dt = 1 / 60.0
    sigma_pos_x = 1e-4
    sigma_vel_x = 1e-2
    sigma_pos_z = 1e-4
    sigma_vel_z = 1e-2

    sigma_pos_x_measure = 0.01
    sigma_pos_z_measure = 0.1

    # State transition matrix
    kf.F = np.array([[1, dt, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, dt],
                     [0, 0, 0, 1]])

    # Measurement matrix
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 0, 1, 0]])

    # Covariance matrices
    kf.P *= np.array([[sigma_pos_x**2, 0, 0, 0],
                      [0, sigma_vel_x**2, 0, 0],
                      [0, 0, sigma_pos_z**2, 0],
                      [0, 0, 0, sigma_vel_z**2]])

    # Measurement noise covariance matrix
    kf.R *= np.array([[sigma_pos_x_measure**2, 0],
                      [0, sigma_pos_z_measure**2]])

    # Process noise covariance matrix
    kf.Q *= np.array([[sigma_pos_x**2, 0, 0, 0],
                      [0, sigma_vel_x**2, 0, 0],
                      [0, 0, sigma_pos_z**2, 0],
                      [0, 0, 0, sigma_vel_z**2]])
    
    kf.x = np.array([initial_position[0], 0, initial_position[1], 0])
    # print("kf.x", kf.x)

    return kf

def load_positions_from_file(file_path):
    positions = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        previous_position = None
        position = None
        for line in lines:
            if line.startswith("Frame"):
                if position is not None:
                    positions.append(position)
                    previous_position = position
                    position = None
                else:
                    positions.append([None, None])                    
            else:
                if line.strip():  # Check if the line is not empty
                    x, y, x2, y2 = map(int, line.strip().strip('()').split(','))
                    centroid_x = int((x + x2) / 2)
                    centroid_y = int((y + y2) / 2)
                    if position is None:
                        position = [centroid_x, centroid_y]
                    # If the new position is closer
                    elif np.linalg.norm(np.array([centroid_x, centroid_y]) - np.array(previous_position)) < np.linalg.norm(position - np.array(previous_position)):
                        position = [centroid_x, centroid_y]
    positions.pop(0)
    # print("positions", positions)
    return positions

def save_coordinates_to_file(file_path, coordinates):
    with open(file_path, 'w') as f:
        for coord in coordinates:
            f.write(f"{coord[0]}, {coord[1]}\n")


def main():

    # Load positions from the text file
    positions = load_positions_from_file('hashed_coordinates.txt')
    
    # Initialize Kalman filter
    kf = initialize_kalman_filter(positions[0] if positions[0] is not None else [0, 0])


    # Video capture
    cap = cv2.VideoCapture("output_video.mp4")

    # Get the video's original width, height, and frames per second
    original_width = int(cap.get(3))
    original_height = int(cap.get(4))
    fps = int(cap.get(5))

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can use other codecs as well, like 'MJPG'
    output_video = cv2.VideoWriter('kalman.avi', fourcc, fps, (original_width, original_height))

    previous_position = None
    
    # Liste pour stocker les coordonnées des points de mesure
    measured_coordinates = []

    # Liste pour stocker les coordonnées estimées par le filtre de Kalman
    kalman_estimated_coordinates = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break


        # Get the current frame number
        frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1

        # Get the corresponding position for the current frame
        pos = positions[frame_num] if frame_num < len(positions) else None

        # Measurement update if there is an observation
        if pos is not None and pos != [None, None]:
            current_measurement = np.array([pos[0], pos[1]])  # Take only the position components
            # print("current_measurement", current_measurement)
            kf.update(current_measurement)
            
            measured_coordinates.append((pos[0], pos[1]))
        else:
            measured_coordinates.append((None, None))

    
        # If there is no observation, continue with the prediction step
        kf.predict()

        # Get the predicted state estimate
        current_state = kf.x
        # print("current_state", current_state)
        
        kalman_estimated_coordinates.append((current_state[0], current_state[2]))


        # Draw Kalman filter prediction on the frame
        predicted_x, predicted_y = int(current_state[0]), int(current_state[2])
        # print("predicted_x, predicted_y", predicted_x, predicted_y)
        cv2.circle(frame, (predicted_x, predicted_y + 5), 5, (0, 0, 255), -1)

        # Draw centroid of the observation in blue
        if pos is not None and pos != [None, None]:
            obs_x, obs_y = pos[0], pos[1]
            cv2.circle(frame, (obs_x, obs_y + 5), 5, (255, 0, 0), -1)

        # Print the position of the current frame
        # print(f"Frame {frame_num} Position: {pos if pos else 'No Observation'}")
        # print("=====================================")

        # Update previous position for the next iteration
        previous_position = pos if pos else previous_position

        # Write the frame to the output video
        output_video.write(frame)
        
         # Enregistrer les coordonnées dans les fichiers
        save_coordinates_to_file('hashed_measured_coordinates.txt', measured_coordinates)
        save_coordinates_to_file('hashed_kalman_estimated_coordinates.txt', kalman_estimated_coordinates)

        # Display the frame
        cv2.imshow("Kalman Filter", frame)

        key = cv2.waitKey(10)
        if key == 27:
            break
    # Release resources
    cap.release()
    output_video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
