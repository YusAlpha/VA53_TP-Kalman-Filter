import matplotlib.pyplot as plt
import numpy as np

def load_coordinates_from_file(file_path):
    coordinates = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            values = line.strip().split(',')
            
            if len(values) == 2:
                x, z = map(lambda val: float(val.strip()) if val.strip().lower() != 'none' else np.nan, values)
                coordinates.append((x, z))
    return coordinates

def plot_coordinates(measured_coordinates, kalman_estimated_coordinates, title, xlabel, ylabel):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 8))
    
    measured_x, measured_z = zip(*measured_coordinates)
    kalman_x, kalman_z = zip(*kalman_estimated_coordinates)

    ax1.plot(measured_x, label='Measured', marker='o', markersize=8, color='blue')
    ax1.plot(kalman_x, label='Kalman Estimated', marker='o', markersize=4, color='orange')
    ax1.set_ylabel('X Coordinate')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(measured_z, label='Measured', marker='o', markersize=8, color='blue')
    ax2.plot(kalman_z, label='Kalman Estimated', marker='o', markersize=4, color='orange')
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel('Z Coordinate')
    ax2.legend()
    ax2.grid(True)

    plt.suptitle(title)
    plt.show()

if __name__ == "__main__":
    measured_coordinates = load_coordinates_from_file('hashed_measured_coordinates.txt')
    kalman_estimated_coordinates = load_coordinates_from_file('hashed_kalman_estimated_coordinates.txt')

    plot_coordinates(measured_coordinates, kalman_estimated_coordinates, 'X and Z Coordinates - Measured vs Kalman Estimated', 'Frame', '')
