def process_coordinates(input_file, output_file):
    with open(input_file, 'r') as infile:
        lines = infile.readlines()

    current_frame = None
    delete_coordinates = False

    for i, line in enumerate(lines):
        if line.startswith("Frame"):
            current_frame = int(line.split()[1][:-1])
            time_in_seconds = current_frame / 60  # Convert frames to seconds
            # Check if the frame should be removed
            delete_coordinates = 2 <= (time_in_seconds % 3) < 3
        elif delete_coordinates:
            # Delete all coordinate lines for the current frame
            lines[i] = ""

    with open(output_file, 'w') as outfile:
        outfile.writelines(lines)

if __name__ == "__main__":
    input_filename = "output_coordinates.txt" 
    output_filename = "hashed_coordinates.txt" 

    process_coordinates(input_filename, output_filename)
