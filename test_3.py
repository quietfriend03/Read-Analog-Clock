import numpy as np

def euclidean_distance(line1, line2):
    midpoint1 = [(line1[0] + line1[2]) / 2, (line1[1] + line1[3]) / 2]
    midpoint2 = [(line2[0] + line2[2]) / 2, (line2[1] + line2[3]) / 2]
    return np.linalg.norm(np.array(midpoint1) - np.array(midpoint2))

def merge_close_hough_lines(lines, distance_threshold):
    merged_lines = []

    while lines:
        current_line = lines.pop(0)
        group = [current_line]

        i = 0
        while i < len(lines):
            distance = euclidean_distance(current_line[0], lines[i][0])

            if distance < distance_threshold:
                group.append(lines.pop(i))
            else:
                i += 1

        # Merge lines in the group
        if len(group) > 1:
            merged_line = np.mean(np.array(group), axis=0).astype(int)
            merged_lines.append(merged_line.tolist())
        else:
            merged_lines.append(group[0])

    return merged_lines

# Example usage
hough_lines = [
    np.array([[156, 184, 295, 94]]),
    np.array([[53, 92, 197, 175]]),
    np.array([[72, 113, 195, 184]]),
    np.array([[158, 255, 182, 139]]),
    np.array([[87, 112, 197, 176]]),
    np.array([[172, 173, 295, 93]])
]
print(hough_lines)
distance_threshold = 20  # Adjust this threshold as needed

merged_lines = merge_close_hough_lines(hough_lines, distance_threshold)

# Print the merged lines
for line in merged_lines:
    print(line)