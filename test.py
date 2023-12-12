import sys
import cv2 as cv
import numpy as np
def euclidean_distance(line1, line2):
    midpoint1 = [(line1[0][0] + line1[0][2]) / 2, (line1[0][1] + line1[0][3]) / 2]
    midpoint2 = [(line2[0][0] + line2[0][2]) / 2, (line2[0][1] + line2[0][3]) / 2]
    return np.linalg.norm(np.array(midpoint1) - np.array(midpoint2))

def merge_close_hough_lines(lines, distance_threshold):
    merged_lines = []

    while len(lines) > 0:
        current_line = lines[0]
        group = [current_line]

        i = 0
        while i < len(lines):
            distance = euclidean_distance(current_line, lines[i])

            if distance < distance_threshold:
                group.append(lines[i])
                lines = np.delete(lines, i, axis=0)
            else:
                i += 1

        # Merge lines in the group
        if len(group) > 1:
            merged_line = np.mean(np.array(group), axis=0).astype(int)
            merged_lines.append(merged_line.tolist())
        else:
            merged_lines.append(group[0][0].tolist())

    return merged_lines

def find_intersection(line1, line2):
    x1, y1, x2, y2 = line1[0]
    x3, y3, x4, y4 = line2[0]

    # Check if the lines are parallel
    det = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if det == 0:
        return None  # Lines are parallel, no intersection

    # Calculate the intersection point
    intersection_x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / det
    intersection_y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / det

    return [int(intersection_x), int(intersection_y)]

def remove_exceeding_part(line, intersection_point, distance_threshold):
    x1, y1, x2, y2 = line[0]
    intersection_x, intersection_y = intersection_point

    # Calculate the distance from each endpoint to the intersection point
    distance1 = np.linalg.norm(np.array([x1, y1]) - np.array([intersection_x, intersection_y]))
    distance2 = np.linalg.norm(np.array([x2, y2]) - np.array([intersection_x, intersection_y]))

    # Keep the longer part of the line within the distance threshold
    if distance1 <= distance_threshold and distance2 <= distance_threshold:
        return line
    elif distance1 >= distance2:
        return [[intersection_x, intersection_y, x1, y1]]
    else:
        return [[x2, y2, intersection_x, intersection_y]]

def main(argv):
    
    default_file = 'clock_image1.jpg'
    filename = argv[0] if len(argv) > 0 else default_file
    
    img = cv.imread(cv.samples.findFile(filename), cv.IMREAD_UNCHANGED)
    if img is None:
        print ('Error opening image!')
        print ('Usage: hough_lines.py [image_name -- default ' + default_file + '] \n')
        return -1
    
    # Convert an image to gray
    src = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Edge detection
    dst = cv.Canny(src, 150, 200)
    
    # Copy edges to the images that will display the results in BGR
    cdstP = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
    linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 50,None, 100, 10)
    merged_lines = merge_close_hough_lines(linesP,30)
    
    if merged_lines is not None:
        intersection_point = find_intersection(merged_lines[0], merged_lines[1])

        if intersection_point is not None:
            # Draw a circle at the intersection point
            cv.circle(cdstP, tuple(intersection_point), 1, (0, 255, 0), cv.LINE_AA)

            # Remove the exceeding part of the lines based on the distance from the intersection point
            distance_threshold = 50  # Adjust this threshold as needed
            filtered_lines = [remove_exceeding_part(line, intersection_point, distance_threshold) for line in merged_lines]
            filtered_lines = [line for line in filtered_lines if line is not None]

            # Draw the filtered lines
            for l in filtered_lines:
                cv.line(cdstP, (l[0][0], l[0][1]), (l[0][2], l[0][3]), (0, 255, 0), 3, cv.LINE_AA)
                
            

    # Display or output the time
    # cv.imshow("Src", src)
    cv.imshow("Detected Lines", cdstP)
    
    cv.waitKey()
    return 0
    
if __name__ == "__main__":
    main(sys.argv[1:])