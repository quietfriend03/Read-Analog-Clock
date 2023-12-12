import cv2
import numpy as np
from skimage.morphology import skeletonize

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
    
# Read image
img = cv2.imread('clock_image1.jpg')
hh, ww = img.shape[:2]

# convert to gray
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# threshold
thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]

# invert so shapes are white on black background
thresh = 255 - thresh

# get contours and save area
cntrs_info = []
contours = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]
index=0
for cntr in contours:
    area = cv2.contourArea(cntr)
    cntrs_info.append((index,area))
    index = index + 1

# sort contours by area
def takeSecond(elem):
    return elem[1]
cntrs_info.sort(key=takeSecond, reverse=True)

# get third largest contour
arms = np.zeros_like(thresh)
index_third = cntrs_info[2][0]
cv2.drawContours(arms,[contours[index_third]],0,(1),-1)

arms_thin = skeletonize(arms)
arms_thin = (255*arms_thin).clip(0,255).astype(np.uint8)

# get hough lines and draw on copy of input
result = img.copy()
lineThresh = 15
minLineLength = 20
maxLineGap = 100 
lines = cv2.HoughLinesP(arms_thin, 1, np.pi/180, lineThresh, None, minLineLength, maxLineGap)
merged_lines = merge_close_hough_lines(lines, 10)
print(merged_lines)
if merged_lines is not None:
    intersection_point = find_intersection(merged_lines[0], merged_lines[1])

    if intersection_point is not None:
        # Draw a circle at the intersection point
        cv2.circle(result, tuple(intersection_point), 1, (0, 255, 0), cv2.LINE_AA)
            # Draw the filtered lines
        for l in merged_lines:
            cv2.line(result, (l[0][0], l[0][1]), (l[0][2], l[0][3]), (0, 255, 0), 3, cv2.LINE_AA)

cv2.imshow('result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()