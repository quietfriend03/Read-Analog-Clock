import cv2
import numpy as np
from skimage.morphology import skeletonize

def euclidean_distance(line1, line2):
    midpoint1 = [(line1[0][0] + line1[0][2]) / 2, (line1[0][1] + line1[0][3]) / 2]
    midpoint2 = [(line2[0][0] + line2[0][2]) / 2, (line2[0][1] + line2[0][3]) / 2]
    return np.linalg.norm(np.array(midpoint1) - np.array(midpoint2))

def angle_between_lines(line1, line2):
    angle1 = np.arctan2(line1[0][3] - line1[0][1], line1[0][2] - line1[0][0])
    angle2 = np.arctan2(line2[0][3] - line2[0][1], line2[0][2] - line2[0][0])
    return np.abs(angle1 - angle2) * (180 / np.pi)

def merge_close_hough_lines(lines, distance_threshold, angle_threshold):
    merged_lines = []

    while len(lines) > 0:
        current_line = lines[0]
        group = [current_line]

        i = 0
        while i < len(lines):
            distance = euclidean_distance(current_line, lines[i])
            angle_diff = angle_between_lines(current_line, lines[i])
            # print("Distance",distance)
            # print("Angle",angle_diff)
            if (distance < distance_threshold) and (angle_diff < angle_threshold):
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

def get_intersection_points_from_lines(lines):
    intersection_points = []

    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            line1 = lines[i]
            line2 = lines[j]

            x1, y1, x2, y2 = line1[0]
            x3, y3, x4, y4 = line2[0]

            # Check if lines are not parallel (slope condition)
            if (x2 - x1) * (y4 - y3) != (x4 - x3) * (y2 - y1):
                # Solve for intersection point using linear algebra
                A = np.array([
                    [y2 - y1, x1 - x2],
                    [y4 - y3, x3 - x4]
                ])

                B = np.array([x1 * (y2 - y1) - y1 * (x2 - x1), x3 * (y4 - y3) - y3 * (x4 - x3)])

                intersection_point = np.linalg.solve(A, B)
                intersection_points.append((int(intersection_point[1]), int(intersection_point[0])))

    return intersection_points

def get_time_from_clock_hands(hour_hand, minute_hand, second_hand ):
    # Calculate the angles of the clock hands with respect to the 12 o'clock position
    hour_angle = np.arctan2(hour_hand[0][3] - hour_hand[0][1], hour_hand[0][2] - hour_hand[0][0]) * (180 / np.pi)
    minute_angle = np.arctan2(minute_hand[0][3] - minute_hand[0][1], minute_hand[0][2] - minute_hand[0][0]) * (180 / np.pi)
    second_angle = np.arctan2(second_hand[0][3] - second_hand[0][1], second_hand[0][2] - second_hand[0][0]) * (180 / np.pi)
    
    print(hour_angle)
    print(minute_angle)
    print(second_angle)
    # Normalize angles to be positive
    hour_angle = (hour_angle + 450) % 360
    minute_angle = (minute_angle + 450) % 360
    second_angle = (second_angle + 450) % 360

    # Convert angles to time values
    hour = int((hour_angle / 30) % 12)  # Each hour is 30 degrees, and modulus 12 for 12-hour clock
    minute = int(minute_angle / 6)  # Each minute is 6 degrees
    second = int(second_angle / 6)  # Each second is 6 degrees

    return hour, minute, second
# Read image
img = cv2.imread('clock_image5.jpg')
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
lineThresh = 37
minLineLength = 55
maxLineGap = 100
lines = cv2.HoughLinesP(arms_thin, 1, np.pi/180, lineThresh, None, minLineLength, maxLineGap)
merged_lines = merge_close_hough_lines(lines, 80, 0.5)
# merged_lines = merge_close_hough_lines(merged_lines, 85, 0.5)
print(merged_lines)

if merged_lines is not None:
    # Sort merged lines by length
    merged_lines.sort(key=lambda x: np.linalg.norm(np.array([x[0][0], x[0][1]]) - np.array([x[0][2], x[0][3]])),
                      reverse=True)
    # Assign labels to lines
    labels = ["second", "minute", "hour"]
    labeled_lines = dict(zip(labels, merged_lines))
    # Draw the clock hands and label them by length
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]
    for label, l in labeled_lines.items():
        color = colors[labels.index(label)]     
        cv2.line(result, (l[0][0], l[0][1]), (l[0][2], l[0][3]), color, 3, cv2.LINE_AA)
        cv2.putText(result, label, (l[0][2], l[0][3]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    hour, minute, second = get_time_from_clock_hands(
        labeled_lines["hour"],
        labeled_lines["minute"],
        labeled_lines["second"]
    )
    
    cv2.putText(result, f"{hour:02d}:{minute:02d}:{second:02d}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    cv2.imshow('result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Not enough lines found to determine clock hands.")