import cv2
import numpy as np
from matplotlib import pyplot as plt


kernel = np.ones((5,5),np.uint8)
img1 = cv2.imread('clock_image1.jpg')
img = cv2.imread('clock_image1.jpg',0)
gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

# Create mask
height,width = img.shape
mask = np.zeros((height,width), np.uint8)
edges = cv2.Canny(thresh, 100, 200)

#cv2.imshow('detected ',gray)
cimg=cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100)
for i in circles[0,:]:
    i[2]=i[2]+4
    # Draw on mask
    cv2.circle(mask,(i[0],i[1]),i[2],(255,255,255),thickness=-1)

# Copy that image using that mask
masked_data = cv2.bitwise_and(img1, img1, mask=mask)

# Apply Threshold
_,thresh = cv2.threshold(mask,1,255,cv2.THRESH_BINARY)
# Find Contour
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
x,y,w,h = cv2.boundingRect(contours[0])

# Crop masked_data
crop = masked_data[y+30:y+h-30,x+30:x+w-30]


################################
kernel_size = 5
blur_crop = cv2.GaussianBlur(crop,(kernel_size, kernel_size),0)
low_threshold = 50
high_threshold = 150
edges = cv2.Canny(blur_crop, low_threshold, high_threshold)

rho = 1                     # distance resolution in pixels
theta = np.pi / 180         # angular resolution in radians
threshold = 15              # minimum number of votes 
min_line_length = 100       # minimum number of pixels making up a line
max_line_gap = 10           # maximum gap in pixels between connectable 
line_image = np.copy(crop) * 0 

# Run Hough on edge detected image
# Output "lines" is an array containing endpoints of detected line
lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                    min_line_length, max_line_gap)

for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)

# Draw the lines on the  image
lines_edges = cv2.addWeighted(crop, 0.8, line_image, 1, 0)

cv2.imshow('line_image', line_image)
cv2.imshow('crop', crop)