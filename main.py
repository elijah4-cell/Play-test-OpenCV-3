import cv2
import numpy as np

# read in image file
img = cv2.imread("tennis ball 2.jpeg")

# https://www.pyimagesearch.com/2014/08/04/opencv-python-color-detection/

# https://www.rapidtables.com/web/color/RGB_Color.html

# define the boundaries
lower_range = np.array([0, 100, 100], dtype = "uint8")
upper_range = np.array([200, 255, 255], dtype = "uint8")

# create mask and apply
mask = cv2.inRange(img, lower_range, upper_range)
output = cv2.bitwise_and(img, img, mask = mask)

# output file
cv2.imwrite("output.png", output) 