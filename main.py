import cv2
import numpy as np

# read in image file
img = cv2.imread("tennis ball 2.jpeg")

#blur output image
#img = cv2.medianBlur(gray_output ,5)

#https://www.pyimagesearch.com/2014/08/04/opencv-python-color-detection/

# https://www.rapidtables.com/web/color/RGB_Color.html

# define the boundaries
lower_range = np.array([0, 100, 100], dtype="uint8")
upper_range = np.array([200, 255, 255], dtype="uint8")

#read in image file
src = cv2.imread('output.png')

# create mask and apply
mask = cv2.inRange(img, lower_range, upper_range)
output = cv2.bitwise_and(img, img, mask=mask)

#turn output/tennis ball gray
gray_output = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

#blur output image
gray_output = cv2.medianBlur(gray_output, 5)

#https://docs.opencv.org/master/da/d53/tutorial_py_houghcircles.html
#https://www.youtube.com/watch?v=dp1r9oT_h9k
#parametres of circle
circles = cv2.HoughCircles(gray_output,cv2.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=0,maxRadius=0)

#turn x and y coordinates into intergers
detected_circles = np.uint16(np.around(circles))
for (x, y, r) in detected_circles[0, :]:
    cv2.circle(gray_output, (x, y), r, (100, 100, 120), 3)
    cv2.circle(gray_output, (x, y), 2, (0, 0, 255), 3)

#print coordinates of centre and radius
print (x, y, r)

# color output img file
cv2.imwrite("c_output.png", img)

# output file
cv2.imwrite("output.png", gray_output)