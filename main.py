import cv2
import numpy as np
import os

#https://www.tutorialspoint.com/python/os_listdir.htm
# open file
folder = "tennis balls"
dir = os.listdir(folder)
print (dir)

# print files


for file in dir:

  path = folder + "/" + file
  print(path)

#print (file)


  # read in image file
  img = cv2.imread(path) 

  Y_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

  # blur image to reduce noise and avoid false circle detection
  #src = cv2.imread(path) 
  #B_img = cv2.blur(src,(5,5))

  #https://www.pyimagesearch.com/2014/08/04/opencv-python-color-detection/
  # https://www.rapidtables.com/web/color/RGB_Color.html
  # define the boundaries
  lower_range = np.array([100, 45, 60], dtype="uint8")
  upper_range = np.array([40, 100, 60], dtype="uint8")

  # create mask and apply
  mask = cv2.inRange(Y_img, lower_range, upper_range)
  output = cv2.bitwise_and(Y_img, Y_img, mask=mask)

  cv2.imwrite("Tennis ball outputs/output.png" + file, output)

folder = "Tennis ball outputs"
dirs = os.listdir(folder)
print (dirs)

for file in dirs:
  path = folder + "/" + file
  print(path)

  src = cv2.imread(path)

  C_output = cv2.cvtColor(src, cv2.COLOR_HSV2BGR)
  G_output = cv2.cvtColor(C_output, cv2.COLOR_BGR2GRAY)


  #https://www.youtube.com/watch?v=8CMTqpZoec8
  #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  #blur gray img
  #B_img = cv2.medianBlur(gray, 5)

  #change blur gray img to colour
  #C_img = cv2.cvtColor(B_img, cv2.COLOR_GRAY2BGR)

  #read in image file
  #src = cv2.imread(path)

  #turn output/tennis ball gray
  #gray_output = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

  # blur image to reduce noise and avoid false circle detection
  #gray_output = cv2.medianBlur(gray_output, 5)

  #https://docs.opencv.org/master/da/d53/tutorial_py_houghcircles.html
  #https://www.youtube.com/watch?v=dp1r9oT_h9k
  #parametres of circle
  #circles = cv2.HoughCircles(G_output, 
  #                cv2.HOUGH_GRADIENT, 1, 200, param1 = 110,
  #            param2 = 30, minRadius = 20, maxRadius = 200)

  #cv2.HoughCircles(B_img,cv2.HOUGH_GRADIENT,1,10,param1=50,param2=30,minRadius=50,maxRadius=0)
  #print (circles)

  #turn x and y coordinates into intergers
  #detected_circles = np.uint16(np.around(circles))
  #print (detected_circles)
  #for (x, y, r) in detected_circles[0, :]:
      #draw outer circle
  #    cv2.circle(G_output, (x, y), r, (0, 255, 0), 3)
      #draw inner circle
  #    cv2.circle(G_output, (x, y), 2, (0, 0, 255), 3)

  #print coordinates of centre and radius
  #print (x, y, r)

  # color output img file
  cv2.imwrite("Tennis ball outputs/c_output.png" + file, C_output)

  # output file
  #cv2.imwrite("Tennis ball outputs/output.png" + file, gray_output)

