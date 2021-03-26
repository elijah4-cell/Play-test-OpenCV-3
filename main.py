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


  #https://www.pyimagesearch.com/2014/08/04/opencv-python-color-detection/
  # https://www.rapidtables.com/web/color/RGB_Color.html
  # define the boundaries
  lower_range = np.array([0, 100, 100], dtype="uint8")
  upper_range = np.array([200, 255, 255], dtype="uint8")

  # create mask and apply
  mask = cv2.inRange(B_img, lower_range, upper_range)
  output = cv2.bitwise_and(B_img, img, mask=mask)

  #read in image file
  src = cv2.imread(path)

  #turn output/tennis ball gray
  gray_output = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

	# blur image to reduce noise and avoid false circle detection
	#gray_output = cv2.medianBlur(gray_output, 5)

  #https://docs.opencv.org/master/da/d53/tutorial_py_houghcircles.html
  #https://www.youtube.com/watch?v=dp1r9oT_h9k
  #parametres of circle
  circles = cv2.HoughCircles(gray_output,cv2.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=0,maxRadius=0)
  #print (circles)

  #turn x and y coordinates into intergers
  detected_circles = np.uint8(np.around(circles))
  #print (detected_circles)
  for (x, y, r) in detected_circles[0, :]:
      cv2.circle(B_img, (x, y), r, (100, 100, 120), 3)
      cv2.circle(B_img, (x, y), 2, (0, 0, 255), 3)

  #print coordinates of centre and radius
  #print (x, y, r)

  # color output img file
  cv2.imwrite("Tennis ball outputs/c_output.png" + file, B_img)

  # output file
  cv2.imwrite("Tennis ball outputs/output.png" + file, gray_output)