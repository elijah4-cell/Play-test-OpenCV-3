import cv2
import numpy as np

img = cv2.imread("tennis ball 2.jpeg")


# https://www.pyimagesearch.com/2014/08/04/opencv-python-color-detection/


# define the list of boundaries
boundaries = [
	#([17, 15, 100], [50, 56, 200]), # red: lower: g, b, r 
	#([86, 31, 4], [220, 88, 50]), # blue
	#([25, 146, 190], [62, 174, 250])#, # yellow
  ([0, 100, 100], [255, 255, 255])#, # yellow
	#([103, 86, 65], [145, 133, 128]) # grey
]

# loop over the boundaries
for (lower, upper) in boundaries:
	# create NumPy arrays from the boundaries
	lower = np.array(lower, dtype = "uint8")
	upper = np.array(upper, dtype = "uint8")
	# find the colors within the specified boundaries and apply
	# the mask
	mask = cv2.inRange(img, lower, upper)
	output = cv2.bitwise_and(img, img, mask = mask)



# print(img)

# lower_range = np.array([25, 146, 190], dtype = "uint8")
# upper_range = np.array([62, 174, 250], dtype = "uint8")
# lower_range = np.array([0, 255, 255], dtype = "uint8")
# upper_range = np.array([255, 255, 255], dtype = "uint8")

# #([25, 146, 190], [62, 174, 250]),

# mask = cv2.inRange(img, lower_range, upper_range)
# output = cv2.bitwise_and(img, img, mask = mask)

cv2.imwrite("output.png", output) 
#v2.imwrite("output.png", img) 

#cv2.imshow('image',img)
#cv2.imshow('mask', mask)

#print("Shape: {}".format(img.shape))