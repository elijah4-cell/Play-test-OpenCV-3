import cv2
import numpy as np

img = cv2.imread("tennis ball 2.jpeg", 0)
edges = cv2.Canny(img, 100, 100)
