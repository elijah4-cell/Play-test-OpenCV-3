import cv2
import numpy as np
import os


def direction():

    dir = 0
    #https://appdividend.com/2020/09/09/python-cv2-image-size-how-to-get-image-size-in-python/#:~:text=To%20get%20the%20proper%20size,with%20OpenCV%2C%20use%20the%20ndarray.
    #gives hight, width, channel of the img
    h, w, c = img.shape

    #print width of img
    print('width:', w)

    #find centre of img by dividing by two
    centre_img = w // 2

    #a is range
    #centre ranges from -40 to 40
    a = 40

    #img by dividing by two - a
    #right param
    Rcentre_img = w // 2 - a

    #find centre of img by dividing by two + a
    #left param
    Lcentre_img = w // 2 + a

    #print centre of img
    print('centre of img:', centre_img)

    if Lcentre_img < x:
        print('x-value of TB:', x)
        print('range centre of img:', Lcentre_img)

        #direction
        print('Right')
        dir = -1

    elif Rcentre_img > x:
        print('x-value of TB:', x)
        print('range centre of img:', Rcentre_img)
        print('Left')
        dir = 1

    elif centre_img == x or Rcentre_img < x or Lcentre_img > x:
        print('x-value of TB:', x)
        print('centre of img:', centre_img)
        print('range Rcentre of img:', Rcentre_img)
        print('range Lcentre of img:', Lcentre_img)
        print('centre')
        dir = 0

    else:
        print('unknown direction')

    print('')

    return dir


#https://www.tutorialspoint.com/python/os_listdir.htm
# open file
folder = "tennis balls"
dir = os.listdir(folder)
#print (dir)

# print files

for file in dir:

    path = folder + "/" + file
    print(path)

    #print (file)

    # read in image file
    img = cv2.imread(path)

    #convert BGR to HSV
    HSV_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #output HSV_img file
    cv2.imwrite("Tennis ball colour outputs/HSV" + file, HSV_img)

    #https://www.pyimagesearch.com/2014/08/04/opencv-python-color-detection/
    # https://www.rapidtables.com/web/color/RGB_Color.html
    # define the boundaries
    lower_range = np.array([0, 60, 100], dtype="uint8")
    upper_range = np.array([200, 255, 255], dtype="uint8")

    # create mask and apply
    mask = cv2.inRange(HSV_img, lower_range, upper_range)

    #output mask file
    cv2.imwrite("Tennis ball colour outputs/mask" + file, mask)

    #
    output = cv2.bitwise_and(HSV_img, HSV_img, mask=mask)

    #convert HSV back to BGR
    BGR_output = cv2.cvtColor(output, cv2.COLOR_HSV2BGR)

    #output HSV2BGR file
    cv2.imwrite("Tennis ball colour outputs/BGR" + file, BGR_output)

    #convert BGR to gray for HoughCircles function
    G_output = cv2.cvtColor(BGR_output, cv2.COLOR_BGR2GRAY)

    #https://www.youtube.com/watch?v=8CMTqpZoec8
    #blur g_output
    B_output = cv2.medianBlur(G_output, 5)

    #output Gray2BGR file
    cv2.imwrite("Tennis ball colour outputs/Blur" + file, B_output)

    #convert gray to BGR
    C_output = cv2.cvtColor(B_output, cv2.COLOR_GRAY2BGR)

    #output Gray2BGR file
    cv2.imwrite("Tennis ball colour outputs/Colour" + file, C_output)

    #https://docs.opencv.org/master/da/d53/tutorial_py_houghcircles.html
    #https://www.youtube.com/watch?v=dp1r9oT_h9k
    #parametres of circle
    circles = cv2.HoughCircles(B_output,
                               cv2.HOUGH_GRADIENT,
                               1,
                               200,
                               param1=75,
                               param2=30,
                               minRadius=15,
                               maxRadius=165)

    #second
    #  circles = cv2.HoughCircles(B_output, cv2.HOUGH_GRADIENT, 1, 200, param1 = 110,param2 = 30, minRadius = 20, maxRadius = 200)

    #first
    #cv2.HoughCircles(B_img,cv2.HOUGH_GRADIENT,1,10,param1=50,param2=30,minRadius=50,maxRadius=0)
    #print (circles)
    try:
        #turn x and y coordinates into intergers
        detected_circles = np.uint16(np.around(circles))
    except:
        print("none")
    #print (detected_circles)
    for (x, y, r) in detected_circles[0, :]:
        #draw outer circle
        cv2.circle(C_output, (x, y), r, (0, 255, 0), 3)
        #draw inner circle
        cv2.circle(C_output, (x, y), 2, (0, 0, 255), 3)

    #print coordinates of centre and radius
    print(x, y, r)

    #gives direction of tennis ball 🎾
    direction()

    # color output C_output file
    cv2.imwrite("Tennis ball outputs/c_output.png" + file, C_output)

    # output G_output file
    cv2.imwrite("Tennis ball outputs/output.png" + file, C_output)
