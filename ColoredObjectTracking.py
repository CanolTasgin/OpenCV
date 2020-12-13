# Followed tutorial on the following URL for this program. 
# https://docs.opencv.org/4.5.0/d2/d96/tutorial_py_table_of_contents_imgproc.html

#In order to use it on Mac/Visual Studio Code, Command+Shift+P --> Choose ">Shell Command: Install 'code' command in PATH" --> open vscode via Terminal with command 'sudo code' in order to give VSCode Camera permissions

#Color Conversion: cv.cvtColor(input_image, flag) //flag = type of conversion
# BGR --> Gray : cv.COLOR_BGR2GRAY
# BGR --> HSV : cv.COLOR_BGR2HSV

#Show all flags:
import cv2 as cv
import numpy as np

flags = [i for i in dir(cv) if i.startswith('COLOR_')]
print( flags )

#Extract Blue colored object from Webcam

cap = cv.VideoCapture(0)
while(1):
    # Take each frame
    _, frame = cap.read()
    # Convert BGR to HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # define range of blue color in HSV
    # lower_blue = np.array([110,50,50])
    # upper_blue = np.array([130,255,255])

    # define range of yellow in HSV
    yellow = np.uint8([[[0,255,255 ]]]) #yellow in BGR
    hsv_yellow = cv.cvtColor(yellow,cv.COLOR_BGR2HSV) #Convert it to HSV and see hsv version
    #print( hsv_yellow ) # 30 255 255

    lowerLimit = np.array([15,80,80]) #H stands for hue. Possible ranges are like following in OpenCV --> H (0-180) S (0-255) V (0-255) 
    upperLimit = np.array([45,255,255]) #Change H for defining color range

    # Threshold the HSV image to get only yellow colors
    mask = cv.inRange(hsv, lowerLimit, upperLimit)

    # Bitwise-AND mask and original image
    res = cv.bitwise_and(frame,frame, mask= mask)
    cv.imshow('frame',frame)
    cv.imshow('mask',mask)
    cv.imshow('res',res)
    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break
cv.destroyAllWindows()