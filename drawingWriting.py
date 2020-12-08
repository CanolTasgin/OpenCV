#Drawing and writing on Image - OpenCV / Applied minimal changes on top of the code. #circle with eyes and mouth, arrowedLine, changes on all shapes
#Followed Tutorial: https://www.youtube.com/watch?v=U6uIrq2eh_o
import numpy as np
import cv2

img = cv2.imread('car.jpg',cv2.IMREAD_COLOR)

cv2.line(img,(0,200),(300,0),(255,255,255),15) #(image, lineStart, lineEnd, lineColor, thickness) / openCV uses bgr instead of rgb

cv2.rectangle(img,(1000,600),(1250,850),(200,0,100),10)

cv2.circle(img,(1050,320), 70, (50,150,0), -1) #(img, centerOfCircle, radius, Color, fillIt if -1) 

smile = np.array([[1040,340],[1060,350],[1070,350],[1090,340]], np.int32)
smile = smile.reshape((-1,1,2)) #(= rows * cols * numChannels)
cv2.polylines(img, [smile], False, (255,255,255), 3) #(image, coordinates, circled or not, color, thickness)

#Eyes
cv2.circle(img, (1050,310), 5, (255,255,255), -1)
cv2.circle(img, (1080,310), 5, (255,255,255), -1)

pts = np.array([[600,700],[540,600],[750,630],[550,860],[500,800]], np.int32)
pts = pts.reshape((-1,1,2))
cv2.polylines(img, [pts], True, (0,0,255), 3) 

font = cv2.FONT_HERSHEY_SIMPLEX

cv2.putText(img,'Bugatti',(50,550), font, 4, (50,50,155), 10, cv2.LINE_8) #(image, text, coordinates, font, fontScale, Color, thickness, lineType)

cv2.arrowedLine(img, (1500,200), (1250,300), (155,0,0), 15) 

cv2.imshow('image',img)

cv2.waitKey(0)
cv2.destroyAllWindows()