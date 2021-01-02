#Used Tutorial: https://www.youtube.com/watch?v=PmZ29Vta7Vc&t=3255s&ab_channel=CodingEntrepreneurs

import numpy as np
import cv2 # pip install opencv-python
import pickle

face_cascade = cv2.CascadeClassifier('../haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('../haarcascades/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('../haarcascades/haarcascade_smile.xml')

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in faces:
        #print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w] #(ycord_start, ycord_end)
        roi_color = frame[y:y+h, x:x+w]
        img_item = "my-image.png"
        cv2.imwrite(img_item, roi_gray) # save my face as png
    	
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'): # press q to quit
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
