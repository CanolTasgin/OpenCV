#Used Tutorial at the beginning: https://www.youtube.com/watch?v=88HdqNDQsEk&list=PLQVvvaa0QuDdttJXlLtAJxJetJcqmqlQq&index=16 
#OpenCV docs: https://docs.opencv.org/4.5.0/db/d28/tutorial_cascade_classifier.html

#In order to use it on Mac/Visual Studio Code, Command+Shift+P --> Choose ">Shell Command: Install 'code' command in PATH" --> open vscode via Terminal with command 'sudo code' in order to give VSCode Camera permissions

import numpy as np
import cv2
import time

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

#haarcascade xml files are under Intel License Agreement. (Detailed information is in the xml files which you can find in the below link)
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')

smile_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_smile.xml')

cap = cv2.VideoCapture(0) #Capture video from camera id 0 (main camera)

while 1:
    ret, img1 = cap.read() 
    img = cv2.flip(img1,1) #flipped video feed in order to make it like mirror
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #make it gray
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) #(image, scaleFactor, minNeighbours)
                                            #scaleFactor: Parameter specifying how much the image size is reduced at each image scale.
                                            #minNeighbours: Parameter specifying how many neighbors each candidate rectangle should have to retain it. 
                                                    #This parameter will affect the quality of the detected faces. Higher value results in less detections but with higher quality. 3~6 is a good value for it.
                                #More Detailed Explanations: 
                                # https://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_Image_Object_Detection_Face_Detection_Haar_Cascade_Classifiers.php
                                # https://stackoverflow.com/questions/20801015/recommended-values-for-opencv-detectmultiscale-parameters/20805153

    faceAmount = 0
    eyeAmount = 0
    smileAmount = 0
    for (x,y,w,h) in faces:
        faceAmount += 1 #track number of faces appeared on the screen
        cv2.rectangle(img,(x,y-70),(x+w,y+h),(255,0,0),2) #(image, start_point, end_point, color, thickness) 
                                                        #Detection mostly doesn't include chin so strecthed it from bottom
        font = cv2.FONT_HERSHEY_SIMPLEX
        #faceInfo = "Face" + "\n" + "Face Width: " + str(w) #new line symbol deosnt work correctly
        faceWidth = "Face Width: " + str(w)
        faceLength = "Face Length: " + str(h+70) 
        cv2.putText(img,"Face",(x+w+10, y+h-60), font, 0.8, (0,255,0), 1, cv2.LINE_AA) #(image, text, startCoords, font, fontSize, Color, Thickness, lineType)
        cv2.putText(img,faceWidth,(x+w+10, y+h-30), font, 0.8, (0,255,0), 1, cv2.LINE_AA)
        cv2.putText(img,faceLength,(x+w+10, y+h), font, 0.8, (0,255,0), 1, cv2.LINE_AA)

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        if faceAmount > 0:
            smile = smile_cascade.detectMultiScale(roi_gray, 1.8, 20) #Detecs smile (it also detecs mouth)
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.6, 10)
            
            for (ex,ey,ew,eh) in smile:
                smileAmount += 1
                if smileAmount <= faceAmount: #same amount of face and smiles can appear
                    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),2) 

            for (ex,ey,ew,eh) in eyes:
                eyeAmount += 1
                if eyeAmount <= 2 * faceAmount: #limit max 2 eyes per face
                    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2) 

    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff # & 0xff --> leaves only the last 8 bits. 
                            #Representations may change when NumLock is activated,
                            #for example for pressing key 'c' --> NumLock :1048675(100000000000001100011)  Otherwise: 99(1100011)
                            #https://stackoverflow.com/questions/35372700/whats-0xff-for-in-cv2-waitkey1
    if k == 27: # If Esc pressed, stop.
        break

cap.release()
cv2.destroyAllWindows()