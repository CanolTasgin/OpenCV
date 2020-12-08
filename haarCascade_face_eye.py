#Used Tutorial: https://www.youtube.com/watch?v=88HdqNDQsEk&list=PLQVvvaa0QuDdttJXlLtAJxJetJcqmqlQq&index=16 / Also implemented mouth/smile detection on top of it
#OpenCV docs: https://docs.opencv.org/4.5.0/db/d28/tutorial_cascade_classifier.html

#In order to use it on Mac/Visual Studio Code, open vscode via Terminal with command 'sudo code' in order to give VSCode Camera permissions

import numpy as np
import cv2

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

#haarcascade xml files are under Intel License Agreement. (Detailed information is in the xml files which you can find in the below link)
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')

smile_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_smile.xml')

cap = cv2.VideoCapture(0) #Capture video from camera id 0 (main camera)

while 1:
    ret, img = cap.read() 
    cv2.flip(img,1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #make it gray
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) #(image, scaleFactor, minNeighbours)
                                            #scaleFactor: Parameter specifying how much the image size is reduced at each image scale.
                                            #minNeighbours: Parameter specifying how many neighbors each candidate rectangle should have to retain it. 
                                                    #This parameter will affect the quality of the detected faces. Higher value results in less detections but with higher quality. 3~6 is a good value for it.
                                #More Detailed Explanations: 
                                # https://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_Image_Object_Detection_Face_Detection_Haar_Cascade_Classifiers.php
                                # https://stackoverflow.com/questions/20801015/recommended-values-for-opencv-detectmultiscale-parameters/20805153

    #img_flip = cv2.flip(img,1)
    #combined_window = np.hstack([gray_flip])

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) #(image, start_point, end_point, color, thickness)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        smile = smile_cascade.detectMultiScale(roi_gray, 1.6, 15) #Detecs smile (it also detecs mouth)
        for (ex,ey,ew,eh) in smile:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),2) 

        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 12)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2) 

    img_flip = cv2.flip(img,1) #flipped video feed in order to make it like mirror
    cv2.imshow('img',img_flip)
    k = cv2.waitKey(30) & 0xff # & 0xff --> leaves only the last 8 bits. 
                            #Representations may change when NumLock is activated,
                            #for example for pressing key 'c' --> NumLock :1048675(100000000000001100011)  Otherwise: 99(1100011)
                            #https://stackoverflow.com/questions/35372700/whats-0xff-for-in-cv2-waitkey1
    if k == 27: # If Esc pressed, stop.
        break

cap.release()
cv2.destroyAllWindows()