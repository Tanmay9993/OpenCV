import numpy as np
import cv2 

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    faces = face_cascade.detectMultiScale(gray, 1.1, 7)  

    for (x, y, w, h) in faces: 
        cv2.rectangle(frame, (x, y), (x + w +5, y + h +20), (255, 0, 0), 4)

        roi_gray = gray[y:y+w, x:x+w]  
        roi_color = frame[y:y+h, x:x+h]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.25, 5) 
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 4)

        soi_gray = gray[y:y+w, x:x+w]  
        soi_color = frame[y:y+h, x:x+h]
        smile = smile_cascade.detectMultiScale(soi_gray, 1.25, 7)  
        for (sx, sy, sw, sh) in smile:
            cv2.rectangle(soi_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 3)

    cv2.imshow('frame', frame)
    
    if cv2.waitKey(1) == ord('z'): # Press 'z' to close the window
        break

cap.release()
cv2.destroyAllWindows()
