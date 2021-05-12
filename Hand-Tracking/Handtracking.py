import mediapipe as mp
import cv2
import time

cap = cv2.VideoCapture(0)
mphands = mp.solutions.hands
hands = mphands.Hands()
mpdraw = mp.solutions.drawing_utils
ptime = 0
ctime = 0

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(imgrgb)
    
    if result.multi_hand_landmarks:
        for handlms in result.multi_hand_landmarks:
            mpdraw.draw_landmarks(img,handlms,mphands.HAND_CONNECTIONS)
            for id,lm in enumerate(handlms.landmark):             
                h , w , c = img.shape
                cx,cy = int(lm.x*w) , int(lm.y*h)
                cv2.circle(img, (cx,cy) , 6,(20,100,200), cv2.FILLED)      

    # To get the FPS
    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime
    cv2.putText(img,str(int(fps)),(10,60), cv2.FONT_HERSHEY_PLAIN, 3,(20,100,200),2)
    
    cv2.imshow('frame',img)
    if cv2.waitKey(1) == ord('z'): # Press 'z' to close the window
        break 

cap.release()
cv2.destroyAllWindows()