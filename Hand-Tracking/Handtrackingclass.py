import mediapipe as mp
import cv2
import time

class HandDetector():

    def __init__(self,mode = False,max_hands = 2,detection_conf = 0.5,tracking_conf = 0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_conf  = detection_conf
        self.tracking_conf = tracking_conf
        self.mphands = mp.solutions.hands
        self.hands = self.mphands.Hands(self.mode,self.max_hands,self.detection_conf,self.tracking_conf)
        self.mpdraw = mp.solutions.drawing_utils


    def findHands(self,img, draw = True):
        img = cv2.flip(img, 1)
        imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result = self.hands.process(imgrgb)
        if self.result.multi_hand_landmarks:
            for handlms in self.result.multi_hand_landmarks:
                if draw:
                    self.mpdraw.draw_landmarks(img,handlms,self.mphands.HAND_CONNECTIONS)
        return img


    def findPosi(self, img, handNo = 0,draw = True):
        if self.result.multi_hand_landmarks:
            for myHand in self.result.multi_hand_landmarks:
                for id,lm in enumerate(myHand.landmark):
                    h , w , c = img.shape
                    cx,cy = int(lm.x*w) , int(lm.y*h)
                    if draw:
                        cv2.circle(img, (cx,cy) , 4,(20,100,220), cv2.FILLED)
        return img


             
def main():
    cap = cv2.VideoCapture(0) 
    ptime = 0
    ctime = 0
    detector = HandDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        img = detector.findPosi(img)
        # To Calculate FPS
        ctime = time.time()
        fps = 1/(ctime-ptime)
        ptime = ctime
        cv2.putText(img,str(int(fps)),(10,60), cv2.FONT_HERSHEY_PLAIN, 2,(250,150,0),2)

        cv2.imshow('frame',img)
        if cv2.waitKey(1) == ord('z'): # Press 'z' to close the window
            break 
    cap.release()
    cv2.destroyAllWindows() 


if __name__ == "__main__":
    main()
