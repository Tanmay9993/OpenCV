import cv2
import numpy as np 

class FaceDetector():
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
        self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

    def findFace(self,img, draw = True):
        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 7)
        if draw:
            for (x, y, w, h) in faces: 
                cv2.rectangle(img, (x, y), (x + w +5, y + h +20), (255, 0, 0), 4)

                roi_gray = gray[y:y+w, x:x+w]  
                roi_color = img[y:y+h, x:x+h]
                eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.25, 5)  
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 4)

                soi_gray = gray[y:y+w, x:x+w]  
                soi_color = img[y:y+h, x:x+h]
                smile = self.smile_cascade.detectMultiScale(soi_gray, 1.25, 7) 
                for (sx, sy, sw, sh) in smile:
                    cv2.rectangle(soi_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 3)

        return img


def main():
    cap = cv2.VideoCapture(0)
    detector = FaceDetector()
    while True:
        ret, frame = cap.read()
        frame = detector.findFace(frame)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('z'): # Press 'z' to close the window
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()