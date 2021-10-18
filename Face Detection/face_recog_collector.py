import cv2
import numpy as np
import os

cap = cv2.VideoCapture(0)

detector = cv2.CascadeClassifier("/Users/kaustubh/Desktop/ML Bootcamp/datasets/haarcascade_frontalface_default.xml")
name = input("enter your name - ")

frames = []
outputs = []

while (True):
   ret, frame = cap.read()
   if ret: # the ret value has to be true if we want to reflect something
       faces = detector.detectMultiScale(frame)
       
       for face in faces:
           x, y, w, h = face

           cut = frame[y:y+h, x:x+w]

           fix = cv2.resize(cut, (100, 100))
           gray = cv2.cvtColor(fix, cv2.COLOR_BGR2GRAY)
           cv2.imshow("My face", gray) # if I come close or go far the face size remains same.

           
       cv2.imshow("My Screen", frame)
       
   key = cv2.waitKey(1)

   if key == ord("q"):
       break
   if key == ord("c"):
       #cv2.imwrite(name + ".jpg", frame)
       frames.append(gray.flatten())
       outputs.append([name])
    


X = np.array(frames)
y = np.array(outputs)

data = np.hstack([y, X]) # horizontal stacking
print(data.shape)

f_name = "face_data.npy"

if os.path.exists(f_name):
    old = np.load(f_name)
    data = np.vstack([old, data])

np.save(f_name, data)


cap.release()
cv2.destroyAllWindows()


