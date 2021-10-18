import cv2
import dlib
import numpy as np
import os

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor("/Users/kaustubh/Desktop/ML Bootcamp/datasets/shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)

mood = input("Enter your mood - ")

frames = []
outputs = []

while (True):

   ret, frame = cap.read()
   
   gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

   faces = detector(gray)

   for face in faces:
        landmarks = predictor(gray, face)
        #print(landmarks.parts)
        nose = landmarks.parts()[27]
        #print(nose.x, nose.y)

        #lip_up = landmarks.parts()[63].y
        #lip_down = landmarks.parts()[67].y

        #if lip_down - lip_up > 5:
            #print("Mouth is open")
        #else:
            #print("Mouth is close")

        expression = np.array([[point.x - face.left(), point.y - face.top()] for point in landmarks.parts()[17:]])
        #print(expression.flatten())

        #for point in landmarks.parts()[48:]:
          #cv2.circle(frame, (point.x, point.y), 2, (200,0,0), 3)
   #print(faces)

   if ret: # the ret value has to be true if we want to reflect something

       cv2.imshow("My Screen", frame)
       

   key = cv2.waitKey(1)

   if key == ord("q"):
        break
   elif key == ord("c"):
       frames.append(expression.flatten())
       outputs.append([mood])

X = np.array(frames)
y = np.array(outputs)

data = np.hstack([y, X]) # horizontal stacking

f_name = "face_mood.npy"

if os.path.exists(f_name):
    old = np.load(f_name)
    data = np.vstack([old, data])

np.save(f_name, data)
    
cap.release()
cv2.destroyAllWindows()


