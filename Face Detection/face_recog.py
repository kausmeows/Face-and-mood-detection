import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

data = np.load("face_data.npy")
print(data.shape, data.dtype)

X = data[:, 1:].astype(np.uint8)
y = data[:, 0]

model = KNeighborsClassifier()
model.fit(X, y)






cap = cv2.VideoCapture(0)

detector = cv2.CascadeClassifier("/Users/kaustubh/Desktop/ML Bootcamp/datasets/haarcascade_frontalface_default.xml")


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

           out = model.predict([gray.flatten()])

           cv2.rectangle(frame, (x, y), (x+w, y+w), (150, 100, 142), 2)
           cv2.putText(frame, str(out[0]), (x, y - 15), cv2.FONT_HERSHEY_COMPLEX, 2, (200, 120, 30), 2)

           print(out)


           cv2.imshow("My face", gray) # if I come close or go far the face size remains same.

           
       cv2.imshow("My Screen", frame)
       
   key = cv2.waitKey(1)

   if key == ord("q"):
       break
  





cap.release()
cv2.destroyAllWindows() 


