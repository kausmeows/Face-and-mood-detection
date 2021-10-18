import cv2

cap = cv2.VideoCapture(0)

detector = cv2.CascadeClassifier("/Users/kaustubh/Desktop/ML Bootcamp/datasets/haarcascade_frontalface_default.xml")

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
    
    
cap.release()
cv2.destroyAllWindows()


