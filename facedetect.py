import cv2

webcam = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier("haarface.xml")
eye_cascade = cv2.CascadeClassifier("haareye.xml")

while True:

    success, img = webcam.read()

    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(imgGray,1.1,4)
    
    for x,y,w,h in faces:

        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        face_frame = imgGray[y:y+h, x:x+w]
    
        eyes = eye_cascade.detectMultiScale(imgGray)
        for ex,ey,ew,eh in eyes:
            cv2.rectangle(img,(ex,ey),(ex+ew,ey+eh),(255,2,0),2)
    
    cv2.imshow("Face Detection",img)
   
    k = cv2.waitKey(30) & 0xff
    if k ==27:
        break

    