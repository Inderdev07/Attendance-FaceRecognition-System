import cv2
import numpy as np
import requests
import time

attendance = []
start = time.time()
period = 8
face_cas = cv2.CascadeClassifier('C:/Users/dogra/OneDrive/Desktop/Attandence system/Automatic_attendence_system/Automatic_attendence_system_using_facial_recognition_python_openCV-main/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(1)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('C:/Users/dogra/OneDrive/Desktop/Attandence system/trainer.yml/trainer.yml.txt')
font = cv2.FONT_HERSHEY_SIMPLEX
url = 'https://kcattendence.000webhostapp.com/update.php'

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cas.detectMultiScale(gray, 1.3, 7)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        id, conf = recognizer.predict(roi_gray)
        print('going inside conf<50')
        
        name = "Unknown"  # Default value
        gender = "Unknown"  # Default value
        rollno = "Unknown"  # Default value
        photo = "Unknown"  # Default value
        
        if conf < 50:
            print(' inside conf<50')
            
            
            # Assign the corresponding data based on the detected ID
            if id == 1:
                name = 'Vishal'
                gender = 'Male'
                rollno = '101'
                photo = 'vishal.jpg'
                cap.release()

            elif id == 2:
                name = 'Inder'
                gender = 'Male'
                rollno = '102'
                photo = 'inder.jpg'
                cap.release()
            elif id == 3:
                name = 'Vikas'
                gender = 'Male'
                rollno = '103'
                photo = ''
                cap.release()        
                
            elif id == 4:
                name = 'Divya'
                gender = 'Female'
                rollno = '104'
                photo = ''
                cap.release()
            elif id == 5:
                name = 'Sarwan Singh'
                gender = 'Male'
                rollno = '104'
                photo = 'sarwan singh.jpeg'
                cap.release()
            elif id == 6:
                name = 'Meha'
                gender = 'Female'
                rollno = '107'
                photo = ''
                cap.release()

            # Send the data to the PHP script
            data = {
                'name': name,
                'gender': gender,
                'rollno': rollno,
                'photo': photo
            }
            response = requests.post(url, data=data)
            print(name, 'data sent to the server')

        else:
            print('Unknown,face not detected')

        cv2.putText(img, str(id) + " " + str(name), (x, y - 10), font, 0.55, (120, 255, 120), 1)

    cv2.imshow('frame', img)

    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()
