import cv2
import os

import numpy as np

datasets = "Datasets"
student_id = os.listdir("%s" % datasets)

# Check students IDs
print("Student IDs: ", end="")
print(*student_id, sep=", ")

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("Cascades/haarcascade_frontalface_default.xml");

extracted_Faces = []
ids = []
for student in student_id:
    path = f"{datasets}/{student}"
    numbers_of_pic = len(os.listdir(path))
    for i in range(1, numbers_of_pic + 1):
        # Read image
        frame = cv2.imread(path + "/" + f"{i}.jpg")
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Grayscale (Turn to uint8)
        # Detect faces in the image
        try:
            faces = detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
                # flags = cv2.CV_HAAR_SCALE_IMAGE
            )

            # Even though there is 1 face, Classifier return a list of point, hence while the use of for loop
            for (x, y, w, h) in faces:
                extracted_Faces.append(gray[y:y + h, x:x + w])
                ids.append(int(student[2:]))
        except cv2.error:
            print(student, i)

# print(ids)

# # Train recognizer
recognizer.train(extracted_Faces, np.array(ids))
recognizer.write('Models/cv2_recognizer.yml')
