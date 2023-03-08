import cv2
import datetime

cap = cv2.VideoCapture(0)

# Load recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('Models/cv2_recognizer.yml')

# Haar Cascade
faceCascade = cv2.CascadeClassifier("Cascades/haarcascade_frontalface_default.xml")
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
        # flags = cv2.CV_HAAR_SCALE_IMAGE
    )

    # print("Found {0} faces!".format(len(faces)))

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

        # Check if confidence is less them 100 ==> "0" is perfect match
        if (confidence < 100):
            id = "SE" + str(id)
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
        cv2.putText(frame, str(id), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, str(confidence), (x + 5, y + h - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    elif cv2.waitKey(1) % 256 == 32:
        # SPACE pressed
        img_name = f"""{datetime.datetime.utcnow().strftime('%d-%m-%Y-%H-%M-%S')}.png"""
        print(img_name)
        cv2.imwrite(img_name, frame)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
