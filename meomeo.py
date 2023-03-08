# This script will detect faces via your webcam.
# Tested with OpenCV3

import cv2
from tracker import *


# this is just to unconfuse pycharm
def comppppp(img1, img2, req=10):
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    # find the key-points and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good.append(m)
    if len(good) >= req:
        return True
    return False


cap = cv2.VideoCapture(0)
ta = cv2.imread("Thong.jpg")
tracker = EuclideanDistTracker()
# Create the haar cascade
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
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

    print("Found {0} faces!".format(len(faces)))
    boxes_ids = tracker.update(faces)

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (x, y)
        fontScale = 2
        fontColor = (255, 255, 255)
        thickness = 1
        lineType = 2
        if comppppp(frame[y:y + h, x:x + w], ta):
            cv2.putText(frame, 'Thong',
                        bottomLeftCornerOfText,
                        font,
                        fontScale,
                        fontColor,
                        thickness,
                        lineType)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # for box_id in boxes_ids:
    #     x, y, w, h, id = box_id
    #     cv2.putText(frame, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
