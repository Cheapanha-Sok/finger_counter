import cv2
import time
import os
import HandTrackingModule as htm

# Camera setup for width and height
wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# Initialize hand detector class
detector = htm.HandDetector(detectionCon=0.75 , maxHands=1)

while True:
    success, frame = cap.read()

    if not success:
        print('Error reading frame')
        break

    frame = detector.findHands(frame)
    lmList = detector.findPosition(frame , False)

    # Debug print to check landmark positions
    if lmList:
        print(lmList)

    # Show the processed image
    cv2.imshow('finger counter', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
