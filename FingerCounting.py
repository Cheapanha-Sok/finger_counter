import cv2
import time
import HandTrackingModule as htm

# Camera setup for width and height
wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.HandDetector(detectionCon=0.75, maxHands=1)
tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    
    # Create a list of the landmarks detected
    lmList = detector.findPosition(img)
    
    if len(lmList) != 0:
        fingers = []
        
        # Determine if the hand is left or right
        if lmList[tipIds[1]][1] < lmList[0][1]:
            # Left hand
            if lmList[tipIds[0]][1] < lmList[tipIds[0]-1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
        else:
            # Right hand
            if lmList[tipIds[0]][1] > lmList[tipIds[0]-1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
        
        # 4 Fingers (index to pinky)
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)q
                
        totalFingers = fingers.count(1)
        
        # Draw the rectangle on the top left corner
        # Display the result on the video frame
        cv2.rectangle(img, (10, 10), (100, 70), (0, 0, 255), cv2.FILLED)
        cv2.putText(img, str(totalFingers), (34, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)

    cv2.imshow('finger counter', img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
