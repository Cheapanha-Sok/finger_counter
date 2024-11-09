import cv2
import time
import os
import HandTrackingModule as htm

# Camera setup for width and height
wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# use os to store the folder FingerImages
folderPath = "FingerImages"
myList = os.listdir(folderPath)
# print(myList)
pTime = 0
overlayList = []

for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')  # Load each image
    resized_image = cv2.resize(image, (200, 200))  # Resize to 200x200
    overlayList.append(resized_image)
    
# print(len(overlayList))
#Initialize hand detector class
detector = htm.HandDetector(detectionCon=0.75 , maxHands=1)
tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    
    #create a list of the landmark that we detect
    lmList = detector.findPosition(img, draw=False)
    
    if len(lmList) != 0:
        fingers = []
        
        #Thumb
        if lmList[tipIds[0]][1] > lmList[tipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
            
        # 4 Fingers
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
                
        # print(fingers)
        totalFingers = fingers.count(1)
        # print(totalFingers)
    
        h,w,c = overlayList[totalFingers-1].shape
        img[0:h, 0:w] = overlayList[totalFingers-1]
        
        cv2.rectangle(img, (20, 225), (170,425), 
                      (0,255,0), cv2.FILLED)
        cv2.putText(img, str(totalFingers), (45,375), cv2.FONT_HERSHEY_PLAIN,
                    10, (255,0,0), 25)
    
    #Calculate the frame rate
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    
    cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (255,0,0),3)

    # if not success:
    #     print('Error reading frame')
    #     break

    # frame = detector.findHands(frame)

    # Show the processed image
    cv2.imshow('finger counter', img)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
