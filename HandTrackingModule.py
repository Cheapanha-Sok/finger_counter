import cv2
import mediapipe as mp

class HandDetector:
    """
    A class for hand detection using Mediapipe.

    Parameters:
    - mode (bool): Whether to treat the input images as static images or not.
    - maxHands (int): Maximum number of hands to detect.
    - detectionCon (float): Minimum detection confidence threshold.
    - trackCon (float): Minimum tracking confidence threshold.
    """

    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode,
                                        max_num_hands=self.maxHands,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        """
        Process the image to find hands and optionally draw landmarks.

        Parameters:
        - img: The input image (BGR).
        - draw (bool): Whether to draw hand landmarks on the image.

        Returns:
        - img: The processed image with or without landmarks drawn.
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img
    
    def findPosition(self, img, handNo=0):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)
                lmList.append([id, cx, cy])
        return lmList
