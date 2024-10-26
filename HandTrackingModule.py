import cv2
import mediapipe as mp
import time

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

        # Initialize Mediapipe Hands and drawing utility
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

    def findPosition(self, img, handNo=0, draw=True):
        """
        Find the position of landmarks for the specified hand.

        Parameters:
        - img: The input image (BGR).
        - handNo (int): The hand number to detect (0 for the first hand).
        - draw (bool): Whether to draw circles on landmark points.

        Returns:
        - lmList: A list of landmarks (id, x, y) for the specified hand.
        """
        lmList = []
        if self.results.multi_hand_landmarks:
            try:
                myHand = self.results.multi_hand_landmarks[handNo]
                h, w, c = img.shape
                for id, lm in enumerate(myHand.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
            except IndexError:
                print(f"Hand number {handNo} not found.")
        
        return lmList
