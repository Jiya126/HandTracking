import cv2
import numpy as np
import time
import os
import handTrackModule as htm

#######################
brushThickness = 15
eraserThickness = 100
########################

drawColor = (255, 0, 255)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.handDetector()
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

while True:

    # 1. Import image
    success, img = cap.read()
    img = cv2.flip(img, 1)


    # 2. Find Hand Landmarks
    img = detector.detectHands(img)
    lmList = detector.detectPos(img, drawY=False)

    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        
        fingers = detector.detectUp()
        
        # Eraser mode
        if fingers[1] and fingers[2] and fingers[3]:
            xp, yp = 0, 0
            drawColor = (0,0,0)
            cv2.rectangle(img, (x1, y1 - 35), (x2, y2 + 35), drawColor, cv2.FILLED)
            if xp == 0 and yp == 0:
                xp, yp = x1, y1
        
            cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)


        # Drawing Mode
        if fingers[1] and fingers[2] == False:
            drawColor = (255,0,255)
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
          
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

            xp, yp = x1, y1

        # Hold mode
        if fingers[1] and fingers[2] and fingers[3] == False:
            xp, yp = 0,0
            drawColor = (255,255,255)
        



       
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)

    img = cv2.bitwise_and(img,imgInv)
    img = cv2.bitwise_or(img,imgCanvas)

    cv2.imshow("Image", img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()