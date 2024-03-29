import cv2
import numpy as np
import time
import os
import handTrackModule as htm

#######################
brushThickness = 15
eraserThickness = 50
drawColor = (255, 0, 255)
eraserColor = (0,0,0)
holdColor = (255,255,255)
rectangle_height = 30
########################

def paintScreen():
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
        lmList = detector.detectPos(img)

        if len(lmList) != 0:
            x1, y1 = lmList[8][1:]
            x2, y2 = lmList[12][1:]

            
            fingers = detector.detectUp()
            
            # Eraser mode
            if fingers[1] and fingers[2] and fingers[3]:
                xp, yp = 0, 0
                cv2.rectangle(img, (x1, y1 - eraserThickness), (x2, y2 + eraserThickness), eraserColor, cv2.FILLED)
                if xp == 0 and yp == 0:
                    xp, yp = x1, y1
            
                cv2.line(imgCanvas, (xp, yp), (x1, y1), eraserColor, eraserThickness)


            # Drawing Mode
            if fingers[1] and fingers[2] == False:
                cv2.circle(img, (x1, y1), brushThickness, drawColor, cv2.FILLED)
            
                if xp == 0 and yp == 0:
                    xp, yp = x1, y1

                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

                xp, yp = x1, y1

            # Hold mode
            if fingers[1] == False:
                xp, yp = 0,0

        
        imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)

        img = cv2.bitwise_and(img,imgInv)
        img = cv2.bitwise_or(img,imgCanvas)


        #adding white foreground, top
        frame_width = int(cap.get(3))
        white_rect = np.ones((rectangle_height, frame_width, 3), dtype=np.uint8) * 255
        overlay = cv2.imread('Group 6.png')
        overlay = cv2.resize(overlay, (frame_width, rectangle_height))
        combined_frame = np.vstack((white_rect, overlay))
        result_frame = img.copy()
        result_frame[0:rectangle_height, 0:frame_width] = combined_frame


        ret, buffer = cv2.imencode('.jpg', result_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                 b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')