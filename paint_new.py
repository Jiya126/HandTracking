import cv2
import numpy as np
import time
import os
import handTrackModule as htm
from tkinter import *
from PIL import Image, ImageTk

#######################
brushThickness = 15
eraserThickness = 50
drawColor = (255, 0, 255)
eraserColor = (0,0,0)
holdColor = (255,255,255)
strip_height = 100
button_x = 1150
button_y = 10
overlay_x = 10  
overlay_y = 10
########################

def paintScreen():
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    detector = htm.handDetector()
    xp, yp = 0, 0
    imgCanvas = np.zeros((720, 1280, 3), np.uint8)

    overlay_image = cv2.imread('Group 7.png')
    overlay_width = 200  
    overlay_height = 150 
    overlay_image = cv2.resize(overlay_image, (overlay_width, overlay_height))


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
            if fingers[1] and fingers[2]:
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

        
        img[overlay_y:overlay_y+overlay_height, overlay_x:overlay_x+overlay_width] = overlay_image

        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                 b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')